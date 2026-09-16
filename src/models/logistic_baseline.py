"""
Logistic Regression Forecasting Baseline (Horizon t+1)
======================================================
First-stage temporal attack forecasting model mapping 3 past observable
Network States [S_(t-2), S_(t-1), S_t] (48 features) to future attack occurrence Y_(t+1).

Anti-Leakage & Scientific Rigor Guarantees:
- Model features are purely observable traffic aggregations from windows t-2, t-1, t.
- Zero future information, zero label leakage, zero timestamp/IP identifiers in input.
- Preprocessing scaler (StandardScaler) is strictly fit on Training data only;
  Validation and Test sets are transformed using the pre-fit parameters.
- Validation threshold calibration detects single-class distributions and explicitly
  flags statistical invalidity rather than manufacturing spurious thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from ..features.network_state import MacroNetworkState, MACRO_FEATURE_NAMES
from ..data.temporal_targets import ForecastingSample


def get_48_feature_names() -> List[str]:
    """
    Generate the ordered list of 48 feature names representing the 3-window history.
    Format:
        t-2_<feature_name>, ..., t-1_<feature_name>, ..., t_<feature_name>
    """
    names: List[str] = []
    for step in ("t-2", "t-1", "t"):
        for feat in MACRO_FEATURE_NAMES:
            names.append(f"{step}_{feat}")
    return names


def build_baseline_dataset(
    samples: Sequence[ForecastingSample],
    states_lookup: Dict[int, MacroNetworkState],
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Construct the (N, 48) feature matrix X and (N,) target vector y for horizon t+k.

    Parameters:
        samples: Sequence of ForecastingSample objects.
        states_lookup: Mapping from window_idx -> MacroNetworkState.
        horizon: Target forecasting step ahead (default: 1 for t+1).

    Returns:
        X: 2D numpy array of shape (N, 48), dtype float64.
        y: 1D numpy array of shape (N,), dtype int64.
        feature_names: List of 48 feature names.
        sample_meta: List of diagnostic sample metadata dictionaries.
    """
    feature_names = get_48_feature_names()

    if not samples:
        return (
            np.empty((0, 48), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            feature_names,
            [],
        )

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    for s in samples:
        # History indices: (t-2, t-1, t)
        h_indices = s.history_window_indices
        if len(h_indices) != 3:
            raise ValueError(f"Sample {s.sample_idx} has history length {len(h_indices)}, expected 3.")

        # Ensure all 3 states are present in lookup
        for idx in h_indices:
            if idx not in states_lookup:
                raise KeyError(f"Window index {idx} missing from states_lookup.")

        s_t_minus_2 = states_lookup[h_indices[0]].vector
        s_t_minus_1 = states_lookup[h_indices[1]].vector
        s_t = states_lookup[h_indices[2]].vector

        # Concatenate in strict chronological order: [S_(t-2), S_(t-1), S_t]
        x_row = np.concatenate([s_t_minus_2, s_t_minus_1, s_t]).astype(np.float64)
        if len(x_row) != 48:
            raise ValueError(f"Concatenated vector length is {len(x_row)}, expected 48.")

        target_y = s.Y.get(horizon)
        if target_y is None:
            raise KeyError(f"Sample {s.sample_idx} missing target horizon {horizon}.")

        x_list.append(x_row)
        y_list.append(int(target_y))

        # Attach non-leaking diagnostic sample metadata
        meta_dict = {
            "sample_idx": s.sample_idx,
            "anchor_window_idx": s.anchor_window_idx,
            "history_window_indices": s.history_window_indices,
            "target_window_indices": s.target_window_indices,
            "is_onset_t1": s.target_metadata.get("anchor_is_onset_t1", False),
            "anchor_attack_flows": s.target_metadata.get("anchor_attack_flows", 0),
        }
        meta_list.append(meta_dict)

    X = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)

    return X, y, feature_names, meta_list


@dataclass
class EvaluationReport:
    """Detailed evaluation metrics container."""
    partition_name: str
    sample_count: int
    positive_count: int
    negative_count: int
    predicted_positive_count: int
    predicted_negative_count: int
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    onset_count: int
    onset_detected_count: int
    validity_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partition": self.partition_name,
            "samples": self.sample_count,
            "positives": self.positive_count,
            "negatives": self.negative_count,
            "pred_positives": self.predicted_positive_count,
            "pred_negatives": self.predicted_negative_count,
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "tp": self.true_positives,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "onsets": self.onset_count,
            "onsets_detected": self.onset_detected_count,
            "validity_notes": "; ".join(self.validity_notes),
        }


class LogisticRegressionBaseline:
    """
    Regularized Logistic Regression Baseline for t+1 attack forecasting.
    Enforces training-only scaler fitting, threshold calibration with single-class
    fallback, and coefficient interpretation.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = get_48_feature_names()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        """
        Fit the standard scaler and logistic regression model strictly on training data.
        """
        if len(X_train) == 0:
            raise ValueError("Cannot fit on empty training feature matrix.")

        # 1. Fit scaler ONLY on training data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # 2. Fit Logistic Regression
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of attack P(Y=1).
        Applies pre-fit scaler transform only.
        """
        if not self.is_fitted or self.scaler is None or self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        if len(X) == 0:
            return np.empty((0,), dtype=np.float64)

        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)

        # Binary classification: class 1 probability
        if probs.shape[1] == 2:
            return probs[:, 1]
        elif probs.shape[1] == 1:
            # Model only observed 1 class during training (edge case)
            single_class = self.model.classes_[0]
            return np.full(len(X), float(single_class == 1), dtype=np.float64)
        else:
            raise ValueError(f"Unexpected predict_proba shape: {probs.shape}")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions Y_hat in {0, 1} based on decision threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def calibrate_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, bool, str]:
        """
        Evaluate thresholds on validation partition.

        Checks whether validation contains both classes:
        - If validation contains only positive examples or only negative examples,
          explicitly flags that calibration is statistically invalid and retains 0.5.
        - If valid, searches thresholds in [0.05, 0.95] for optimal metric score.

        Returns:
            (selected_threshold, is_calibration_valid, explanation_message)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibrating threshold.")

        if len(y_val) == 0:
            return 0.5, False, "Validation set is empty."

        unique_classes = np.unique(y_val)
        if len(unique_classes) < 2:
            msg = (
                f"Validation set contains only class {unique_classes[0]} ({len(y_val)} samples). "
                "Threshold calibration is statistically invalid without both positive and negative examples. "
                "Retaining provisional default threshold 0.5."
            )
            return 0.5, False, msg

        probs = self.predict_proba(X_val)
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thresh = 0.5
        best_score = -1.0

        for t in thresholds:
            preds = (probs >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, preds, zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if score > best_score:
                best_score = score
                best_thresh = float(t)

        msg = f"Calibrated optimal threshold {best_thresh:.3f} achieving validation {metric}={best_score:.4f}."
        return best_thresh, True, msg

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        partition_name: str = "Test",
        sample_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> EvaluationReport:
        """
        Comprehensive evaluation returning Precision, Recall, F1, FPR, Confusion Matrix,
        ROC-AUC, PR-AUC, and onset detection count.
        """
        if len(X) == 0:
            return EvaluationReport(
                partition_name=partition_name,
                sample_count=0,
                positive_count=0,
                negative_count=0,
                predicted_positive_count=0,
                predicted_negative_count=0,
                threshold=threshold,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                fpr=0.0,
                true_negatives=0,
                false_positives=0,
                false_negatives=0,
                true_positives=0,
                roc_auc=None,
                pr_auc=None,
                onset_count=0,
                onset_detected_count=0,
                validity_notes=["Empty partition"],
            )

        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        pred_pos_count = int(np.sum(preds == 1))
        pred_neg_count = int(np.sum(preds == 0))

        # Precision, Recall, F1
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))

        # Confusion Matrix
        # Handle cases where y has only 1 class by specifying labels=[0, 1]
        cm = confusion_matrix(y, preds, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        # False Positive Rate: FP / (FP + TN)
        if (fp + tn) > 0:
            fpr = float(fp / (fp + tn))
        else:
            fpr = 0.0

        # ROC-AUC & PR-AUC (require both classes to be mathematically defined)
        validity_notes: List[str] = []
        if len(np.unique(y)) >= 2:
            roc_auc = float(roc_auc_score(y, probs))
            pr_auc = float(average_precision_score(y, probs))
        else:
            roc_auc = None
            pr_auc = None
            validity_notes.append("ROC-AUC/PR-AUC undefined: single-class partition")

        # Attack-Onset Analysis (y_t=0 AND Y_(t+1)=1)
        onset_count = 0
        onset_detected = 0
        if sample_meta:
            for idx, meta in enumerate(sample_meta):
                if meta.get("is_onset_t1", False):
                    onset_count += 1
                    if preds[idx] == 1:
                        onset_detected += 1

        if onset_count == 0:
            validity_notes.append("Zero attack-onset events in partition")

        if neg_count < 10:
            validity_notes.append(f"Small negative sample count (N_neg={neg_count}) limits FPR/Precision confidence")

        return EvaluationReport(
            partition_name=partition_name,
            sample_count=len(y),
            positive_count=pos_count,
            negative_count=neg_count,
            predicted_positive_count=pred_pos_count,
            predicted_negative_count=pred_neg_count,
            threshold=threshold,
            precision=prec,
            recall=rec,
            f1=f1,
            fpr=fpr,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            true_positives=tp,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            onset_count=onset_count,
            onset_detected_count=onset_detected,
            validity_notes=validity_notes,
        )

    def get_feature_coefficients(self) -> pd.DataFrame:
        """
        Return Logistic Regression feature coefficients mapped to:
        - history step: t-2, t-1, t
        - macro feature name
        Sorted by absolute magnitude.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")

        coefs = self.model.coef_[0]
        feature_names = self.feature_names

        rows = []
        for name, coef in zip(feature_names, coefs):
            parts = name.split("_", 1)
            step = parts[0]
            macro_name = parts[1] if len(parts) > 1 else ""
            rows.append({
                "feature": name,
                "history_step": step,
                "macro_feature": macro_name,
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
            })

        df_coef = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        return df_coef
