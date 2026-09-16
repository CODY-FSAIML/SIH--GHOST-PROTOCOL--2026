| Model | Status | Threshold | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | Confusion | Pos pred | Prevalence | Onset recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Logistic Regression baseline (H3 State) | evaluated | 0.50 | 0.5526 | 1.0000 | 0.7118 | 0.9868 | 0.5306 | 0.6027 | TN=2, FP=149, FN=0, TP=184 | 333 | 0.5493 | 8/8 |
| Transformer H10 State + Velocity | evaluated | 0.50 | 0.5817 | 0.6576 | 0.6173 | 0.6042 | 0.6349 | 0.7071 | TN=57, FP=87, FN=63, TP=121 | 208 | 0.5610 | 6/8 |
| Transformer H10 State-only | unavailable | - | - | - | - | - | - | - | No saved compatible state-only Transformer checkpoint exists; evaluation did not retrain it. | - | - | - |
| Learned latent World Model | evaluated | 0.50 | 0.5541 | 0.6685 | 0.6059 | 0.6875 | 0.5763 | 0.6667 | TN=45, FP=99, FN=61, TP=123 | 222 | 0.5610 | 7/8 |
| Transformer + GraphSAGE fusion | evaluated | 0.50 | 0.5967 | 0.9891 | 0.7444 | 0.8542 | 0.6148 | 0.6608 | TN=21, FP=123, FN=2, TP=182 | 305 | 0.5610 | 7/8 |
