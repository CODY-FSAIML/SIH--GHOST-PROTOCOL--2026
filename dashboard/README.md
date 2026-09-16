# NetGuardAI Demo Dashboard

Run from the project root with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The dashboard performs inference only. It loads the verified temporal Transformer,
GraphSAGE, fusion checkpoint, and temporal scaler. Scenario 42 and Scenario 50 are
included as selectable inputs; compatible raw CTU-13 CSV uploads are also accepted.

The trajectory section uses the available model score as a signal for an explicitly
labeled `demo/simulation` transition layer. Those transition values are not learned
or calibrated probabilities. MITRE mapping remains an integration point and is not
fabricated in the demo.
