# DelayCast ✈️ — Flight Delay Prediction (Operations Decision Support)

Predicts whether a US domestic flight will arrive **more than 15 minutes late**
(the US DOT definition of a delayed flight) — the kind of go/no-go signal an airline
Integrated Operations Center (IOC) uses to catch at-risk flights before delays cascade.

Built end-to-end on the Azure data stack: **ADLS Gen2 → Azure Databricks (PySpark) →
AzureML + MLflow → Streamlit**.

## Architecture

```
BTS source (URLs)
   │  ingest
   ▼
ADLS Gen2  ──►  Databricks / PySpark  ──►  Delta feature table  ──►  AzureML + MLflow
(raw CSVs)      (clean + features)         (features/)               (train, track,
                                                                      register champion)
                                                                            │
                                                                            ▼
                                                                       Streamlit app
                                                                       (live demo)
```

## Pipeline stages

| Stage | Tools | Notebook / file |
|---|---|---|
| Data engineering | Azure Databricks + PySpark, ADLS Gen2 (Delta) | `notebooks/01_ingest`, `02_eda`, `03_features` |
| Data science | AzureML + MLflow | `notebooks/04_train_evaluate` |
| Serving / demo | Streamlit (Community Cloud) | `app/streamlit_app.py` |

## Key design decisions

- **Target:** `delayed = 1 if ARR_DELAY > 15 min else 0`.
- **Primary metric:** recall (missing a real delay costs more than a false alarm);
  also report precision, PR-AUC, and accuracy (for context — class is ~20% positive).
- **Leakage rule:** only features known at scheduled departure. No `DEP_DELAY`, no
  delay-cause columns (`CARRIER/WEATHER/NAS_DELAY`), no actual times.
- **Serving feature lookup:** `03_features` writes a small table of latest delay rates
  per carrier/airport so the app can rebuild features at predict time — avoids
  training/serving skew.

## Status

Repo scaffolded; Stage 1 (ingest) in progress. See `CLAUDE.md` for the full build plan.

## Reproduce

_To be filled in as stages complete (cluster config, env vars, deploy steps)._
