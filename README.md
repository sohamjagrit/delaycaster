# DelayCaster

Predicts whether a US domestic flight will arrive **more than 15 minutes late**
(the US DOT definition of a delayed flight) 

Built end-to-end on the Azure data stack: **ADLS Gen2 → Azure Databricks (PySpark) →
AzureML + MLflow → Streamlit**.

## Architecture

```
BTS source (URLs)
   │  ingest
   ▼
ADLS Gen2  ──►  Databricks / PySpark  ──►  Delta feature table  ──►  AzureML + MLflow
(raw CSVs)      (clean + features)         (features/)               (train, track,
                                                                      register best model)
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



