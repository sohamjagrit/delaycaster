"""DelayCast training script — runs as one AzureML command job per model.

Reads the full feature Parquet snapshot from the ADLS-backed datastore, does a
**time-based** train/test split (predicting delays is a forecasting task — a random
split would leak the future), trains one model with class-imbalance handling, logs
all metrics + the fitted *pipeline* to MLflow, and prints the primary metrics.

Why a Pipeline (preprocessor + estimator) rather than pre-encoded features: the
pickled pipeline carries its own one-hot encoder, so the Streamlit app feeds raw
columns (carrier="AA", origin="DFW", ...) straight in — no separate encoder to keep
in sync, no training/serving skew. The encoder mirrors src/features.FEATURE_COLUMNS.

Inside an AzureML job, MLflow's tracking URI is already wired to the workspace, so
mlflow.log_* lands in the experiment automatically — no URI to set here.
"""

import argparse
import os

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Must match src/features.py exactly (train/serve contract).
CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "distance", "dep_hour", "day_of_week", "month", "is_weekend",
    "carrier_delay_rate", "origin_delay_rate", "dest_delay_rate", "carrier_origin_delay_rate",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "delayed"


def build_estimator(model: str, scale_pos_weight: float):
    """One estimator per --model. Every one handles the ~20% class imbalance."""
    if model == "lr":
        # n_jobs=1: binary lbfgs doesn't parallelize, and n_jobs>1 triggers a joblib
        # memmap of the full sparse X to disk (overflows the container temp). No upside.
        return LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=1)
    if model == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=18, min_samples_leaf=50,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
    if model == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, tree_method="hist",
            n_jobs=-1, random_state=42, eval_metric="aucpr",
        )
    if model == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=600, num_leaves=64, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
    raise ValueError(f"unknown model: {model}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to the mounted train_parquet folder")
    p.add_argument("--model", required=True, choices=["lr", "rf", "xgb", "lgbm"])
    p.add_argument("--test-start", default="2024-07-01", help="rows on/after this date = test")
    p.add_argument("--model-out", default="outputs/model",
                   help="folder to write the fitted model pickle into (an AzureML job output)")
    args = p.parse_args()

    # joblib/loky memmaps big arrays to a temp dir to share with parallel workers. The
    # container default (/dev/shm, ~64 MB) overflows on a 16.7M-row sparse matrix →
    # "No space left on device". Redirect it to the job working dir, which sits on the
    # large temp SSD. Fixes RF (n_jobs>1) without touching its parallelism.
    joblib_tmp = os.path.join(os.getcwd(), "joblib_tmp")
    os.makedirs(joblib_tmp, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = joblib_tmp

    mlflow.set_tag("model_family", args.model)
    mlflow.log_param("model", args.model)
    mlflow.log_param("test_start", args.test_start)
    mlflow.log_param("split", "time-based")

    # --- load full dataset (pyarrow reads the whole folder) ---
    df = pd.read_parquet(args.data, columns=FEATURES + [TARGET, "FlightDate"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    cut = pd.Timestamp(args.test_start)
    train_df = df[df["FlightDate"] < cut]
    test_df = df[df["FlightDate"] >= cut]
    mlflow.log_metric("n_train", len(train_df))
    mlflow.log_metric("n_test", len(test_df))
    print(f"train={len(train_df):,}  test={len(test_df):,}")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    pos = max(int(y_train.sum()), 1)
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    mlflow.log_metric("train_pos_rate", pos / len(y_train))

    # --- pipeline: one-hot the categoricals, pass numerics through ---
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", build_estimator(args.model, scale_pos_weight))])

    print(f"fitting {args.model} ...")
    pipe.fit(X_train, y_train)

    # --- evaluate ---
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "recall": recall_score(y_test, pred),            # PRIMARY
        "pr_auc": average_precision_score(y_test, proba),  # PRIMARY (imbalance-robust)
        "precision": precision_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),         # context only
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        print(f"  {k:10s} = {v:.4f}")

    # --- persist the fitted pipeline to the job output (NOT mlflow.log_model) ---
    # AzureML's MLflow tracking server doesn't implement mlflow 3.x's "logged models"
    # endpoint, so mlflow.sklearn.log_model returns 404. Sidestep it: dump the sklearn
    # Pipeline to the job's output folder with joblib. The control plane registers that
    # output into the AzureML Model Registry, and the Streamlit app loads this same pickle.
    os.makedirs(args.model_out, exist_ok=True)
    model_path = os.path.join(args.model_out, "model.pkl")
    joblib.dump(pipe, model_path)
    print(f"saved model to {model_path}")
    print("done.")


if __name__ == "__main__":
    main()
