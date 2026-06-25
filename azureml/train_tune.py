"""LightGBM tuning + training-diagnostics job for DelayCast (one job = one config).

Differences vs train.py (the Stage-2 baseline trainer):
  * 3-way TIME split — train / validation / test. Hyperparameters are judged on the
    VALIDATION set; the TEST set is scored once for an honest final number. (A random
    k-fold split would leak the future into the past — never do that for forecasting.)
  * Validation-based EARLY STOPPING — `n_estimators` becomes an upper bound; the best
    iteration is chosen by the validation metric, so we don't over/under-fit the count.
  * TRAINING-LOSS CURVES — per-iteration train & valid `binary_logloss` and `auc` are
    logged to MLflow as stepped metrics, so you can literally watch over/underfitting.
  * THRESHOLD SWEEP — recall/precision logged across thresholds on validation, making
    the "recall is a threshold lever, not just a model lever" point concrete.
  * Every key hyperparameter is a CLI arg, so the control plane sweeps configs.

Saves the fitted sklearn Pipeline (OneHot + LGBM) as model.pkl to the job output, same
contract as Stage 2 (so the Streamlit app loads it unchanged).
"""

import argparse
import os

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.compose import ColumnTransformer
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

# Must match src/features.FEATURE_COLUMNS (train/serve contract).
CATEGORICAL = ["carrier", "origin", "dest"]
NUMERIC = [
    "distance", "dep_hour", "day_of_week", "month", "is_weekend",
    "carrier_delay_rate", "origin_delay_rate", "dest_delay_rate", "carrier_origin_delay_rate",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "delayed"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--valid-start", default="2024-01-01", help="rows >= this and < test-start = validation")
    p.add_argument("--test-start", default="2024-07-01", help="rows >= this = test (scored once)")
    p.add_argument("--model-out", default="outputs/model")
    p.add_argument("--tag", default="run", help="human label for this config (job/run name)")
    # --- hyperparameters (one job = one config) ---
    p.add_argument("--n-estimators", type=int, default=3000, help="UPPER bound; early stopping picks best")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-lambda", type=float, default=0.0)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    # joblib temp → big SSD (see train.py note); avoids /dev/shm overflow.
    jt = os.path.join(os.getcwd(), "joblib_tmp")
    os.makedirs(jt, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = jt

    mlflow.set_tag("phase", "tune")
    mlflow.set_tag("config_tag", args.tag)
    for k, v in vars(args).items():
        if k != "data":
            mlflow.log_param(k, v)

    # --- load + 3-way time split ---
    df = pd.read_parquet(args.data, columns=FEATURES + [TARGET, "FlightDate"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    vcut, tcut = pd.Timestamp(args.valid_start), pd.Timestamp(args.test_start)
    tr = df[df["FlightDate"] < vcut]
    va = df[(df["FlightDate"] >= vcut) & (df["FlightDate"] < tcut)]
    te = df[df["FlightDate"] >= tcut]
    for nm, d in [("train", tr), ("valid", va), ("test", te)]:
        mlflow.log_metric(f"n_{nm}", len(d))
    print(f"[{args.tag}] train={len(tr):,} valid={len(va):,} test={len(te):,}")

    Xtr, ytr = tr[FEATURES], tr[TARGET]
    Xva, yva = va[FEATURES], va[TARGET]
    Xte, yte = te[FEATURES], te[TARGET]

    # --- one-hot (fit on train only), then transform all splits ---
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    pre.fit(Xtr)
    Xtr_t, Xva_t, Xte_t = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)

    clf = LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        class_weight="balanced",     # ~20% positives — same imbalance handling as Stage 2
        n_jobs=-1,
        random_state=42,
    )

    # --- fit with eval_set so we get TRAINING-LOSS CURVES + early stopping ---
    evals = {}
    clf.fit(
        Xtr_t, ytr,
        eval_set=[(Xtr_t, ytr), (Xva_t, yva)],
        eval_names=["train", "valid"],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[
            early_stopping(args.early_stopping_rounds),
            log_evaluation(50),
            record_evaluation(evals),
        ],
    )
    best_iter = clf.best_iteration_ or args.n_estimators
    mlflow.log_metric("best_iteration", best_iter)

    # per-iteration curves -> MLflow (this is the "training loss" view)
    for split in ("train", "valid"):
        for metric, values in evals.get(split, {}).items():
            for step, val in enumerate(values):
                mlflow.log_metric(f"{split}_{metric}", val, step=step)

    # --- metrics on valid (for tuning) and test (final, honest) ---
    def metrics(X, y, thr=0.5):
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= thr).astype(int)
        return {
            "recall": recall_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "pr_auc": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba),
            "accuracy": accuracy_score(y, pred),
            "f1": f1_score(y, pred, zero_division=0),
        }

    vm = metrics(Xva_t, yva)
    tm = metrics(Xte_t, yte)
    for k, v in vm.items():
        mlflow.log_metric(f"valid_{k}", v)
    for k, v in tm.items():
        mlflow.log_metric(f"test_{k}", v)
    print(f"[{args.tag}] VALID {vm}")
    print(f"[{args.tag}] TEST  {tm}")

    # --- threshold sweep on validation: recall is a threshold lever ---
    pv = clf.predict_proba(Xva_t)[:, 1]
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = (pv >= thr).astype(int)
        mlflow.log_metric("sweep_recall", recall_score(yva, pred), step=int(thr * 100))
        mlflow.log_metric("sweep_precision", precision_score(yva, pred, zero_division=0), step=int(thr * 100))

    # --- save the fitted pipeline (OneHot + LGBM) for serving ---
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    os.makedirs(args.model_out, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.model_out, "model.pkl"))
    print("done.")


if __name__ == "__main__":
    main()
