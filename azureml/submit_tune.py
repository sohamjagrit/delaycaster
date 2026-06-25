"""Hyperparameter sweep for the LightGBM champion — run from the Mac (repo root).

Submits one AzureML job per config (all running train_tune.py): a `baseline` that
reproduces the Stage-2 LightGBM, plus a few tuned candidates. Each job:
  * uses the same 3-way time split,
  * logs training-loss curves + a threshold sweep,
  * scores VALIDATION (for picking) and TEST (for the honest final number).

After all finish, this compares configs on **validation PR-AUC**, then prints the
winner's **test** metrics next to the baseline's — tuned-vs-untuned, no test leakage.
Job names are intuitive (e.g. `lgbm-reg-heavy-06221530`), not random GUIDs.

    .venv/bin/python azureml/submit_tune.py
    .venv/bin/python azureml/submit_tune.py --no-wait
"""

import argparse
import os
import time

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

SUBSCRIPTION = os.environ.get("AZ_SUB", "b925acee-9622-43a6-b56c-61018ef146b9")
RESOURCE_GROUP = "delaycaster-rg"
WORKSPACE = "delaycaster-aml"

DATA_URI = "azureml://datastores/delaycaster_blob/paths/features/train_parquet"
EXPERIMENT = "delaycast-tune"        # separate experiment so tuning runs don't mix with Stage 2
VALID_START = "2024-01-01"
TEST_START = "2024-07-01"
HERE = os.path.dirname(os.path.abspath(__file__))

# baseline = the Stage-2 LightGBM (so we can measure what tuning actually buys).
# The rest probe the usual levers: learning rate, tree width (num_leaves), leaf size
# (min_child_samples), and L2 (reg_lambda). n_estimators is large everywhere because
# early stopping on the validation set chooses the real count.
CONFIGS = {
    "baseline":  dict(learning_rate=0.05, num_leaves=64,  min_child_samples=20,  reg_lambda=0.0),
    "slow-deep": dict(learning_rate=0.03, num_leaves=128, min_child_samples=50,  reg_lambda=1.0),
    "reg-heavy": dict(learning_rate=0.05, num_leaves=64,  min_child_samples=200, reg_lambda=5.0),
    "wide":      dict(learning_rate=0.05, num_leaves=256, min_child_samples=100, reg_lambda=2.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), help="subset of config names")
    args = ap.parse_args()

    ml = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print("workspace:", ml.workspace_name)
    ts = time.strftime("%m%d%H%M")

    submitted = {}
    for tag in args.configs:
        hp = CONFIGS[tag]
        flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in hp.items())
        job = command(
            code=HERE,
            command=(
                "python train_tune.py --data ${{inputs.data}} "
                f"--valid-start {VALID_START} --test-start {TEST_START} "
                f"--tag {tag} {flags} --model-out ${{outputs.model_out}}"
            ),
            inputs={"data": Input(type="uri_folder", path=DATA_URI)},
            outputs={"model_out": Output(type=AssetTypes.URI_FOLDER)},
            environment="delaycast-train@latest",
            compute="cpu-train",
            experiment_name=EXPERIMENT,
            name=f"lgbm-{tag}-{ts}",        # intuitive, unique job name
            display_name=f"lgbm-{tag}",
        )
        run = ml.jobs.create_or_update(job)
        submitted[tag] = run.name
        print(f"submitted {tag:10s} -> {run.name}\n   {run.studio_url}")

    if args.no_wait:
        print("\n--no-wait: see AzureML Studio (experiment 'delaycast-tune').")
        return

    terminal = {"Completed", "Failed", "Canceled"}
    while True:
        statuses = {t: ml.jobs.get(n).status for t, n in submitted.items()}
        print(statuses)
        if all(s in terminal for s in statuses.values()):
            break
        time.sleep(60)

    # --- compare on VALIDATION pr_auc; report TEST for tuned vs baseline ---
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(ml.workspaces.get(WORKSPACE).mlflow_tracking_uri)
    rows = []
    for tag, name in submitted.items():
        if statuses[tag] != "Completed":
            print(f"skip {tag}: {statuses[tag]}")
            continue
        m = mlflow.get_run(name).data.metrics
        rows.append({
            "config": tag, "run_id": name, "best_iter": m.get("best_iteration"),
            "valid_pr_auc": m.get("valid_pr_auc"), "valid_recall": m.get("valid_recall"),
            "test_pr_auc": m.get("test_pr_auc"), "test_recall": m.get("test_recall"),
            "test_precision": m.get("test_precision"), "test_roc_auc": m.get("test_roc_auc"),
        })
    if not rows:
        print("no completed runs — check logs in AzureML Studio.")
        return

    board = pd.DataFrame(rows).sort_values("valid_pr_auc", ascending=False)
    print("\n=== ranked by VALIDATION pr_auc (tuning metric) ===")
    print(board[["config", "best_iter", "valid_pr_auc", "valid_recall",
                 "test_pr_auc", "test_recall", "test_precision"]].to_string(index=False))

    best = board.iloc[0]
    base = board[board["config"] == "baseline"]
    print(f"\n>>> WINNER (valid pr_auc): {best['config']}")
    if not base.empty:
        b = base.iloc[0]
        print("TUNED vs UNTUNED on TEST:")
        print(f"  pr_auc : {b['test_pr_auc']:.4f} (baseline) -> {best['test_pr_auc']:.4f} ({best['config']})")
        print(f"  recall : {b['test_recall']:.4f} (baseline) -> {best['test_recall']:.4f} ({best['config']})")
    print("\nTo promote the winner, register its model_out as delaycast-champion (see submit_train.py).")


if __name__ == "__main__":
    main()
