"""Local control plane for Stage 2 — run this from your Mac (repo root), NOT Databricks.

Why local: you're already `az login`'d (so DefaultAzureCredential just works) and the
repo — including azureml/train.py — is on disk (so the `code` upload resolves). The
orchestration needs no Spark; the heavy training runs on AzureML's cpu-train cluster.

Run:
    pip install azure-ai-ml azure-identity mlflow azureml-mlflow pandas
    python azureml/submit_train.py            # submit all 4 + compare + register
    python azureml/submit_train.py --no-wait  # just submit, don't block

It submits one AzureML command job per model (lr/rf/xgb/lgbm), waits, compares on
recall + PR-AUC, and registers the winner as `delaycast-champion`.
"""

import argparse
import os
import time

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

SUBSCRIPTION = os.environ.get("AZ_SUB", "b925acee-9622-43a6-b56c-61018ef146b9")
RESOURCE_GROUP = "delaycaster-rg"
WORKSPACE = "delaycaster-aml"

# Read via the account-key blob datastore (delaycaster_blob) rather than the identity-based
# ADLS datastore: AmlCompute has no managed identity, so identity-based mounts fail with
# NoIdentityOnCompute. Same files, same container — just key auth instead of compute identity.
DATA_URI = "azureml://datastores/delaycaster_blob/paths/features/train_parquet"
EXPERIMENT = "delaycast"
TEST_START = "2024-07-01"          # time split: train < this date, test on/after
MODELS = ["lr", "rf", "xgb", "lgbm"]
HERE = os.path.dirname(os.path.abspath(__file__))   # the azureml/ folder (holds train.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true", help="submit only, don't poll/compare")
    ap.add_argument("--models", nargs="+", default=MODELS, help="subset of models to run")
    args = ap.parse_args()

    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print("Connected to workspace:", ml_client.workspace_name)

    # --- submit one job per model ---
    submitted = {}
    for m in args.models:
        job = command(
            code=HERE,                                  # uploads train.py
            command=(
                "python train.py --data ${{inputs.data}} "
                f"--model {m} --test-start {TEST_START} "
                "--model-out ${{outputs.model_out}}"
            ),
            inputs={"data": Input(type="uri_folder", path=DATA_URI)},
            outputs={"model_out": Output(type=AssetTypes.URI_FOLDER)},
            environment="delaycast-train@latest",
            compute="cpu-train",
            experiment_name=EXPERIMENT,
            display_name=f"train-{m}",
        )
        run = ml_client.jobs.create_or_update(job)
        submitted[m] = run.name
        print(f"submitted {m:5s} -> {run.name}\n   {run.studio_url}")

    if args.no_wait:
        print("\n--no-wait: jobs submitted. Watch them in AzureML Studio.")
        return

    # --- wait for all to reach a terminal state ---
    terminal = {"Completed", "Failed", "Canceled"}
    while True:
        statuses = {m: ml_client.jobs.get(n).status for m, n in submitted.items()}
        print(statuses)
        if all(s in terminal for s in statuses.values()):
            break
        time.sleep(60)

    # --- compare on recall, then PR-AUC, over the runs that completed ---
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(ml_client.workspaces.get(WORKSPACE).mlflow_tracking_uri)
    rows = []
    for m, name in submitted.items():
        if statuses[m] != "Completed":
            print(f"skip {m}: job {statuses[m]}")
            continue
        mtr = mlflow.get_run(name).data.metrics
        rows.append({"model": m, "run_id": name, **{
            k: mtr.get(k) for k in ["recall", "pr_auc", "precision", "roc_auc", "accuracy"]
        }})

    if not rows:
        print("No completed runs to compare. Check job logs in AzureML Studio.")
        return

    board = pd.DataFrame(rows).sort_values(["recall", "pr_auc"], ascending=False)
    print("\n=== leaderboard ===")
    print(board.to_string(index=False))

    champ = board.iloc[0]
    # Register the champion's saved pickle (the job's model_out output) into the AzureML
    # Model Registry via the SDK — not mlflow.register_model, which would hit the same
    # unsupported mlflow 3.x endpoint. CUSTOM_MODEL = a plain artifact folder.
    champ_uri = f"azureml://jobs/{champ['run_id']}/outputs/model_out"
    reg = ml_client.models.create_or_update(Model(
        path=champ_uri,
        name="delaycast-champion",
        type=AssetTypes.CUSTOM_MODEL,
        description=f"{champ['model']} champion · recall={champ['recall']:.4f} pr_auc={champ['pr_auc']:.4f}",
    ))
    print(f"\n>>> CHAMPION {champ['model']} -> registered delaycast-champion v{reg.version}")
    print(f"    recall={champ['recall']:.4f}  pr_auc={champ['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
