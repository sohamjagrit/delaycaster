# AzureML Stage 2 setup — Azure CLI runbook (macOS)

End-to-end, copy-paste runbook to stand up the AzureML side of DelayCast: install the CLI,
authenticate, create the workspace + datastore + compute + environment, and wire ADLS access.
Run every command from the **repo root** (`/Users/soham/Desktop/delaycaster`) so the relative
`azureml/*.yml` paths resolve.

> Prereq: `03_features` has already written `features/train_parquet` to ADLS. The training
> jobs read that folder — without it, Step 7 has nothing to train on.

---

## Step 0 — Install the Azure CLI

You're on macOS, so use Homebrew (the supported, auto-updating path).

```bash
# If you don't have Homebrew yet:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew update
brew install azure-cli
```

Verify it installed (expect azure-cli 2.60+):

```bash
az version
```

> Alternative without Homebrew:
> `curl -L https://aka.ms/InstallAzureCli | bash` then restart your shell.

---

## Step 1 — Log in

```bash
az login
```

This opens a browser. Sign in with the account that owns **Azure subscription 1**. On success
the terminal prints your subscriptions as JSON.

If the browser doesn't open (or you're on a headless box):

```bash
az login --use-device-code
```

---

## Step 2 — Select the right subscription

List what you can see, then pin the one this project uses:

```bash
az account list --output table
az account set --subscription "Azure subscription 1"
```

Capture the subscription id into a shell variable — later steps and `04_train_evaluate` need it:

```bash
SUB_ID=$(az account show --query id -o tsv)
echo "Subscription: $SUB_ID"
```

---

## Step 3 — Add the `ml` extension and register providers

The `az ml` commands live in an extension that isn't installed by default:

```bash
az extension add -n ml
az extension list --output table     # confirm 'ml' is listed
az ml -h                             # confirm the command group loads
```

Register the resource providers the workspace depends on (no-op if already registered; each
takes a minute):

```bash
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.Insights
```

---

## Step 4 — Create the AzureML workspace

Reuses the existing resource group and region (no new infra sprawl, stays in West US 2):

```bash
az ml workspace create \
  --name delaycaster-aml \
  --resource-group delaycaster-rg \
  --location westus2
```

This takes **3–5 minutes** — it provisions the workspace plus its supporting storage account,
Key Vault, Container Registry, and Application Insights. Confirm it's there:

```bash
az ml workspace show -n delaycaster-aml -g delaycaster-rg --query name -o tsv
```

Set default workspace + RG so you can drop `-g/-w` on later commands (optional, convenient):

```bash
az configure --defaults group=delaycaster-rg workspace=delaycaster-aml
```

---

## Step 5 — Connect AzureML to your ADLS Gen2 lake (datastore)

This registers `delaycaster_lake`, pointing at the `data` container on `delaycasterdata`, so
jobs can read `features/train_parquet` directly — no copying 9 GB around.

```bash
az ml datastore create --file azureml/datastore_adls.yml \
  -g delaycaster-rg -w delaycaster-aml
```

The YAML uses **identity-based** auth (no secret in the repo), so you must grant the read role —
see Step 6. Verify the datastore registered:

```bash
az ml datastore show -n delaycaster_lake --query name -o tsv
```

---

## Step 6 — Grant ADLS read access (RBAC for identity-based auth)

Two identities need **Storage Blob Data Reader** on the storage account: **you** (for any local
interactive read) and the **workspace managed identity** (used by jobs). Training only reads this
datastore, so Reader is enough.

```bash
# Storage account resource id (the scope of the grant)
STORAGE_ID=$(az storage account show -n delaycasterdata -g delaycaster-rg --query id -o tsv)

# Your own object id
ME=$(az ad signed-in-user show --query id -o tsv)

# The workspace's managed identity object id
WS_MI=$(az ml workspace show -n delaycaster-aml -g delaycaster-rg \
        --query identity.principal_id -o tsv)

az role assignment create --assignee "$ME"    --role "Storage Blob Data Reader" --scope "$STORAGE_ID"
az role assignment create --assignee "$WS_MI" --role "Storage Blob Data Reader" --scope "$STORAGE_ID"
```

Role assignments can take a minute or two to propagate.

> **Quick fallback if RBAC is fiddly:** skip Step 6 and switch the datastore to account-key auth.
> Uncomment the `credentials:` block in `azureml/datastore_adls.yml`, paste the storage key, and
> re-run Step 5. Don't commit the real key. Get the key with:
> `az storage account keys list -n delaycasterdata -g delaycaster-rg --query "[0].value" -o tsv`

---

## Step 7 — Create the training compute

Memory-optimized single node that **scales to zero** when idle (so it isn't billing between runs):

```bash
az ml compute create \
  --name cpu-train \
  --type AmlCompute \
  --size Standard_E8s_v3 \
  --min-instances 0 \
  --max-instances 1 \
  --idle-time-before-scale-down 900 \
  -g delaycaster-rg -w delaycaster-aml
```

`Standard_E8s_v3` = 8 vCPU / 64 GB RAM — enough headroom for LightGBM/XGBoost on the full 20M
rows. Confirm:

```bash
az ml compute show -n cpu-train --query "{name:name,size:size,state:provisioning_state}" -o table
```

> If trial quota blocks `E8s_v3` (same quota story as the Databricks `DS3_v2` block), fall back to
> `Standard_DS3_v2` (4 vCPU / 14 GB) or `Standard_D4s_v3`. List what your quota allows:
> `az ml compute list-sizes --query "[?contains(name,'Standard_E') || contains(name,'Standard_D')].name" -o table`

---

## Step 8 — Create the training environment

Builds the conda image (sklearn / xgboost / lightgbm / mlflow) the jobs run in:

```bash
az ml environment create --file azureml/env_train.yml \
  -g delaycaster-rg -w delaycaster-aml

az ml environment show -n delaycast-train --query "{name:name,version:version}" -o table
```

The first build takes a few minutes; later jobs reuse the cached image.

---

## Step 9 — Verify everything is in place

```bash
echo "Subscription : $SUB_ID"
az ml workspace   show -n delaycaster-aml --query name -o tsv
az ml datastore   show -n delaycaster_lake --query name -o tsv
az ml compute     show -n cpu-train       --query provisioning_state -o tsv
az ml environment show -n delaycast-train --query name -o tsv
```

All four resolving = setup done.

---

## Step 10 — Run training

Open `notebooks/04_train_evaluate.py`, set `SUBSCRIPTION = "<paste $SUB_ID>"`, and run it. It
submits 4 AzureML jobs, waits, compares on recall + PR-AUC, and registers `delaycast-champion`.

Watch jobs from the CLI if you like:

```bash
az ml job list --query "[?experiment_name=='delaycast'].{name:display_name,status:status}" -o table
```

---

## Cost control / teardown

- `cpu-train` returns to **0 nodes** after 15 min idle — no compute charge between runs.
- When Stage 2 is done you can delete just the compute: `az ml compute delete -n cpu-train --yes`.
- Keep `delaycaster-rg` until the whole project wraps (it holds the 20.66M rows + the registered
  model). Final teardown of everything: `az group delete -n delaycaster-rg --yes --no-wait`.
