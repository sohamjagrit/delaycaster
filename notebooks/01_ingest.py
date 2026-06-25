# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest BTS On-Time Performance data into ADLS Gen2
# MAGIC
# MAGIC **What this does:** downloads the BTS pre-zipped monthly On-Time Performance file
# MAGIC straight from the source, lands the raw CSV in the data lake under
# MAGIC `raw/bts/YYYY/`, then reads it back with Spark to inspect the schema and row count.
# MAGIC
# MAGIC **Why:** this is the reproducible ingestion step — re-running this notebook
# MAGIC recreates the raw layer from source, no manual downloads. We start with a single
# MAGIC month (2024-01) to validate the whole path cheaply before scaling to all 36 months.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Connect Spark to the data lake (ADLS Gen2)
# MAGIC
# MAGIC Spark needs permission to read/write our storage account. We read the storage
# MAGIC account key from a **Databricks secret scope** (never hardcode credentials).
# MAGIC The `abfss://` scheme is the ADLS Gen2 driver — think of it as the `s3://` of Azure.

# COMMAND ----------

STORAGE_ACCOUNT = "delaycasterdata"          # your ADLS Gen2 account
CONTAINER       = "data"                       # the container you created

# Auth note: the storage account key is supplied via the CLUSTER Spark config, not here,
# because a key set at cell-runtime isn't honored by dbutils.fs on this cluster type.
# In the cluster's Advanced > Spark config we set (key pulled from the secret scope, no plaintext):
#   fs.azure.account.key.delaycasterdata.dfs.core.windows.net {{secrets/delaycaster/storage-key}}
# So nothing is hardcoded and the secret-scope discipline holds — the credential just
# loads at cluster startup instead of at runtime.

# Base path into the lake. Everything we read/write hangs off here.
LAKE_BASE = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
print("Lake base path:", LAKE_BASE)

# Quick connectivity check — lists the container root. Should not error.
display(dbutils.fs.ls(LAKE_BASE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download + unzip one month from BTS, land the CSV in the lake
# MAGIC
# MAGIC BTS publishes pre-zipped monthly files at a predictable URL. We download the zip
# MAGIC to the driver's local disk, unzip it, then copy the CSV into the lake at
# MAGIC `raw/bts/YYYY/`. `file:` = the cluster's local disk; `abfss:` = the lake.

# COMMAND ----------

import os, zipfile, urllib.request

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

def ingest_month(year: int, month: int):
    """Download one BTS month, unzip, and land the CSV in the lake under raw/bts/YYYY/."""
    url = BTS_URL_TEMPLATE.format(year=year, month=month)
    local_zip = f"/tmp/bts_{year}_{month}.zip"
    local_dir = f"/tmp/bts_{year}_{month}"

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, local_zip)
    print(f"  saved zip: {os.path.getsize(local_zip)/1e6:.1f} MB")

    # Unzip — the archive contains one big CSV (plus a readme we ignore).
    with zipfile.ZipFile(local_zip) as z:
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        z.extract(csv_name, local_dir)
    local_csv = os.path.join(local_dir, csv_name)
    print(f"  extracted CSV: {csv_name}")

    # Copy local CSV -> lake. We rename to a clean, predictable filename.
    dest = f"{LAKE_BASE}/raw/bts/{year}/bts_{year}_{month:02d}.csv"
    dbutils.fs.cp(f"file:{local_csv}", dest)
    print(f"  landed in lake: {dest}")
    return dest

# Validate the pipeline on a SINGLE month first.
dest_path = ingest_month(2024, 1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Read the raw CSV with Spark and inspect it
# MAGIC
# MAGIC `spark.read` builds a distributed DataFrame. Note: Spark is *lazy* — it doesn't
# MAGIC actually read the file until an action like `.count()` or `.display()` forces it.

# COMMAND ----------

raw = (
    spark.read
    .option("header", True)        # first row is column names
    .option("inferSchema", True)   # let Spark guess column types (fine for inspection)
    .csv(f"{LAKE_BASE}/raw/bts/2024/bts_2024_01.csv")
)

print("Columns:", len(raw.columns))
raw.printSchema()

# COMMAND ----------

# Row count — this is an ACTION, so it triggers Spark to actually scan the file.
print("Row count for 2024-01:", raw.count())

# COMMAND ----------

# Peek at the columns we care about. NOTE: the BTS *pre-zipped* file uses these
# names (CamelCase), which differ from the custom-download form's UPPER_SNAKE names.
# Left = what CLAUDE.md called it; right = the actual column in this file.
#   FL_DATE        -> FlightDate
#   OP_CARRIER     -> IATA_CODE_Reporting_Airline   (2-letter code: AA, UA, DL)
#   ORIGIN         -> Origin
#   DEST           -> Dest
#   CRS_DEP_TIME   -> CRSDepTime                     (scheduled dep, known before flight)
#   DEP_DELAY      -> DepDelay      <-- LEAKAGE: known only after departure
#   ARR_DELAY      -> ArrDelay                       (used to build the target only)
#   CANCELLED      -> Cancelled
#   DISTANCE       -> Distance
#   CARRIER_DELAY  -> CarrierDelay  <-- LEAKAGE: delay cause, known after the fact
#   WEATHER_DELAY  -> WeatherDelay  <-- LEAKAGE
#   NAS_DELAY      -> NASDelay      <-- LEAKAGE
cols_of_interest = [
    "FlightDate", "IATA_CODE_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepDelay", "ArrDelay",
    "Cancelled", "Distance",
    "CarrierDelay", "WeatherDelay", "NASDelay",
]
# Keep only columns that actually exist so this cell never breaks across releases.
present = [c for c in cols_of_interest if c in raw.columns]
print("Present columns of interest:", present)
display(raw.select(*present).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Loop the full 3 years (36 months) into the lake
# MAGIC
# MAGIC Only run this AFTER the single-month validation above looks right. Three
# MAGIC production habits baked in so re-running is safe and a bad month is visible:
# MAGIC
# MAGIC - **Idempotent** — skip any month whose CSV already landed, so a re-run after a
# MAGIC   failure doesn't re-download ~9 GB. Pass `overwrite=True` to force a refresh.
# MAGIC - **Retries** — BTS occasionally stalls; retry the download a few times with backoff
# MAGIC   before giving up on a month.
# MAGIC - **Fail-soft + audit** — one bad month doesn't kill the run; we collect a per-month
# MAGIC   row count so a silently truncated download is obvious at the end.

# COMMAND ----------

import time

YEARS  = [2022, 2023, 2024]
MONTHS = range(1, 13)

def _lake_csv_path(year: int, month: int) -> str:
    return f"{LAKE_BASE}/raw/bts/{year}/bts_{year}_{month:02d}.csv"

def _already_landed(year: int, month: int) -> bool:
    """True if this month's CSV already exists in the lake (idempotency check)."""
    try:
        dbutils.fs.ls(_lake_csv_path(year, month))
        return True
    except Exception:
        return False

def ingest_month_safe(year: int, month: int, overwrite: bool = False, retries: int = 3):
    """Idempotent, retrying wrapper around ingest_month. Returns the lake path or None."""
    if not overwrite and _already_landed(year, month):
        print(f"[skip] {year}-{month:02d} already in lake")
        return _lake_csv_path(year, month)
    for attempt in range(1, retries + 1):
        try:
            return ingest_month(year, month)
        except Exception as e:
            wait = 5 * attempt
            print(f"[retry] {year}-{month:02d} attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(wait)
    print(f"[FAIL] {year}-{month:02d} gave up after {retries} attempts")
    return None

# Download every month (skips ones already present).
results = []
for year in YEARS:
    for month in MONTHS:
        results.append((year, month, ingest_month_safe(year, month)))

landed = [(y, m, p) for (y, m, p) in results if p is not None]
failed = [(y, m) for (y, m, p) in results if p is None]
print(f"\nLanded {len(landed)}/{len(results)} months. Failed: {failed or 'none'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Audit — confirm every month landed with a sane row count
# MAGIC
# MAGIC Read all 36 months back at once (Spark globs the whole `raw/bts` tree) and count
# MAGIC rows per source file. Each month should be ~500k–600k; a tiny or missing count
# MAGIC means a truncated/failed download to re-run above with `overwrite=True`.

# COMMAND ----------

from pyspark.sql import functions as F

audit = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)            # audit only needs row counts, skip the type-inference pass
    .csv(f"{LAKE_BASE}/raw/bts/*/*.csv")
    .withColumn("source_file", F.input_file_name())
    .groupBy("source_file")
    .count()
    .orderBy("source_file")
)
print("Total rows across all landed months:", audit.agg(F.sum("count")).collect()[0][0])
display(audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC - Confirm the audit shows 36 files, each ~500k–600k rows (~20M total).
# MAGIC - Re-run any failed/short month above with `ingest_month_safe(y, m, overwrite=True)`.
# MAGIC - EDA lives in `02_eda`; feature engineering + the leakage rule in `03_features`.
