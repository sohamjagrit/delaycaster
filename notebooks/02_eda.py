# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Exploratory Data Analysis
# MAGIC
# MAGIC **What this does:** reads all 36 raw BTS months, builds the `delayed` target
# MAGIC (`ArrDelay > 15`), and quantifies the things we quote in the README + interview:
# MAGIC class balance, cancelled-flight handling, the delay distribution, nulls in the
# MAGIC feature columns, and the worst carriers / origin airports.
# MAGIC
# MAGIC **Why:** the ~20% positive-class rate is the single most important number in the
# MAGIC project — it's why we optimize **recall + PR-AUC** instead of accuracy (a
# MAGIC "never delayed" model already scores ~80%). EDA also decides that we **drop
# MAGIC cancelled rows** (their `ArrDelay` is null — "was it delayed?" is undefined).
# MAGIC
# MAGIC **Compute:** run on the classic `delaycaster` cluster (NOT serverless — see the
# MAGIC ADLS account-key auth gotcha in CLAUDE.md). The storage key is supplied via the
# MAGIC cluster Spark config, so nothing is hardcoded here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Connect to the lake and read all 36 months

# COMMAND ----------

from pyspark.sql import functions as F

STORAGE_ACCOUNT = "delaycasterdata"
CONTAINER       = "data"
LAKE_BASE       = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Read the whole raw tree at once (Spark globs all year/month CSVs). We keep only the
# columns we reason about in EDA + feature engineering — the raw layer still holds all 109.
RAW_COLS = [
    "FlightDate", "IATA_CODE_Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "Distance",
    "ArrDelay",      # target source only
    "Cancelled",     # drop == 1
    "DepDelay", "CarrierDelay", "WeatherDelay", "NASDelay",  # LEAKAGE — read only to show we exclude them
]

raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(f"{LAKE_BASE}/raw/bts/*/*.csv")
    .select(*RAW_COLS)
)

total_rows = raw.count()
print(f"Total raw rows across 36 months: {total_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cancelled flights — why we drop them
# MAGIC
# MAGIC A cancelled flight never arrived, so `ArrDelay` is null and "was it delayed >15?"
# MAGIC is undefined. We can't label them, so they leave the modelling set. We report the
# MAGIC count so the drop is explicit and auditable.

# COMMAND ----------

cancelled_cnt = raw.filter(F.col("Cancelled") == 1).count()
null_arrdelay = raw.filter(F.col("ArrDelay").isNull()).count()
print(f"Cancelled rows:            {cancelled_cnt:,} ({cancelled_cnt/total_rows:.2%})")
print(f"Rows with null ArrDelay:   {null_arrdelay:,} ({null_arrdelay/total_rows:.2%})")

# Modelling set: actually-flown flights with a known arrival delay.
flown = raw.filter((F.col("Cancelled") == 0) & F.col("ArrDelay").isNotNull())
flown_rows = flown.count()
print(f"Flown rows (modelling set): {flown_rows:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Class balance — the headline number
# MAGIC
# MAGIC `delayed = 1 if ArrDelay > 15 else 0` (US DOT on-time threshold). We expect ~20%
# MAGIC positive. This is what justifies recall over accuracy.

# COMMAND ----------

labeled = flown.withColumn("delayed", (F.col("ArrDelay") > 15).cast("int"))

balance = (
    labeled.groupBy("delayed").count()
    .withColumn("share", F.round(F.col("count") / flown_rows, 4))
    .orderBy("delayed")
)
display(balance)

delayed_rate = labeled.filter(F.col("delayed") == 1).count() / flown_rows
print(f">>> POSITIVE-CLASS RATE (delayed >15 min): {delayed_rate:.2%}")
print(f">>> A trivial 'never delayed' model would score accuracy = {1 - delayed_rate:.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Delay distribution
# MAGIC
# MAGIC Quantiles of `ArrDelay` (minutes) on flown flights — shows the long right tail and
# MAGIC where the 15-minute line sits. Negatives = early arrivals.

# COMMAND ----------

quantiles = flown.approxQuantile("ArrDelay", [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], 0.001)
for q, v in zip([5, 25, 50, 75, 90, 95, 99], quantiles):
    print(f"  p{q:<3} ArrDelay = {v:>7.1f} min")

# Histogram-friendly view for the README screenshot (clip the tail at 180 min).
display(
    flown.withColumn("ArrDelay_clip", F.least(F.greatest(F.col("ArrDelay"), F.lit(-60)), F.lit(180)))
         .select("ArrDelay_clip")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Nulls in the feature columns
# MAGIC
# MAGIC Confirms the columns we'll actually feed the model are clean. (The leakage columns
# MAGIC `CarrierDelay`/`WeatherDelay`/`NASDelay` are mostly null by design — they're only
# MAGIC populated when a delay had that cause — which is exactly why they leak.)

# COMMAND ----------

feature_cols = ["IATA_CODE_Reporting_Airline", "Origin", "Dest", "CRSDepTime", "Distance", "FlightDate"]
null_counts = labeled.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c) for c in feature_cols
])
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Worst carriers and origin airports (delay rate)
# MAGIC
# MAGIC Sanity check that delay rate varies meaningfully by carrier and airport — if it
# MAGIC didn't, the rolling delay-rate features in `03_features` would be useless. We
# MAGIC require a minimum volume so a tiny carrier with 3 flights doesn't top the chart.

# COMMAND ----------

by_carrier = (
    labeled.groupBy("IATA_CODE_Reporting_Airline")
    .agg(F.count("*").alias("flights"), F.avg("delayed").alias("delay_rate"))
    .filter(F.col("flights") > 10000)
    .orderBy(F.desc("delay_rate"))
)
display(by_carrier)

by_origin = (
    labeled.groupBy("Origin")
    .agg(F.count("*").alias("flights"), F.avg("delayed").alias("delay_rate"))
    .filter(F.col("flights") > 10000)
    .orderBy(F.desc("delay_rate"))
)
display(by_origin)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Findings → decisions (carry into 03_features)
# MAGIC - **Drop cancelled / null-ArrDelay rows** — unlabelable. Done above (`flown`).
# MAGIC - **Class balance ≈ 20% delayed** → optimize **recall + PR-AUC**, use class weights.
# MAGIC - **Delay rate varies by carrier/origin/dest** → rolling delay-rate features are justified.
# MAGIC - **Feature columns are clean** (no meaningful nulls) → no imputation needed for base features.
# MAGIC - Next: `03_features` builds base + rolling features on this `flown`/`labeled` set.
