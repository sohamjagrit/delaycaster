# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Feature engineering (PySpark)
# MAGIC
# MAGIC **What this does:** turns the raw BTS flights into the modelling table. Builds
# MAGIC base features (carrier, origin, dest, distance, dep_hour, day_of_week, month,
# MAGIC is_weekend) and **leakage-safe rolling 30-day delay-rate** features (per carrier,
# MAGIC origin, dest, carrier+origin). Writes three artifacts to the lake:
# MAGIC
# MAGIC 1. **Gold Delta table** `features/flights_features` — the lakehouse source of truth.
# MAGIC 2. **Parquet snapshot** `features/train_parquet` — what the AzureML training job reads.
# MAGIC 3. **Serving lookup** `features/serving_lookup` — latest delay rate per entity, so the
# MAGIC    Streamlit app can rebuild the full feature vector from raw inputs (no train/serve skew).
# MAGIC
# MAGIC **Leakage rule (CRITICAL):** every feature uses only information known at *scheduled
# MAGIC departure*. The rolling rates are computed over a window that **ends the day before**
# MAGIC each flight (`rangeBetween(-30, -1)`), so a flight never sees its own day's outcomes.
# MAGIC We never touch DepDelay, Carrier/Weather/NAS delay, or actual times.
# MAGIC
# MAGIC **Compute:** classic `delaycaster` cluster (NOT serverless — ADLS key-auth gotcha).
# MAGIC This is the step that justifies a multi-worker cluster if you scale workers up.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the labelled, flown flights (from the EDA decisions)

# COMMAND ----------

from pyspark.sql import functions as F, Window

STORAGE_ACCOUNT = "delaycasterdata"
CONTAINER       = "data"
LAKE_BASE       = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"
FEATURES_BASE   = f"{LAKE_BASE}/features"

# Only the columns we need. Raw layer keeps all 109; we select deliberately here.
raw = (
    spark.read.option("header", True).option("inferSchema", True)
    .csv(f"{LAKE_BASE}/raw/bts/*/*.csv")
    .select(
        "FlightDate", "IATA_CODE_Reporting_Airline", "Origin", "Dest",
        "CRSDepTime", "Distance", "ArrDelay", "Cancelled",
    )
)

# EDA decisions: drop cancelled / unlabelable rows, build the target.
flown = (
    raw
    .filter((F.col("Cancelled") == 0) & F.col("ArrDelay").isNotNull())
    .withColumn("FlightDate", F.to_date("FlightDate"))
    .withColumn("delayed", (F.col("ArrDelay") > 15).cast("int"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Base features (all known at scheduled departure)
# MAGIC
# MAGIC `dep_hour = CRSDepTime // 100` (CRSDepTime is HHMM, e.g. 1430 → 14). `day_of_week`
# MAGIC uses Monday=0 to match `datetime.weekday()` in `src/features.py` (Spark's
# MAGIC `dayofweek` is Sunday=1, so we remap) — this parity is what keeps serving honest.

# COMMAND ----------

base = (
    flown
    .withColumnRenamed("IATA_CODE_Reporting_Airline", "carrier")
    .withColumnRenamed("Origin", "origin")
    .withColumnRenamed("Dest", "dest")
    .withColumn("distance", F.col("Distance").cast("double"))
    .withColumn("dep_hour", (F.col("CRSDepTime").cast("int") / 100).cast("int"))
    # Spark dayofweek: 1=Sun..7=Sat  ->  remap to Python weekday: 0=Mon..6=Sun
    .withColumn("day_of_week", ((F.dayofweek("FlightDate") + 5) % 7))
    .withColumn("month", F.month("FlightDate"))
    .withColumn("is_weekend", (F.col("day_of_week") >= 5).cast("int"))
    .withColumn("carrier_origin", F.concat_ws("|", "carrier", "origin"))
    .select(
        "FlightDate", "carrier", "origin", "dest", "carrier_origin",
        "distance", "dep_hour", "day_of_week", "month", "is_weekend", "delayed",
    )
)

# Global base rate — the serving fallback for unseen entities, and the seed for early
# dates that have no 30-day history yet.
GLOBAL_RATE = base.agg(F.avg("delayed")).first()[0]
print(f"Global delay rate: {GLOBAL_RATE:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Leakage-safe rolling 30-day delay rates
# MAGIC
# MAGIC Pattern (and why it scales): first collapse to **per-entity, per-day** counts
# MAGIC (`delayed_sum`, `flights`) — this shrinks 20M flight rows to a small grid. Then a
# MAGIC window over just that grid sums the **prior 30 days** (`rangeBetween(-30, -1)` on an
# MAGIC integer day index — the `-1` upper bound is what excludes the flight's own day, i.e.
# MAGIC no leakage). Rate = rolling delayed / rolling flights. Finally join the rate back to
# MAGIC each flight by (entity, date). Doing the window on the daily grid instead of 20M raw
# MAGIC rows is the difference between seconds and a shuffle meltdown.

# COMMAND ----------

DAY = F.datediff("FlightDate", F.lit("1970-01-01"))  # integer day index for the range window

def rolling_rate(df, key_col: str, rate_name: str):
    """Per-(key, day) rolling 30-day delay rate, computed strictly from prior days."""
    daily = (
        df.groupBy(key_col, "FlightDate")
          .agg(F.sum("delayed").alias("d_sum"), F.count("*").alias("d_cnt"))
          .withColumn("day_idx", DAY)
    )
    w = (
        Window.partitionBy(key_col)
        .orderBy("day_idx")
        .rangeBetween(-30, -1)          # [D-30, D-1] — excludes the current day → no leakage
    )
    return (
        daily
        .withColumn("roll_sum", F.sum("d_sum").over(w))
        .withColumn("roll_cnt", F.sum("d_cnt").over(w))
        # Early dates have no prior window -> fall back to the global rate.
        .withColumn(
            rate_name,
            F.when(F.col("roll_cnt") > 0, F.col("roll_sum") / F.col("roll_cnt"))
             .otherwise(F.lit(GLOBAL_RATE)),
        )
        .select(key_col, "FlightDate", rate_name)
    )

carrier_rate = rolling_rate(base, "carrier", "carrier_delay_rate")
origin_rate  = rolling_rate(base, "origin",  "origin_delay_rate")
dest_rate    = rolling_rate(base, "dest",    "dest_delay_rate")
co_rate      = rolling_rate(base, "carrier_origin", "carrier_origin_delay_rate")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Assemble the modelling table
# MAGIC
# MAGIC Join each rolling rate back onto the flights by (entity, FlightDate). Keep
# MAGIC `FlightDate` in the table so training can do a **time-based** train/test split
# MAGIC (train on earlier months, test on later) — the only honest split for a forecasting
# MAGIC task.

# COMMAND ----------

features = (
    base
    .join(carrier_rate, ["carrier", "FlightDate"], "left")
    .join(origin_rate,  ["origin",  "FlightDate"], "left")
    .join(dest_rate,    ["dest",    "FlightDate"], "left")
    .join(co_rate,      ["carrier_origin", "FlightDate"], "left")
    .fillna(GLOBAL_RATE, subset=[
        "carrier_delay_rate", "origin_delay_rate", "dest_delay_rate", "carrier_origin_delay_rate",
    ])
    .drop("carrier_origin")
    .select(
        "FlightDate",
        # canonical order must match src/features.FEATURE_COLUMNS
        "carrier", "origin", "dest",
        "distance", "dep_hour", "day_of_week", "month", "is_weekend",
        "carrier_delay_rate", "origin_delay_rate", "dest_delay_rate", "carrier_origin_delay_rate",
        "delayed",
    )
)

print(f"Feature table rows: {features.count():,}")
display(features.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write artifact 1 — gold Delta table

# COMMAND ----------

(
    features.write.format("delta").mode("overwrite")
    .save(f"{FEATURES_BASE}/flights_features")
)
print("Wrote gold Delta:", f"{FEATURES_BASE}/flights_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write artifact 2 — Parquet snapshot for AzureML
# MAGIC
# MAGIC AzureML reads Parquet from the ADLS datastore cleanly (no Delta reader needed in the
# MAGIC training image). Repartition to a handful of files so the job reads it efficiently.

# COMMAND ----------

(
    features.repartition(8).write.mode("overwrite")
    .parquet(f"{FEATURES_BASE}/train_parquet")
)
print("Wrote Parquet snapshot:", f"{FEATURES_BASE}/train_parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write artifact 3 — serving lookup table
# MAGIC
# MAGIC The latest rolling rate per entity (the value as of the most recent flight date),
# MAGIC plus a `__global__` fallback row. Long format: `entity_kind, entity_key, delay_rate`.
# MAGIC The Streamlit app loads this and rebuilds the feature vector via `src/features.py`.

# COMMAND ----------

def latest_per_entity(rate_df, key_col, rate_name, kind):
    w = Window.partitionBy(key_col).orderBy(F.desc("FlightDate"))
    return (
        rate_df
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(
            F.lit(kind).alias("entity_kind"),
            F.col(key_col).alias("entity_key"),
            F.col(rate_name).alias("delay_rate"),
        )
    )

lookup = (
    latest_per_entity(carrier_rate, "carrier", "carrier_delay_rate", "carrier")
    .unionByName(latest_per_entity(origin_rate, "origin", "origin_delay_rate", "origin"))
    .unionByName(latest_per_entity(dest_rate, "dest", "dest_delay_rate", "dest"))
    .unionByName(latest_per_entity(co_rate, "carrier_origin", "carrier_origin_delay_rate", "carrier_origin"))
    .unionByName(
        spark.createDataFrame([("__global__", "__global__", float(GLOBAL_RATE))],
                              ["entity_kind", "entity_key", "delay_rate"])
    )
)

# Small table -> coalesce to one file so the app downloads a single parquet.
lookup.coalesce(1).write.mode("overwrite").parquet(f"{FEATURES_BASE}/serving_lookup")
print("Wrote serving lookup:", f"{FEATURES_BASE}/serving_lookup")
display(lookup.filter(F.col("entity_kind") == "carrier").orderBy(F.desc("delay_rate")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC - `train_parquet` is what AzureML reads — see `azureml/` + `04_train_evaluate`.
# MAGIC - Champion training uses a **time split** on `FlightDate` (train ≤ 2024-06, test after),
# MAGIC   never a random split, because predicting delays is a forecasting task.
# MAGIC - The serving lookup + `src/features.py` are the bridge to the Streamlit app (Stage 3).
