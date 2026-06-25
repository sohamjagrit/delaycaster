# Data Engineering — Fundamentals, Best Practices & Interview Prep

A reference companion to the DelayCast build. Part 1 is the mental model and best
practices (with how each one shows up in *this* project). Part 2 is the top 20
data-engineering interview questions with crisp answers you can defend.

---

## Part 1 — Fundamentals & Best Practices

### 1. The core job of data engineering
Move data from where it's *produced* to where it's *useful*, reliably and repeatably,
at the right freshness and cost. Everything below serves that sentence.

### 2. The medallion (bronze → silver → gold) architecture
- **Bronze (raw):** a faithful, unmodified copy of the source. Never transform here.
  *DelayCast:* the 109-column BTS CSVs in `raw/bts/YYYY/` — kept exactly as downloaded.
- **Silver (cleaned):** typed, de-duplicated, validated, joined. One row = one real event.
  *DelayCast:* dropped cancelled flights, parsed dates/times, enforced schema.
- **Gold (curated):** business-ready aggregates / feature tables consumed by ML or BI.
  *DelayCast:* the Delta feature table + serving lookup in `data/features/`.

**Why layer at all?** You can always rebuild silver/gold from bronze. If a transform has a
bug, you reprocess — you never lose the source. Layers also isolate failures.

### 3. ELT vs ETL
- **ETL** (transform *before* load): older pattern, transform in flight, load the result.
- **ELT** (load raw first, transform *in* the warehouse/lake): modern default. Land bronze,
  then transform with the lake's compute. Cheaper storage + reprocessability make this win.
  *DelayCast is ELT:* land raw CSVs first (`01_ingest`), transform later (`03_features`).

### 4. Idempotency (the most underrated habit)
Re-running a job must produce the same result, not duplicates or corruption. Achieve it with:
- **Skip-if-exists / overwrite-by-partition** instead of blind appends.
- **Deterministic output paths** (one partition per logical unit).
  *DelayCast:* `ingest_month_safe` skips months already landed; a re-run after a crash is safe.

### 5. Schema management
- **Don't `inferSchema` in production reads** — it costs an extra full scan and can type a
  column differently across files, breaking a union. Pin an explicit schema for repeatable jobs.
  Inference is fine for *one-off inspection* only.
- **Plan for schema evolution:** sources add/rename columns. Delta/Parquet handle additive
  evolution; renames and type changes need a migration.

### 6. Partitioning & file layout
- Partition by a column you actually filter on (commonly date): `…/year=2024/month=01/`.
- **Avoid the small-files problem:** thousands of tiny files kill read performance. Compact
  (Delta `OPTIMIZE`, or coalesce on write) toward ~128 MB–1 GB files.
- **Avoid data skew:** one giant partition stalls a whole stage; salt or repartition hot keys.

### 7. Columnar formats & the lakehouse
- **Parquet/ORC** (columnar) beat CSV/JSON for analytics: compression + column pruning +
  predicate pushdown. Land in CSV if that's the source, but *store curated data columnar*.
- **Delta Lake / Iceberg / Hudi** add ACID transactions, time travel, schema enforcement, and
  upserts (MERGE) on top of Parquet — this is the "lakehouse": warehouse reliability on lake
  storage. *DelayCast* writes the gold layer as **Delta**.

### 8. Batch vs streaming
- **Batch:** process a bounded chunk on a schedule. Simpler, cheaper, most pipelines.
- **Streaming:** process unbounded events continuously (Kafka, Spark Structured Streaming,
  Flink). Use only when freshness genuinely demands it — it's more operationally expensive.
- **Micro-batch** is the practical middle ground (Structured Streaming's default).

### 9. Distributed compute fundamentals (Spark mental model)
- **Lazy evaluation:** transformations (`select`, `filter`, `join`) build a plan; nothing runs
  until an **action** (`count`, `collect`, `write`, `display`) triggers it.
- **Job → Stage → Task:** an action = a **job**; a job splits into **stages** at shuffle
  boundaries; a stage runs as parallel **tasks** (one per partition).
- **Shuffle is the expensive part:** `groupBy`, `join`, `distinct` move data across the network.
  Minimize shuffles; broadcast the small side of a join (`broadcast(df)`) to avoid one.
- **Narrow vs wide transformations:** narrow (`map`, `filter`) stay within a partition; wide
  (`groupBy`, `join`) require a shuffle.
- **Right-size compute:** pure I/O (file movement) is single-threaded — don't pay for workers
  Spark can't use. Heavy transforms over many rows *do* benefit from more cores.
  *DelayCast:* single-node for ingest (I/O), bigger compute for feature engineering.

### 10. Orchestration
- Pipelines are **DAGs** of tasks with dependencies, retries, schedules, and alerting.
- Tools: **Airflow**, **Azure Data Factory**, **Dagster**, **Databricks Workflows**, **dbt**
  (for SQL transforms). A *Spark job* (execution unit) ≠ a *Workflow Job* (scheduled pipeline).
- Production file landing belongs in an orchestrator (ADF/Function), **not** a Spark cluster.

### 11. Data quality & contracts
- **Validate at boundaries:** row counts, null rates, ranges, uniqueness, referential integrity.
- **Fail loud, fail early** — a silently truncated load is worse than a hard error.
- Tools: Great Expectations, dbt tests, Delta constraints. *DelayCast:* per-file row-count audit
  catches a short/failed month.

### 12. Target leakage (critical for ML-facing pipelines)
Never let a feature carry information unavailable at prediction time. It inflates offline
metrics and collapses in production. *DelayCast:* excludes `DepDelay`, `CarrierDelay`,
`WeatherDelay`, actual times — only scheduled-departure-time info is allowed.

### 13. Training/serving skew
The features computed in the pipeline must match those computed at serving time, exactly.
Fix with a **feature store** or a shared transform module + a **serving lookup table**.
*DelayCast:* `03_features` writes a lookup of latest delay-rates so the app rebuilds the same
vector from raw inputs.

### 14. Secrets & security
Never hardcode credentials. Use a **secret scope / vault**; reference, don't embed. Scope tokens
narrowly, expire and revoke them. *DelayCast:* storage key in a Databricks secret scope,
referenced via cluster Spark config — never in the notebook.

### 15. Cost & observability
- Storage is cheap; **compute is the bill** — terminate idle clusters, right-size instances.
- Log run metadata: row counts, durations, data versions. You can't fix what you can't see.
- Tag resources; delete what you no longer need.

### 16. Incremental processing & CDC
Reprocessing everything every run doesn't scale. Process only what changed via **watermarks**
(high-water-mark timestamps) or **Change Data Capture** (CDC) from source systems. Upsert with
`MERGE` into the lakehouse.

### 17. Slowly Changing Dimensions (SCD)
Dimension attributes change over time (a customer moves city). **SCD Type 1** overwrites;
**Type 2** keeps history with effective-date / current-flag rows. Type 2 is the classic
warehouse-modeling interview topic.

### 18. Modeling: normalized vs dimensional
- **Normalized (3NF):** minimal redundancy — good for OLTP/source systems.
- **Dimensional (star schema):** fact tables + denormalized dimensions — good for analytics
  (fewer joins, faster aggregates). Know when each applies.

---

## Part 2 — Top 20 Data Engineering Interview Questions

> Answer with a definition → a trade-off → a concrete example. The example is what lands it.

**1. Explain ETL vs ELT and when you'd choose each.**
ETL transforms before loading; ELT lands raw then transforms in the warehouse/lake. ELT is the
modern default (cheap storage, reprocessability, lake compute); ETL still fits when you must
mask/shrink sensitive data before it ever lands.

**2. What is the medallion (bronze/silver/gold) architecture and why use it?**
Layered refinement: raw faithful copy → cleaned/validated → curated business tables. It gives
reprocessability (rebuild downstream from bronze), failure isolation, and clear ownership.

**3. What makes a pipeline idempotent, and why does it matter?**
Re-running yields the same result — no dupes/corruption. Achieve via skip-if-exists, partition
overwrite, deterministic paths, or MERGE. Matters because jobs *will* retry after failures.

**4. Walk me through how Spark executes a query (lazy eval, jobs/stages/tasks).**
Transformations build a lazy DAG; an action triggers a job; the job splits into stages at
shuffle boundaries; each stage runs as parallel tasks (one per partition).

**5. What is a shuffle and how do you minimize it?**
Network redistribution of data for wide ops (`join`, `groupBy`, `distinct`). Minimize by
broadcasting small tables, pre-partitioning on the join key, filtering early, and avoiding
unnecessary `distinct`/re-grouping.

**6. How do you handle data skew?**
Detect a few partitions dominating runtime. Fix with salting (add a random key suffix),
adaptive query execution, broadcast joins, or repartitioning the hot key.

**7. Partitioning vs bucketing — what's the difference?**
Partitioning splits data into directories by a column (prunes on filter). Bucketing hashes rows
into a fixed number of files per partition (co-locates join keys, avoids shuffle on join).

**8. What's the small-files problem and how do you fix it?**
Many tiny files = huge metadata/scheduling overhead and slow reads. Fix by compacting
(Delta `OPTIMIZE`, `coalesce`/`repartition` on write) toward ~128 MB–1 GB files.

**9. Why columnar formats (Parquet) over CSV/JSON for analytics?**
Column pruning, better compression, predicate pushdown, and embedded schema — far less I/O for
analytical scans that touch few columns.

**10. What does Delta Lake (a lakehouse format) add over plain Parquet?**
ACID transactions, schema enforcement/evolution, time travel (versioned data), and upserts/
deletes via `MERGE` — warehouse guarantees on cheap lake storage.

**11. Batch vs streaming — how do you decide?**
Driven by freshness SLA and cost. Default to batch/micro-batch; reach for true streaming only
when sub-minute latency is a hard requirement. Streaming adds state, ordering, and ops burden.

**12. How do you do incremental processing / CDC?**
Track a high-water mark (watermark) or consume change events from the source, then upsert with
`MERGE`. Avoids full reprocessing and scales with data growth.

**13. Explain Slowly Changing Dimensions (Type 1 vs Type 2).**
Type 1 overwrites (no history); Type 2 inserts a new versioned row with effective dates / a
current flag to preserve history. Type 2 is standard for auditable dimensions.

**14. Star schema vs normalized (3NF) — when each?**
3NF minimizes redundancy for transactional systems; star schema denormalizes into facts +
dimensions for fast analytics with fewer joins. Choose by workload (OLTP vs OLAP).

**15. How do you ensure data quality in a pipeline?**
Validate at boundaries — row counts, null/range/uniqueness checks, referential integrity — and
fail loudly. Use Great Expectations / dbt tests / Delta constraints; alert on violations.

**16. What is target leakage and how do you prevent it?**
A feature encoding info unavailable at prediction time, inflating metrics and failing in prod.
Prevent by only using data known *before* the event (exclude post-event columns), and audit
feature provenance.

**17. How do you avoid training/serving skew?**
Compute features with shared code or a feature store so training and serving use identical
logic; ship a serving lookup/feature table refreshed on schedule.

**18. How do you handle late-arriving / out-of-order data in streaming?**
Event-time processing with watermarks: define how long to wait for stragglers, window on event
time, and drop/handle data beyond the watermark.

**19. How do you orchestrate and monitor pipelines?**
Model as a DAG (Airflow/ADF/Dagster/Databricks Workflows) with dependencies, retries, schedules,
and alerting; emit run metadata (row counts, durations, versions) and SLAs for observability.

**20. How do you control cost in a cloud data platform?**
Storage is cheap, compute is the bill: right-size and auto-terminate clusters, separate I/O from
heavy transforms, prefer columnar + partition pruning to cut scanned bytes, cache deliberately,
and delete idle resources. Tag everything to attribute spend.

---

### How DelayCast demonstrates these (your interview cheat-sheet)
- **Medallion + ELT:** raw BTS CSVs (bronze) → cleaned (silver) → Delta feature table (gold).
- **Idempotency:** skip-if-landed ingest loop with retries + row-count audit.
- **Right-sized compute:** single-node for I/O ingest, scaled compute for feature engineering.
- **Leakage discipline:** explicit exclusion of post-departure columns.
- **Training/serving skew:** serving lookup table rebuilt from raw inputs in the app.
- **Secrets:** storage key in a secret scope, never in code.
- **Cost control:** terminate clusters, keep only cheap storage at rest.
