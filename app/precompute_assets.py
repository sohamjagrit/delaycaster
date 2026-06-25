"""Precompute the small, committed assets the Streamlit app reads at runtime.

The app must run on Streamlit Community Cloud with **no Azure access**, so we render
the full 20M-row feature table down to a handful of tiny aggregate CSVs + JSON lookups
here, once, and commit those. Run locally after downloading `train_parquet` from ADLS:

    .venv/bin/python app/precompute_assets.py

Inputs  (gitignored, not committed):
    data_tmp/train_parquet/*.parquet     # full feature snapshot from ADLS
    app/_data/serving_lookup.parquet      # long-format latest delay rate per entity

Outputs (committed, read by app/streamlit_app.py):
    app/_data/eda/*.csv                   # aggregate tables for the EDA charts
    app/_data/serving_lookup.json         # dict-of-dicts the app feeds src/features.py
    app/_data/route_distance.json         # median Distance per origin|dest (distance gap fix)
    app/_data/carriers.json / airports.json  # dropdown option lists
    app/_data/summary.json                # headline numbers for the overview page
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(ROOT, "data_tmp", "train_parquet")
DATA_DIR = os.path.join(ROOT, "app", "_data")
EDA_DIR = os.path.join(DATA_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

# Carrier code -> friendly name (the big US carriers in the BTS data). Anything not
# here just shows the raw IATA code in the app.
CARRIER_NAMES = {
    "AA": "American", "DL": "Delta", "UA": "United", "WN": "Southwest",
    "B6": "JetBlue", "AS": "Alaska", "NK": "Spirit", "F9": "Frontier",
    "G4": "Allegiant", "HA": "Hawaiian", "OO": "SkyWest", "YX": "Republic",
    "MQ": "Envoy", "OH": "PSA", "9E": "Endeavor", "QX": "Horizon",
    "YV": "Mesa", "CP": "Compass", "EV": "ExpressJet", "ZW": "Air Wisconsin",
    "C5": "Champlain", "PT": "Piedmont", "G7": "GoJet", "9K": "Cape Air",
    "AX": "Trans States", "VX": "Virgin America", "DH": "Independence",
}

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main() -> None:
    parts = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    if not parts:
        raise SystemExit(f"No parquet found in {PARQUET_DIR} — download train_parquet first.")
    print(f"Reading {len(parts)} parquet parts ...")
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    n = len(df)
    print(f"Loaded {n:,} rows, {df['FlightDate'].min().date()} -> {df['FlightDate'].max().date()}")

    # ---- headline summary numbers (overview page) --------------------------
    overall_rate = float(df["delayed"].mean())
    summary = {
        "n_rows": int(n),
        "n_carriers": int(df["carrier"].nunique()),
        "n_airports": int(pd.unique(df[["origin", "dest"]].values.ravel()).size),
        "n_routes": int(df.groupby(["origin", "dest"]).ngroups),
        "date_min": str(df["FlightDate"].min().date()),
        "date_max": str(df["FlightDate"].max().date()),
        "overall_delay_rate": overall_rate,
        "trivial_accuracy": float(1 - overall_rate),
    }
    _dump_json(summary, "summary.json")
    print(f"Overall delay rate: {overall_rate:.4f}")

    # ---- class balance ------------------------------------------------------
    bal = (
        df["delayed"].value_counts().rename_axis("delayed").reset_index(name="count")
        .sort_values("delayed")
    )
    bal["label"] = bal["delayed"].map({0: "On time (≤15 min)", 1: "Delayed (>15 min)"})
    bal["share"] = bal["count"] / n
    _csv(bal, "class_balance.csv")

    # ---- delay rate by departure hour --------------------------------------
    by_hour = (
        df.groupby("dep_hour")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    by_hour = by_hour[(by_hour["dep_hour"] >= 0) & (by_hour["dep_hour"] <= 23)]
    _csv(by_hour, "delay_by_hour.csv")

    # ---- delay rate by month (seasonality) ---------------------------------
    by_month = (
        df.groupby("month")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    by_month["month_name"] = by_month["month"].map(lambda m: MONTH_NAMES[int(m)])
    _csv(by_month, "delay_by_month.csv")

    # ---- delay rate by day of week -----------------------------------------
    by_dow = (
        df.groupby("day_of_week")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    by_dow["dow_name"] = by_dow["day_of_week"].map(lambda d: DOW_NAMES[int(d)])
    _csv(by_dow, "delay_by_dow.csv")

    # ---- monthly trend (over the 3 years) ----------------------------------
    trend = (
        df.assign(ym=df["FlightDate"].dt.to_period("M").astype(str))
        .groupby("ym")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    _csv(trend, "delay_trend.csv")

    # ---- worst / best carriers (min volume so tiny carriers don't top it) --
    by_carrier = (
        df.groupby("carrier")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    by_carrier = by_carrier[by_carrier["flights"] > 50000].copy()
    by_carrier["name"] = by_carrier["carrier"].map(lambda c: CARRIER_NAMES.get(c, c))
    by_carrier = by_carrier.sort_values("delay_rate", ascending=False)
    _csv(by_carrier, "delay_by_carrier.csv")

    # ---- busiest origin airports + their delay rate ------------------------
    by_origin = (
        df.groupby("origin")
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    by_origin = by_origin[by_origin["flights"] > 50000].copy()
    _csv(by_origin.sort_values("flights", ascending=False).head(40), "top_origins.csv")
    _csv(by_origin.sort_values("delay_rate", ascending=False).head(25), "worst_origins.csv")

    # ---- delay rate vs distance band ---------------------------------------
    bins = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 6000]
    labels = ["0-250", "250-500", "500-750", "750-1k", "1k-1.5k", "1.5k-2k", "2k-3k", "3k+"]
    band = pd.cut(df["distance"], bins=bins, labels=labels, right=False)
    by_dist = (
        df.assign(band=band).groupby("band", observed=True)
        .agg(flights=("delayed", "size"), delay_rate=("delayed", "mean"))
        .reset_index()
    )
    _csv(by_dist, "delay_by_distance.csv")

    # ---- dropdown option lists ---------------------------------------------
    carriers = sorted(by_carrier["carrier"].tolist())
    carrier_opts = [{"code": c, "name": CARRIER_NAMES.get(c, c)} for c in carriers]
    _dump_json(carrier_opts, "carriers.json")

    airports = sorted(by_origin.sort_values("flights", ascending=False)["origin"].tolist())
    _dump_json(airports, "airports.json")

    # ---- route -> median distance (closes the distance input gap) ----------
    routes = (
        df.groupby(["origin", "dest"])["distance"].median().round().astype(int).reset_index()
    )
    route_map = {f"{r.origin}|{r.dest}": int(r.distance) for r in routes.itertuples()}
    _dump_json(route_map, "route_distance.json")
    print(f"Route distance lookup: {len(route_map):,} routes")

    # ---- serving lookup long -> dict-of-dicts the app feeds src/features ---
    lk = pd.read_parquet(os.path.join(DATA_DIR, "serving_lookup.parquet"))
    lookup: dict[str, dict[str, float]] = {}
    for kind, grp in lk.groupby("entity_kind"):
        lookup[kind] = {str(k): float(v) for k, v in zip(grp["entity_key"], grp["delay_rate"])}
    global_rate = lookup.get("__global__", {}).get("__global__", overall_rate)
    lookup["__global_rate__"] = global_rate  # convenience scalar for the app
    _dump_json(lookup, "serving_lookup.json")
    print(f"Serving lookup kinds: {[k for k in lookup if not k.startswith('__')]}")

    print("\nAll assets written to app/_data/ — these are committed for the deployed app.")


def _csv(frame: pd.DataFrame, name: str) -> None:
    path = os.path.join(EDA_DIR, name)
    frame.to_csv(path, index=False)
    print(f"  wrote {os.path.relpath(path, ROOT)}  ({len(frame)} rows)")


def _dump_json(obj, name: str) -> None:
    path = os.path.join(DATA_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
