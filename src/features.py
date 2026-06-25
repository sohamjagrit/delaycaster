"""Reusable feature logic shared between training and serving.

Keeping the feature *contract* in one module is how we prevent training/serving skew:
`03_features.py` materializes the training table using these exact column names and the
base-feature derivations below; the Streamlit app imports the SAME functions to rebuild a
feature vector from raw user inputs + the serving lookup table. If the two ever drift, the
model silently sees inputs it wasn't trained on — the most common production ML bug.

Pure Python / pandas only (no PySpark) so the lightweight Streamlit app can import it
without a Spark runtime. The PySpark feature pipeline mirrors these definitions.
"""

from __future__ import annotations

import datetime as _dt
from typing import Mapping

import pandas as pd

# --- The feature contract -----------------------------------------------------
# Order matters: the model is trained on a DataFrame with exactly these columns in
# this order. Serving must reproduce it 1:1.

CATEGORICAL_FEATURES = ["carrier", "origin", "dest"]

NUMERIC_BASE_FEATURES = ["distance", "dep_hour", "day_of_week", "month", "is_weekend"]

# Rolling 30-day historical delay-rate features. At serve time these come from the
# lookup table (latest value per entity); at train time they're computed as-of each
# flight date in the PySpark pipeline (no leakage — strictly past flights).
DELAY_RATE_FEATURES = [
    "carrier_delay_rate",
    "origin_delay_rate",
    "dest_delay_rate",
    "carrier_origin_delay_rate",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_BASE_FEATURES + DELAY_RATE_FEATURES

TARGET = "delayed"  # 1 if ArrDelay > 15 min else 0

# Fallback when an entity is unseen at serve time (e.g. a new airport). The global
# base rate (~0.20) is the least-surprising prior. 03_features writes the true global
# rate into the lookup table's "__global__" row; this constant is the import-time default.
GLOBAL_DELAY_RATE = 0.20


def derive_base_features(
    carrier: str,
    origin: str,
    dest: str,
    distance: float,
    dep_hour: int,
    flight_date: _dt.date,
) -> dict:
    """Derive the base (non-rolling) features from raw inputs available at scheduled
    departure. `flight_date` gives day_of_week / month / is_weekend; nothing here uses
    information known only after the flight departs (leakage rule)."""
    dow = flight_date.weekday()  # 0=Monday ... 6=Sunday — matches the PySpark pipeline
    return {
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "distance": float(distance),
        "dep_hour": int(dep_hour),
        "day_of_week": dow,
        "month": flight_date.month,
        "is_weekend": int(dow >= 5),
    }


def lookup_delay_rates(
    carrier: str,
    origin: str,
    dest: str,
    lookup: Mapping[str, Mapping[str, float]],
    global_rate: float = GLOBAL_DELAY_RATE,
) -> dict:
    """Pull the latest rolling delay rates for each entity from the serving lookup.

    `lookup` is a dict-of-dicts keyed by entity kind, e.g.::

        {"carrier": {"AA": 0.23, ...},
         "origin":  {"DFW": 0.27, ...},
         "dest":    {"ORD": 0.31, ...},
         "carrier_origin": {"AA|DFW": 0.25, ...}}

    Unseen entities fall back to the global base rate so serving never crashes on a
    carrier/airport the model didn't see in training.
    """
    return {
        "carrier_delay_rate": lookup.get("carrier", {}).get(carrier, global_rate),
        "origin_delay_rate": lookup.get("origin", {}).get(origin, global_rate),
        "dest_delay_rate": lookup.get("dest", {}).get(dest, global_rate),
        "carrier_origin_delay_rate": lookup.get("carrier_origin", {}).get(
            f"{carrier}|{origin}", global_rate
        ),
    }


def build_feature_row(
    carrier: str,
    origin: str,
    dest: str,
    distance: float,
    dep_hour: int,
    flight_date: _dt.date,
    lookup: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Assemble the full single-row feature DataFrame the model expects, in canonical
    column order. This is the function the Streamlit app calls at predict time."""
    row = {
        **derive_base_features(carrier, origin, dest, distance, dep_hour, flight_date),
        **lookup_delay_rates(carrier, origin, dest, lookup),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]
