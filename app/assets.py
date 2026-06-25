"""Cached loaders for the committed app assets (model + precomputed tables).

Everything here is read-only and cached for the Streamlit session. The app needs NO
Azure access at runtime: the champion pickle and all aggregate CSV/JSON files were
produced once by `app/precompute_assets.py` and ship with the repo. That keeps the
Community Cloud deploy a single `streamlit run` with no secrets.
"""

from __future__ import annotations

import glob
import json
import os

import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "_data")
EDA_DIR = os.path.join(DATA_DIR, "eda")
MODEL_GLOB = os.path.join(APP_DIR, "_model", "**", "model.pkl")


@st.cache_resource(show_spinner="Loading champion model…")
def load_model():
    """Load the registered champion pipeline (OneHotEncoder + LGBMClassifier).

    Returns None if the pickle isn't present (e.g. a checkout where the model wasn't
    committed) so the app can degrade gracefully instead of crashing.
    """
    matches = glob.glob(MODEL_GLOB, recursive=True)
    if not matches:
        return None
    return joblib.load(matches[0])


@st.cache_data
def load_json(name: str):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


@st.cache_data
def load_eda(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(EDA_DIR, name))


@st.cache_data
def summary() -> dict:
    return load_json("summary.json")


@st.cache_data
def serving_lookup() -> dict:
    return load_json("serving_lookup.json")


@st.cache_data
def route_distance() -> dict:
    return load_json("route_distance.json")


@st.cache_data
def carriers() -> list[dict]:
    return load_json("carriers.json")


@st.cache_data
def airports() -> list[str]:
    return load_json("airports.json")
