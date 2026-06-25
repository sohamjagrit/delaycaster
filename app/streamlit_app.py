"""DelayCast — flight delay risk, an IOC decision-support demo.

A 3-page tour of the project:
  1. Overview   — what it is, the Azure pipeline, the leakage rule, the headline metric
  2. Data & EDA — the 20M-row exploration that justifies the modelling choices
  3. Champion   — the registered model card + a live "score this flight" predictor

The app reads only small precomputed assets (see app/assets.py) so it runs on Streamlit
Community Cloud with no Azure access. The champion pickle is the same artifact registered
as `delaycast-champion` v1 in the AzureML Model Registry.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make both this app dir and the repo's src/ importable, regardless of the working
# directory the app is launched from (local `streamlit run`, AppTest, or Cloud).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_APP_DIR)
for _p in (_APP_DIR, os.path.join(_REPO_DIR, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import features as F  # noqa: E402  (shared train/serve feature contract)
import assets  # noqa: E402

st.set_page_config(page_title="DelayCast", page_icon="✈️", layout="wide")

PRIMARY = "#1f4e79"
ACCENT = "#e15759"
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Champion comparison — the final AzureML test metrics (time split @ 2024-07-01).
MODEL_RESULTS = pd.DataFrame(
    [
        {"model": "LightGBM 🏆", "recall": 0.5726, "pr_auc": 0.3499, "precision": 0.3111, "roc_auc": 0.6907, "accuracy": 0.6770},
        {"model": "XGBoost",     "recall": 0.5697, "pr_auc": 0.3464, "precision": 0.3084, "roc_auc": 0.6879, "accuracy": 0.6746},
        {"model": "Logistic Reg.", "recall": 0.5690, "pr_auc": 0.3120, "precision": 0.2820, "roc_auc": 0.6580, "accuracy": 0.6418},
    ]
)


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------
def page_overview() -> None:
    s = assets.summary()
    st.title("✈️ DelayCast")
    st.markdown(
        "#### Will this US domestic flight arrive more than 15 minutes late?\n"
        "A decision-support tool for an airline **Integrated Operations Center (IOC)** — "
        "the go/no-go signal a dispatcher uses to catch at-risk flights *before* delays cascade."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flight records", f"{s['n_rows']/1e6:.1f}M")
    c2.metric("Carriers", s["n_carriers"])
    c3.metric("Airports", s["n_airports"])
    c4.metric("Baseline delay rate", f"{s['overall_delay_rate']*100:.1f}%")
    st.caption(
        f"Real US Bureau of Transportation Statistics on-time data, "
        f"{s['date_min']} → {s['date_max']} ({s['n_routes']:,} unique routes)."
    )

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("End-to-end ML lifecycle on Azure")
        st.markdown(
            """
| Stage | Tool | What happens |
|---|---|---|
| **Ingest** | ADLS Gen2 | 36 monthly BTS files (~20M rows) land in the data lake |
| **Engineer** | Azure Databricks + PySpark | Clean, label, build leakage-safe rolling features |
| **Train** | AzureML + MLflow | LR / XGBoost / LightGBM, every run tracked & compared |
| **Register** | AzureML Model Registry | Champion saved as `delaycast-champion` v1 |
| **Serve** | Streamlit | This app scores a flight on demand |
"""
        )
        st.markdown(
            "**Pipeline:** "
            "`BTS files → ADLS Gen2 (bronze) → Databricks/PySpark features (gold) "
            "→ AzureML training → Model Registry → Streamlit`"
        )
    with right:
        st.subheader("The leakage rule 🚫")
        st.markdown(
            "The most common mistake in flight-delay projects is using columns only known "
            "*after* the flight departs. DelayCast uses **only** information available at "
            "scheduled departure time:"
        )
        st.markdown(
            "- ✅ carrier, origin, dest, distance, scheduled hour, date\n"
            "- ✅ rolling **30-day** historical delay rates (computed strictly from *prior* days)\n"
            "- 🚫 `DepDelay`, `CarrierDelay`, `WeatherDelay`, actual times → **excluded**"
        )
        st.info(
            "Rolling rates use a window ending the **day before** each flight "
            "(`rangeBetween(-30, -1)`), so a flight never sees its own day's outcomes.",
            icon="🔒",
        )

    st.divider()
    m1, m2 = st.columns([1, 1.2])
    with m1:
        st.subheader("Why recall, not accuracy?")
        st.markdown(
            f"Only **{s['overall_delay_rate']*100:.0f}%** of flights are delayed, so a model that "
            f"always says *\"on time\"* already scores **{s['trivial_accuracy']*100:.0f}%** accuracy "
            "— and catches zero delays."
        )
        st.markdown(
            "An IOC would rather investigate a flight that turns out fine than miss one that "
            "cascades, so we optimize **recall** (catch the delays) and **PR-AUC** "
            "(robust to the class imbalance)."
        )
    with m2:
        st.subheader("Champion: LightGBM")
        cc = st.columns(3)
        cc[0].metric("Recall (primary)", "0.573")
        cc[1].metric("PR-AUC", "0.350")
        cc[2].metric("ROC-AUC", "0.691")
        st.caption(
            "Time-based split: trained on 16.7M flights (2022-01 → 2024-06), "
            "tested on 3.6M later flights. See the **Champion model** page for the full comparison."
        )

    st.divider()
    st.caption(
        "Portfolio project demonstrating the Azure ML stack (ADLS Gen2 → Databricks → AzureML). "
        "Random Forest was attempted but canceled — too slow on the 4-core trial node — "
        "an honest note, not a completed model."
    )


# ---------------------------------------------------------------------------
# Page 2 — Data & EDA
# ---------------------------------------------------------------------------
def _rate_bar(df, x, y, title, xlabel, color_by_rate=True, text_pct=True):
    fig = px.bar(df, x=x, y=y, title=title)
    fig.update_traces(
        marker_color=df[y] if color_by_rate else PRIMARY,
        marker_colorscale="RdYlGn_r" if color_by_rate else None,
    )
    if text_pct:
        fig.update_traces(text=[f"{v*100:.0f}%" for v in df[y]], textposition="outside")
    fig.update_layout(
        xaxis_title=xlabel, yaxis_title="delay rate",
        yaxis_tickformat=".0%", coloraxis_showscale=False,
        margin=dict(t=50, b=10), height=380,
    )
    return fig


def page_eda() -> None:
    s = assets.summary()
    st.title("📊 Data & EDA")
    st.markdown(
        f"Exploration of **{s['n_rows']/1e6:.1f} million** real BTS flights "
        f"({s['date_min']} → {s['date_max']}). These are the patterns the rolling-rate "
        "features and the model learn to exploit."
    )

    # --- class balance ---
    st.subheader("1 · Class balance — the headline number")
    bal = assets.load_eda("class_balance.csv")
    cA, cB = st.columns([1, 1.4])
    with cA:
        fig = px.pie(bal, names="label", values="count", hole=0.55,
                     color="label",
                     color_discrete_map={"On time (≤15 min)": "#59a14f",
                                         "Delayed (>15 min)": ACCENT})
        fig.update_layout(height=320, margin=dict(t=10, b=10), showlegend=True,
                          legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width="stretch")
    with cB:
        rate = float(bal.loc[bal["delayed"] == 1, "share"].iloc[0])
        st.metric("Delayed (>15 min)", f"{rate*100:.1f}%")
        st.markdown(
            f"Roughly **1 in 5** flights is delayed. This imbalance is *the* reason the project "
            f"optimizes **recall + PR-AUC** rather than accuracy — a trivial \"never delayed\" "
            f"classifier already scores **{(1-rate)*100:.0f}%** accuracy while catching nothing.\n\n"
            "**15 minutes** is the US DOT definition of an on-time arrival — the industry-standard "
            "threshold."
        )

    st.divider()

    # --- time-of-day + seasonality ---
    st.subheader("2 · When do delays happen?")
    g1, g2 = st.columns(2)
    with g1:
        by_hour = assets.load_eda("delay_by_hour.csv")
        st.plotly_chart(
            _rate_bar(by_hour, "dep_hour", "delay_rate",
                      "Delay rate by scheduled departure hour", "departure hour", text_pct=False),
            width="stretch",
        )
        st.caption("Delays build through the day as disruptions accumulate — early flights are safest.")
    with g2:
        by_month = assets.load_eda("delay_by_month.csv")
        st.plotly_chart(
            _rate_bar(by_month, "month_name", "delay_rate",
                      "Delay rate by month (seasonality)", "month", text_pct=False),
            width="stretch",
        )
        st.caption("Summer thunderstorms and winter holidays drive the seasonal peaks.")

    g3, g4 = st.columns(2)
    with g3:
        by_dow = assets.load_eda("delay_by_dow.csv")
        st.plotly_chart(
            _rate_bar(by_dow, "dow_name", "delay_rate",
                      "Delay rate by day of week", "day of week"),
            width="stretch",
        )
    with g4:
        by_dist = assets.load_eda("delay_by_distance.csv")
        st.plotly_chart(
            _rate_bar(by_dist, "band", "delay_rate",
                      "Delay rate by flight distance", "distance (miles)", text_pct=False),
            width="stretch",
        )

    st.divider()

    # --- trend over 3 years ---
    st.subheader("3 · Delay rate over three years")
    trend = assets.load_eda("delay_trend.csv")
    fig = px.line(trend, x="ym", y="delay_rate", markers=True)
    fig.update_traces(line_color=PRIMARY)
    fig.add_hline(y=s["overall_delay_rate"], line_dash="dash", line_color=ACCENT,
                  annotation_text="3-yr average")
    fig.update_layout(xaxis_title="month", yaxis_title="delay rate",
                      yaxis_tickformat=".0%", height=360, margin=dict(t=20, b=10))
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- carriers + airports ---
    st.subheader("4 · It varies by carrier and airport — so rolling rates are useful")
    h1, h2 = st.columns(2)
    with h1:
        bc = assets.load_eda("delay_by_carrier.csv").sort_values("delay_rate")
        fig = px.bar(bc, x="delay_rate", y="name", orientation="h",
                     title="Delay rate by carrier (>50k flights)")
        fig.update_traces(marker_color=bc["delay_rate"], marker_colorscale="RdYlGn_r",
                          text=[f"{v*100:.0f}%" for v in bc["delay_rate"]], textposition="outside")
        fig.update_layout(xaxis_title="delay rate", yaxis_title="", xaxis_tickformat=".0%",
                          coloraxis_showscale=False, height=480, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with h2:
        wo = assets.load_eda("worst_origins.csv").head(15).sort_values("delay_rate")
        fig = px.bar(wo, x="delay_rate", y="origin", orientation="h",
                     title="Worst origin airports by delay rate (top 15)")
        fig.update_traces(marker_color=wo["delay_rate"], marker_colorscale="RdYlGn_r",
                          text=[f"{v*100:.0f}%" for v in wo["delay_rate"]], textposition="outside")
        fig.update_layout(xaxis_title="delay rate", yaxis_title="", xaxis_tickformat=".0%",
                          coloraxis_showscale=False, height=480, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown(
        "Because the delay rate differs so much across carriers (and airports), the model gets "
        "real signal from the **rolling 30-day delay-rate** features — they summarize *recent* "
        "operational health per carrier / origin / dest / route without leaking the future."
    )

    with st.expander("Busiest origin airports (by volume)"):
        top = assets.load_eda("top_origins.csv").head(20).copy()
        top["delay rate"] = (top["delay_rate"] * 100).round(1).astype(str) + "%"
        top["flights"] = top["flights"].map(lambda v: f"{v:,}")
        st.dataframe(top[["origin", "flights", "delay rate"]], hide_index=True,
                     width="stretch")


# ---------------------------------------------------------------------------
# Page 3 — Champion model + live predictor
# ---------------------------------------------------------------------------
def _risk_band(p: float):
    if p < 0.25:
        return "LOW RISK", "#2e7d32", "🟢"
    if p < 0.45:
        return "AT RISK", "#ef6c00", "🟡"
    return "HIGH RISK", "#c62828", "🔴"


def _gauge(p: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#e8f5e9"},
                {"range": [25, 45], "color": "#fff3e0"},
                {"range": [45, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


def page_champion() -> None:
    st.title("🏆 Champion model — score a flight")
    model = assets.load_model()

    # --- model card ---
    with st.container(border=True):
        st.subheader("Model card")
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "**Algorithm** · LightGBM (gradient-boosted trees)\n\n"
            "**Pipeline** · OneHotEncoder + LGBMClassifier\n\n"
            "**Registry** · `delaycast-champion` v1 (AzureML)"
        )
        c2.markdown(
            "**Training** · 16.7M flights, 2022-01 → 2024-06\n\n"
            "**Test** · 3.6M later flights (time split @ 2024-07-01)\n\n"
            "**Imbalance** · class weights (~20% positive)"
        )
        c3.markdown(
            "**Primary metric** · Recall **0.573**\n\n"
            "**PR-AUC** · 0.350 · **ROC-AUC** · 0.691\n\n"
            "**Why time split** · delay prediction is forecasting — a random split leaks the future"
        )

    st.markdown("##### How the candidates compared")
    g1, g2 = st.columns([1.1, 1])
    with g1:
        show = MODEL_RESULTS.copy()
        for col in ["recall", "pr_auc", "precision", "roc_auc", "accuracy"]:
            show[col] = show[col].map(lambda v: f"{v:.3f}")
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption("LightGBM wins on both primary metrics (recall + PR-AUC). "
                   "RF was canceled — too slow on the 4-core trial node.")
    with g2:
        fig = px.bar(MODEL_RESULTS, x="model", y=["recall", "pr_auc"], barmode="group",
                     title="Recall & PR-AUC by model",
                     color_discrete_sequence=[PRIMARY, ACCENT])
        fig.update_layout(height=320, yaxis_title="score", xaxis_title="",
                          legend_title="", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Score this flight")

    if model is None:
        st.error(
            "Champion model pickle not found under `app/_model/`. Download it with:\n\n"
            "`az ml model download --name delaycast-champion --version 1 "
            "--download-path app/_model -g delaycaster-rg -w delaycaster-aml`"
        )
        return

    carrier_opts = assets.carriers()
    airport_opts = assets.airports()
    lookup = assets.serving_lookup()
    routes = assets.route_distance()
    code_to_name = {c["code"]: c["name"] for c in carrier_opts}

    with st.form("predict"):
        r1 = st.columns(3)
        carrier = r1[0].selectbox("Carrier", [c["code"] for c in carrier_opts],
                                  format_func=lambda c: f"{code_to_name.get(c, c)} ({c})")
        origin = r1[1].selectbox("Origin", airport_opts,
                                 index=airport_opts.index("DFW") if "DFW" in airport_opts else 0)
        dest = r1[2].selectbox("Destination", airport_opts,
                               index=airport_opts.index("ORD") if "ORD" in airport_opts else 1)
        r2 = st.columns(3)
        dep_hour = r2[0].slider("Scheduled departure hour", 0, 23, 17)
        flight_date = r2[1].date_input("Flight date", dt.date(2024, 12, 23))
        dow = flight_date.weekday()
        r2[2].markdown(
            f"<br>**{DOW_NAMES[dow]}, {MONTH_NAMES[flight_date.month-1]}**"
            f"{' · weekend' if dow >= 5 else ''}",
            unsafe_allow_html=True,
        )
        submitted = st.form_submit_button("Predict delay risk", type="primary",
                                          width="stretch")

    if not submitted:
        st.caption("Pick a flight above and hit **Predict**.")
        return

    if origin == dest:
        st.warning("Origin and destination are the same — pick different airports.")
        return

    # Distance: median for this route (closes the input gap), fall back to global median.
    distance = routes.get(f"{origin}|{dest}")
    distance_known = distance is not None
    if not distance_known:
        distance = int(pd.Series(list(routes.values())).median())

    row = F.build_feature_row(carrier, origin, dest, distance, dep_hour, flight_date, lookup)
    proba = float(model.predict_proba(row)[0, 1])
    label, color, emoji = _risk_band(proba)

    st.markdown("###")
    res1, res2 = st.columns([1, 1.1])
    with res1:
        st.plotly_chart(_gauge(proba, color), width="stretch")
    with res2:
        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background:{color}22;"
            f"border-left:6px solid {color}'>"
            f"<span style='font-size:34px'>{emoji} <b>{label}</b></span><br>"
            f"<span style='font-size:20px'>Delay probability: <b>{proba*100:.0f}%</b></span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**{code_to_name.get(carrier, carrier)}** · {origin} → {dest} · "
            f"{DOW_NAMES[dow]} {dep_hour:02d}:00 · ~{distance:,} mi"
            + ("" if distance_known else " *(route unseen — using median distance)*")
        )

    with st.expander("What the model saw (feature vector)"):
        disp = row.T.reset_index()
        disp.columns = ["feature", "value"]
        # Mixed string/float column -> stringify so Arrow can serialize it cleanly.
        disp["value"] = disp["value"].map(
            lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        )
        st.dataframe(disp, hide_index=True, width="stretch")
        st.caption(
            "The 4 `*_delay_rate` features come from the **serving lookup** — the latest rolling "
            "30-day delay rate per carrier / origin / dest / route. The app assembles this exact "
            "vector via `src/features.py`, the same module the training pipeline used → no train/serve skew."
        )


# ---------------------------------------------------------------------------
# Nav
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Data & EDA": page_eda,
    "Champion model": page_champion,
}

with st.sidebar:
    st.markdown("## ✈️ DelayCast")
    st.caption("Flight delay prediction · IOC decision support")
    choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(
        "Built on Azure: ADLS Gen2 · Databricks/PySpark · AzureML + MLflow. "
        "~20M real BTS flight records. Champion: LightGBM (recall 0.573)."
    )

PAGES[choice]()
