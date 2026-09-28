import os
from datetime import date

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("SELLWISE_API_URL")
if not API_URL:
    try:
        API_URL = st.secrets["SELLWISE_API_URL"]
    except (FileNotFoundError, KeyError):
        API_URL = "http://127.0.0.1:8000"
API_URL = API_URL.rstrip("/")
STATE_NAMES = {"CA": "California", "TX": "Texas", "WI": "Wisconsin"}


@st.cache_data(ttl=60)
def get_api_data(path: str, params: dict[str, str] | None = None) -> dict:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="SellWise | Forecasts", page_icon="S", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --ink: #182622;
        --muted: #62716c;
        --accent: #147d69;
        --rule: #dce5e0;
    }
    .stApp { background: #f7f9f7; color: var(--ink); }
    .block-container { padding-top: 2.2rem; }
    h1, h2, h3 { color: var(--ink); }
    h1 { font-family: Georgia, 'Times New Roman', serif; font-weight: 500; }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--rule); }
    .sellwise-kicker {
        color: var(--accent); font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
    }
    .sellwise-subtitle { color: var(--muted); margin-top: -0.6rem; }
    </style>
    <div class="sellwise-kicker">SELLWISE / M5 EVALUATION</div>
    """,
    unsafe_allow_html=True,
)
st.title("Forecast console")
st.markdown('<div class="sellwise-subtitle">Daily item demand · 28-day horizon</div>', unsafe_allow_html=True)
st.divider()

try:
    health = get_api_data("/health")
    if not health["forecast_available"]:
        st.error("Forecast or calendar artifacts are unavailable. Generate the ensemble forecast, then refresh.")
        st.stop()
    catalog = get_api_data("/catalog")
    horizon = get_api_data("/horizon")
except requests.RequestException as error:
    st.error(f"Could not reach the SellWise API at {API_URL}: {error}")
    st.stop()

stores = catalog["stores"]
if not stores:
    st.warning("No forecast series are available.")
    st.stop()

with st.sidebar:
    st.markdown("### Forecast selection")
    store_id = st.selectbox(
        "Store",
        stores,
        format_func=lambda value: f"{STATE_NAMES.get(value[:2], value[:2])} · {value}",
    )
    try:
        store_catalog = get_api_data("/catalog", {"store_id": store_id})
        department_id = st.selectbox("Department", store_catalog["departments"])
        department_catalog = get_api_data(
            "/catalog", {"store_id": store_id, "department_id": department_id}
        )
    except requests.RequestException as error:
        st.error(f"Could not load the item catalog: {error}")
        st.stop()

    items = department_catalog["items"]
    if not items:
        st.warning("No items are available for this selection.")
        st.stop()
    item_id = st.selectbox("Item", items)

    first_date = date.fromisoformat(horizon["start_date"])
    last_date = date.fromisoformat(horizon["end_date"])
    selected_range = st.date_input(
        "Date range",
        value=(first_date, last_date),
        min_value=first_date,
        max_value=last_date,
        format="YYYY-MM-DD",
    )

if isinstance(selected_range, (tuple, list)):
    if not selected_range:
        range_start, range_end = first_date, last_date
    elif len(selected_range) == 1:
        range_start = range_end = selected_range[0]
    else:
        range_start, range_end = selected_range[0], selected_range[-1]
else:
    range_start = range_end = selected_range

try:
    result = get_api_data(
        "/forecasts",
        {
            "store_id": store_id,
            "department_id": department_id,
            "item_id": item_id,
            "start_date": range_start.isoformat(),
            "end_date": range_end.isoformat(),
        },
    )
except requests.RequestException as error:
    st.error(f"Could not load the forecast: {error}")
    st.stop()

forecast_data = pd.DataFrame(result["data"])
st.markdown(f"### {item_id}")
st.caption(f"{department_id}  ·  {store_id}  ·  {range_start:%b %d} – {range_end:%b %d, %Y}")

if forecast_data.empty:
    st.info("No forecast points fall within the selected date range.")
else:
    forecast_data["date"] = pd.to_datetime(forecast_data["date"])
    forecast_total = forecast_data["forecast"].sum()
    average_per_day = forecast_data["forecast"].mean()
    peak = forecast_data.loc[forecast_data["forecast"].idxmax()]

    total_col, average_col, peak_col = st.columns(3)
    total_col.metric("Forecast total", f"{forecast_total:,.1f} units")
    average_col.metric("Daily average", f"{average_per_day:,.2f} units")
    peak_col.metric("Peak day", f"{peak['forecast']:,.2f} units", peak["date"].strftime("%b %d"))

    chart_col, table_col = st.columns([1.7, 1])
    with chart_col:
        st.markdown("#### Daily forecast")
        chart_data = forecast_data.set_index("date")[["forecast"]]
        st.line_chart(chart_data, color="#147d69", height=340)
    with table_col:
        st.markdown("#### Values")
        display_data = forecast_data[["date", "forecast"]].copy()
        display_data["date"] = display_data["date"].dt.strftime("%b %d, %Y")
        display_data["forecast"] = display_data["forecast"].map(lambda value: f"{value:,.3f}")
        st.dataframe(
            display_data.rename(columns={"date": "Date", "forecast": "Units"}),
            hide_index=True,
            use_container_width=True,
            height=340,
        )

st.divider()
st.markdown("### Model evaluation")
try:
    evaluation = get_api_data("/metrics")
    wrmsse_col, levels_col = st.columns([1, 2])
    wrmsse_col.metric("Overall WRMSSE", f"{evaluation['WRMSSE_final']:.4f}")
    levels = pd.DataFrame(
        {
            "Hierarchy level": [f"Level {level}" for level in range(1, 13)],
            "WRMSSE": [evaluation[f"Level{level}"] for level in range(1, 13)],
        }
    ).set_index("Hierarchy level")
    with levels_col:
        st.bar_chart(levels, color="#c05a3c", height=280)
except (requests.RequestException, KeyError, ValueError) as error:
    st.warning(f"Evaluation metrics are unavailable: {error}")