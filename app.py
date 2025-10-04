# app.py
import os
import json
import sqlite3
from pathlib import Path
from datetime import date, timedelta, datetime
from urllib.parse import urlencode
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

# =================== CONFIG ===================
st.set_page_config(
    page_title="Asteroid Tracker 🚀",
    page_icon="🌌",
    layout="wide"
)

# custom CSS theme
st.markdown(
    """
    <style>
    /* main background + text */
    .block-container {
        background-color: #0e1525;
        color: #f5f5f5;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background-color: #151b2e;
    }

    /* headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
    }

    /* cards/metrics */
    div[data-testid="stMetric"] {
        background-color: #121a2e;
        border-radius: 12px;
        padding: 12px;
        margin: 6px 0;
        border: 1px solid #26365e33;
    }

    /* buttons */
    button[kind="primary"], .stDownloadButton>button {
        background-color: #36508f !important;
        color: #f5f5f5 !important;
        border-radius: 10px !important;
        border: none !important;
    }
    button[kind="secondary"] {
        background-color: #1b2235 !important;
        color: #f5f5f5 !important;
        border-radius: 10px !important;
        border: none !important;
    }

    /* inputs */
    input, textarea, select {
        background-color: #1b2235 !important;
        color: #f5f5f5 !important;
        border-radius: 8px !important;
        border: 1px solid #30406b !important;
    }

    /* dataframes */
    .stDataFrame {
        background-color: #0f1729 !important;
        border-radius: 10px;
        border: 1px solid #26365e33;
    }
    </style>
    """,
    unsafe_allow_html=True
)


try:
    API_KEY = os.getenv("NASA_API_KEY") or st.secrets.get("NASA_API_KEY", "")
except st.errors.StreamlitSecretNotFoundError:
    API_KEY = os.getenv("NASA_API_KEY", "")
today = date.today()
default_start = today
default_end = today + timedelta(days=7)

MAX_RANGE_DAYS = 7

@st.cache_data(ttl=300, show_spinner=False)
def fetch_neo_feed(start, end, api_key):
    url = (
        "https://api.nasa.gov/neo/rest/v1/feed"
        f"?start_date={start}&end_date={end}&api_key={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

# ------------------- Persistence (SQLite) -------------------
DB_PATH = os.getenv("ASTEROID_DB_PATH", str(Path.cwd() / "app_data.db"))

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS watchlist (name TEXT PRIMARY KEY)"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS alerts (ts TEXT, payload TEXT)"
    )
    conn.commit()
    conn.close()

def db_get_watchlist():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM watchlist ORDER BY name ASC")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def db_set_watchlist(names):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM watchlist")
    cur.executemany("INSERT INTO watchlist(name) VALUES(?)", [(n,) for n in sorted(set(names))])
    conn.commit()
    conn.close()

def db_add_alert(payload_dict):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alerts(ts, payload) VALUES(?, ?)",
        (datetime.utcnow().isoformat() + "Z", json.dumps(payload_dict)),
    )
    conn.commit()
    conn.close()

def db_get_alerts(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ts, payload FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"ts": ts, **(json.loads(payload) if payload else {})}
        for ts, payload in rows
    ]

init_db()

# =================== SIDEBAR ==================
st.sidebar.title("⚙️ Controls")

# Query params load
qp = st.experimental_get_query_params()

preset = st.sidebar.radio(
    "Date range",
    ["Today", "Next 3 days", "Next 7 days", "Custom"],
    index=qp.get("preset", ["Next 7 days"]).index(qp.get("preset", ["Next 7 days"])[0]) if qp.get("preset") else 2,
    horizontal=False,
)

if preset == "Today":
    start_date = today
    end_date = today
elif preset == "Next 3 days":
    start_date = today
    end_date = today + timedelta(days=3)
elif preset == "Next 7 days":
    start_date = today
    end_date = today + timedelta(days=7)
else:
    start_date = st.sidebar.date_input("Start date", pd.to_datetime(qp.get("start", [str(default_start)])[0]).date())
    end_date = st.sidebar.date_input("End date", pd.to_datetime(qp.get("end", [str(default_end)])[0]).date())

size_threshold = st.sidebar.slider("Min diameter (m)", 0, 2000, int(qp.get("min_d", [100])[0]), 50)
distance_threshold_km = st.sidebar.slider("Hazard distance (< km)", 100000, 5000000, int(qp.get("haz_km", [1000000])[0]), 50000)
unit_speed = st.sidebar.selectbox("Speed unit", ["kph", "km/s"], index=(0 if qp.get("unit", ["km/s"])[0] == "kph" else 1))

# Advanced filters
name_query = st.sidebar.text_input("Search name contains", qp.get("q", [""])[0])
min_distance = st.sidebar.number_input("Min distance (km)", min_value=0, value=int(qp.get("min_km", [0])[0]))
max_distance = st.sidebar.number_input("Max distance (km)", min_value=0, value=int(qp.get("max_km", [0])[0]))

# Risk weights (normalized)
w_d = float(qp.get("w_d", [0.5])[0])
w_di = float(qp.get("w_di", [0.3])[0])
w_v = float(qp.get("w_v", [0.2])[0])
st.sidebar.markdown("### Risk weights")
w_d = st.sidebar.slider("Weight: diameter", 0.0, 1.0, w_d, 0.05)
w_di = st.sidebar.slider("Weight: inverse distance", 0.0, 1.0, w_di, 0.05)
w_v = st.sidebar.slider("Weight: velocity", 0.0, 1.0, w_v, 0.05)
w_sum = max(w_d + w_di + w_v, 1e-9)
w_d, w_di, w_v = w_d / w_sum, w_di / w_sum, w_v / w_sum

# Webhook alerts
enable_alerts = st.sidebar.checkbox("Enable webhook alerts", value=(qp.get("alerts", ["0"])[0] == "1"))
webhook_url = st.sidebar.text_input("Webhook URL", qp.get("wh", [""])[0])

# Sync query params
st.experimental_set_query_params(
    preset=preset,
    start=str(start_date),
    end=str(end_date),
    min_d=str(size_threshold),
    haz_km=str(distance_threshold_km),
    unit=unit_speed,
    q=name_query,
    min_km=str(min_distance),
    max_km=str(max_distance),
    w_d=str(round(w_d, 3)),
    w_di=str(round(w_di, 3)),
    w_v=str(round(w_v, 3)),
    alerts=("1" if enable_alerts else "0"),
    wh=webhook_url,
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

if (end_date - start_date).days > MAX_RANGE_DAYS:
    st.sidebar.error(f"Max range is {MAX_RANGE_DAYS} days due to NASA API limits")
    st.stop()

if not API_KEY:
    st.sidebar.warning("Set NASA_API_KEY in environment or .streamlit/secrets.toml for higher limits.")

# =================== FETCH DATA ==================
with st.spinner("Fetching near‑Earth object data from NASA…"):
    try:
        data = fetch_neo_feed(start_date, end_date, API_KEY or "DEMO_KEY")
    except requests.HTTPError as e:
        st.error(f"API error: {e}")
        st.stop()
    except requests.RequestException as e:
        st.error("Network error while contacting NASA API. Please try again.")
        st.stop()

records = []
index_objects = {}
for d, objs in data.get("near_earth_objects", {}).items():
    for obj in objs:
        name = obj.get("name")
        neo_id = obj.get("id")
        jpl = obj.get("nasa_jpl_url")
        diameter = obj.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max")
        absolute_mag = obj.get("absolute_magnitude_h")
        hazardous = obj.get("is_potentially_hazardous_asteroid", False)
        approach_list = obj.get("close_approach_data", [])
        if not approach_list:
            continue
        approach = approach_list[0]
        try:
            distance = float(approach.get("miss_distance", {}).get("kilometers", "nan"))
            velocity = float(approach.get("relative_velocity", {}).get("kilometers_per_hour", "nan"))
        except (TypeError, ValueError):
            continue
        index_objects[neo_id] = obj
        records.append([d, neo_id, name, diameter, distance, velocity, jpl, absolute_mag, hazardous])

df = pd.DataFrame(
    records,
    columns=[
        "date",
        "id",
        "name",
        "diameter_m",
        "distance_km",
        "velocity_kph",
        "jpl_url",
        "absolute_mag",
        "is_hazardous_api",
    ],
).dropna()
df["date"] = pd.to_datetime(df["date"]).dt.date

# Filters
df = df[df["diameter_m"] >= size_threshold]
if name_query:
    df = df[df["name"].str.contains(name_query, case=False, na=False)]
if max_distance and int(max_distance) > 0:
    df = df[df["distance_km"] <= float(max_distance)]
if min_distance and int(min_distance) > 0:
    df = df[df["distance_km"] >= float(min_distance)]

# Risk score and percentiles
if not df.empty:
    df["p_diameter"] = df["diameter_m"].rank(pct=True)
    df["p_velocity"] = df["velocity_kph"].rank(pct=True)
    df["p_distance_inv"] = (1 - df["distance_km"].rank(pct=True))
    df["risk_score"] = (
        w_d * df["p_diameter"] + w_di * df["p_distance_inv"] + w_v * df["p_velocity"]
    )

# =================== HEADER ==================
st.markdown(
    """
    <div style="padding: 12px 16px; background: linear-gradient(90deg,#0b1224,#0f1a36); border:1px solid #26365e33; border-radius: 12px;">
      <h1 style="margin:0">🌌 Asteroid Tracker</h1>
      <p style="margin: 4px 0 0 0; color:#cfd8ff">Real-time NASA NEO dashboard — interactive, educational, hackathon-ready.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =================== METRICS ==================
col1, col2, col3, col4 = st.columns(4)
if not df.empty:
    largest = df.loc[df["diameter_m"].idxmax()]
    closest = df.loc[df["distance_km"].idxmin()]
    avg_speed_kph = float(df["velocity_kph"].mean())
    count = len(df)

    col1.metric("Largest asteroid", f'{largest["diameter_m"]:.1f} m', largest["name"])
    col2.metric("Closest approach", f'{closest["distance_km"]:.0f} km', closest["name"])
    col3.metric("Avg speed", f'{(avg_speed_kph/1000 if unit_speed=="km/s" else avg_speed_kph):.2f} {unit_speed}')
    col4.metric("Objects", f"{count}")
else:
    st.info("No asteroids match the current filters.")

# =================== DATA TABLE ==================
tab_overview, tab_data, tab_viz, tab_watch, tab_about = st.tabs(["Overview", "Data", "Visualizations", "Watchlist", "About"])

with tab_data:
    st.subheader("📋 Asteroid data")
    st.dataframe(
        df[["date","name","diameter_m","distance_km","velocity_kph","risk_score"]] if "risk_score" in df.columns else df,
        use_container_width=True,
    )
    if not df.empty:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="asteroids.csv", mime="text/csv")

    st.markdown("Select an asteroid to view details:")
    if not df.empty:
        sel_name = st.selectbox("Asteroid", ["(select)"] + df["name"].tolist())
        if sel_name and sel_name != "(select)":
            row = df[df["name"] == sel_name].iloc[0]
            with st.expander(f"Details for {row['name']}"):
                st.write({
                    "date": str(row["date"]),
                    "diameter_m": float(row["diameter_m"]),
                    "distance_km": float(row["distance_km"]),
                    "velocity_kph": float(row["velocity_kph"]),
                    "absolute_mag": float(row.get("absolute_mag", float("nan"))),
                    "api_hazard_flag": bool(row.get("is_hazardous_api", False)),
                    "risk_score": float(row.get("risk_score", float("nan"))),
                })
                if isinstance(row.get("jpl_url"), str) and row.get("jpl_url"):
                    st.markdown(f"[Open in NASA JPL]({row['jpl_url']})")

# =================== VISUALS ==================
with tab_viz:
    if not df.empty:
        st.subheader("🚨 Danger map")
        fig1 = px.scatter(
            df,
            x="diameter_m",
            y="distance_km",
            hover_name="name",
            size="diameter_m",
            color="risk_score" if "risk_score" in df.columns else "velocity_kph",
            labels={
                "diameter_m": "Diameter (m)",
                "distance_km": "Distance (km)",
                "velocity_kph": "Velocity (kph)",
                "risk_score": "Risk score",
            },
            color_continuous_scale="Reds" if "risk_score" in df.columns else "Blues",
        )
        fig1.update_yaxes(type="log")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("⚡ Velocity distribution")
        x_field = "velocity_kph" if unit_speed == "kph" else None
        df_speed = df.copy()
        if unit_speed == "km/s":
            df_speed["velocity_kmps"] = df_speed["velocity_kph"] / 3600.0
            x_field = "velocity_kmps"
        fig2 = px.histogram(
            df_speed,
            x=x_field,
            nbins=24,
            title=f"Asteroid speeds ({unit_speed})",
            color_discrete_sequence=["#29B6F6"],
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🗓️ Timeline: count per day")
        df_daily = (
            df.groupby("date")["name"].count().reset_index().rename(columns={"name": "count"})
        )
        fig3 = px.line(df_daily, x="date", y="count", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

with tab_watch:
    st.subheader("⭐ Watchlist")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = db_get_watchlist()

    # Add from current data
    if not df.empty:
        add_sel = st.multiselect("Add from results", df["name"].tolist())
        if st.button("Add to watchlist") and add_sel:
            st.session_state.watchlist = sorted(list(set(st.session_state.watchlist + add_sel)))

    wl_df = pd.DataFrame({"name": st.session_state.watchlist})
    edited = st.data_editor(wl_df, num_rows="dynamic", use_container_width=True)
    st.session_state.watchlist = edited["name"].dropna().astype(str).tolist()

    cols = st.columns(3)
    if cols[0].button("Save watchlist"):
        db_set_watchlist(st.session_state.watchlist)
        st.success("Watchlist saved.")

    # Export / Import JSON
    if cols[1].button("Export watchlist JSON"):
        export_bytes = json.dumps({"watchlist": st.session_state.watchlist}, indent=2).encode("utf-8")
        st.download_button("Download watchlist.json", export_bytes, file_name="watchlist.json", mime="application/json")
    uploaded = cols[2].file_uploader("Import JSON", type=["json"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            content = json.loads(uploaded.read())
            imported = content.get("watchlist", [])
            if isinstance(imported, list):
                st.session_state.watchlist = sorted(set(map(str, imported)))
                db_set_watchlist(st.session_state.watchlist)
                st.success("Watchlist imported and saved.")
        except Exception:
            st.warning("Failed to import JSON.")

    # Cross-reference details
    if st.session_state.watchlist:
        st.markdown("Details of items in current window:")
        subset = df[df["name"].isin(st.session_state.watchlist)]
        if subset.empty:
            st.info("No watched items in the current date window.")
        else:
            st.dataframe(subset[["date","name","diameter_m","distance_km","velocity_kph","risk_score"]], use_container_width=True)

# =================== ALERT ==================
with tab_overview:
    if not df.empty:
        danger = df[df["distance_km"] < float(distance_threshold_km)]
        safe = df[~df.index.isin(danger.index)]
        if not danger.empty:
            st.error("⚠️ WARNING: potential hazardous objects detected!")
            st.dataframe(danger[["date", "name", "diameter_m", "distance_km"]], use_container_width=True)
        else:
            st.success("✅ No hazardous approaches in this timeframe")

        st.subheader("Recently observed")
        st.dataframe(
            safe.sort_values(["date", "distance_km"]).head(20)[["date", "name", "diameter_m", "distance_km"]],
            use_container_width=True,
        )
        # Alerts webhook
        if enable_alerts and webhook_url and not danger.empty:
            try:
                payload = danger[["date","name","diameter_m","distance_km","velocity_kph","risk_score"]].to_dict(orient="records")
                alert_body = {"hazards": payload, "preset": preset, "range": [str(start_date), str(end_date)]}
                requests.post(webhook_url, json=alert_body, timeout=5)
                db_add_alert(alert_body)
                st.info("Alert webhook posted.")
            except Exception:
                st.warning("Failed to post webhook alert.")
    else:
        st.info("Adjust filters or date range to see results.")

with tab_about:
    st.markdown(
        """
        - Data source: **NASA NEO Feed API**.
        - Speed is reported in kph from the API; you can toggle to km/s.
        - Hazard threshold is configurable in the sidebar.
        - Tip: Narrow the date range and increase the size threshold to highlight larger threats.
        """
    )
    st.markdown("""
    #### Telemetry
    This session has:
    """)
    if "views" not in st.session_state:
        st.session_state.views = 0
    st.session_state.views += 1
    st.write({
        "page_views": st.session_state.views,
        "watchlist_items": len(st.session_state.get("watchlist", [])),
    })
