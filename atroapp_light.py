# Lightweight Asteroid Tracker - Optimized for Performance
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
import numpy as np

# =================== CONFIG ===================
st.set_page_config(
    page_title="Asteroid Tracker 🚀",
    page_icon="🌌",
    layout="wide"
)

# Lightweight CSS theme
st.markdown(
    """
    <style>
    .block-container {
        background-color: #0e1525;
        color: #f5f5f5;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #151b2e;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
    }
    div[data-testid="stMetric"] {
        background-color: #121a2e;
        border-radius: 12px;
        padding: 12px;
        margin: 6px 0;
        border: 1px solid #26365e33;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================== API SETUP ===================
try:
    API_KEY = os.getenv("NASA_API_KEY") or st.secrets.get("NASA_API_KEY", "")
except st.errors.StreamlitSecretNotFoundError:
    API_KEY = os.getenv("NASA_API_KEY", "0XJdpfps5K0DV5ccdgrhhFlR3Sn0zIiggSyIXzVP")

today = date.today()
default_start = today
default_end = today + timedelta(days=7)
MAX_RANGE_DAYS = 7

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_neo_feed(start, end, api_key):
    url = (
        "https://api.nasa.gov/neo/rest/v1/feed"
        f"?start_date={start}&end_date={end}&api_key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        if e.response.status_code == 429:
            st.warning("⚠️ NASA API rate limit exceeded. Using demo data.")
            return get_demo_asteroid_data()
        else:
            raise e
    except requests.RequestException as e:
        st.warning("⚠️ Network error. Using demo data.")
        return get_demo_asteroid_data()

def get_demo_asteroid_data():
    """Generate lightweight demo asteroid data"""
    import random
    
    demo_data = {"near_earth_objects": {}}
    base_date = datetime.now().date()
    
    for i in range(7):
        date_str = str(base_date + timedelta(days=i))
        demo_asteroids = []
        
        # Generate 3-5 demo asteroids per day (reduced from 5-10)
        num_asteroids = random.randint(3, 5)
        for j in range(num_asteroids):
            asteroid = {
                "id": f"demo_{i}_{j}",
                "name": f"Demo Asteroid {i*5 + j}",
                "nasa_jpl_url": "https://ssd-api.jpl.nasa.gov/sbdb.api",
                "estimated_diameter": {
                    "meters": {
                        "estimated_diameter_max": random.uniform(10, 500)  # Smaller range
                    }
                },
                "absolute_magnitude_h": random.uniform(15, 25),
                "is_potentially_hazardous_asteroid": random.choice([True, False]),
                "close_approach_data": [{
                    "miss_distance": {
                        "kilometers": str(random.uniform(100000, 5000000))
                    },
                    "relative_velocity": {
                        "kilometers_per_hour": str(random.uniform(10000, 50000))  # Lower range
                    }
                }]
            }
            demo_asteroids.append(asteroid)
        
        demo_data["near_earth_objects"][date_str] = demo_asteroids
    
    return demo_data

# =================== SIDEBAR ==================
st.sidebar.title("⚙️ Controls")

# Date range controls
st.sidebar.markdown("### 📅 Date Range")
preset = st.sidebar.radio(
    "Date range",
    ["Today", "Next 3 days", "Next 7 days", "Custom"],
    index=2,
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
    start_date = st.sidebar.date_input("Start date", default_start)
    end_date = st.sidebar.date_input("End date", default_end)

# Filter controls
st.sidebar.markdown("### 🔍 Filters")
size_threshold = st.sidebar.slider("Min diameter (m)", 0, 1000, 50, 25)  # Reduced max
distance_threshold_km = st.sidebar.slider("Hazard distance (< km)", 100000, 2000000, 1000000, 50000)  # Reduced max
unit_speed = st.sidebar.selectbox("Speed unit", ["kph", "km/s"], index=1)

# Advanced filters (simplified)
name_query = st.sidebar.text_input("Search name contains", "")

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

if (end_date - start_date).days > MAX_RANGE_DAYS:
    st.sidebar.error(f"Max range is {MAX_RANGE_DAYS} days due to NASA API limits")
    st.stop()

if not API_KEY or API_KEY == "DEMO_KEY":
    st.sidebar.warning("⚠️ Using NASA Demo API (rate limited)")

# =================== FETCH DATA ==================
with st.spinner("Fetching asteroid data..."):
    data = fetch_neo_feed(start_date, end_date, API_KEY or "DEMO_KEY")
    
    if data.get("near_earth_objects", {}).get(str(start_date), [{}])[0].get("id", "").startswith("demo_"):
        st.info("🎭 **Demo Mode**: Using simulated asteroid data. Get a free NASA API key for real data!")

records = []
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
        records.append([d, neo_id, name, diameter, distance, velocity, jpl, absolute_mag, hazardous])

df = pd.DataFrame(
    records,
    columns=[
        "date", "id", "name", "diameter_m", "distance_km", 
        "velocity_kph", "jpl_url", "absolute_mag", "is_hazardous_api"
    ],
).dropna()
df["date"] = pd.to_datetime(df["date"]).dt.date

# Apply filters
df = df[df["diameter_m"] >= size_threshold]
if name_query:
    df = df[df["name"].str.contains(name_query, case=False, na=False)]

# Simple risk score (no complex calculations)
if not df.empty:
    df["risk_score"] = (
        0.4 * (df["diameter_m"] / df["diameter_m"].max()) +
        0.3 * (1 - df["distance_km"] / df["distance_km"].max()) +
        0.3 * (df["velocity_kph"] / df["velocity_kph"].max())
    )

# =================== HEADER ==================
st.markdown(
    """
    <div style="padding: 12px 16px; background: linear-gradient(90deg,#0b1224,#0f1a36); border:1px solid #26365e33; border-radius: 12px;">
      <h1 style="margin:0">🌌 Lightweight Asteroid Tracker</h1>
      <p style="margin: 4px 0 0 0; color:#cfd8ff">Fast, efficient asteroid monitoring — optimized for performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =================== METRICS ==================
st.markdown("### 📊 Key Metrics")
col1, col2, col3 = st.columns(3)
if not df.empty:
    largest = df.loc[df["diameter_m"].idxmax()]
    closest = df.loc[df["distance_km"].idxmin()]
    avg_speed_kph = float(df["velocity_kph"].mean())
    count = len(df)

    with col1:
        st.metric(
            "Largest Diameter", 
            f'{largest["diameter_m"]:.1f} m',
            help=f"Asteroid: {largest['name']}"
        )
    with col2:
        st.metric(
            "Closest Approach", 
            f'{closest["distance_km"]:.0f} km',
            help=f"Asteroid: {closest['name']}"
        )
    with col3:
        st.metric(
            "Average Velocity", 
            f'{(avg_speed_kph/1000 if unit_speed=="km/s" else avg_speed_kph):.2f} {unit_speed}',
            help=f"Based on {count} objects"
        )
else:
    st.info("No asteroids match the current filters.")

# =================== HAZARD CHECK ==================
if not df.empty:
    hazard_asteroids = df[df["distance_km"] < distance_threshold_km]
    
    if not hazard_asteroids.empty:
        st.error(f"⚠️ **WARNING: {len(hazard_asteroids)} potentially hazardous objects detected within {distance_threshold_km:,.0f} km!**")
        
        st.markdown("#### 🚨 Hazardous Objects")
        st.dataframe(
            hazard_asteroids[["date", "name", "diameter_m", "distance_km", "velocity_kph", "risk_score"]],
            width='stretch'
        )
        
        hazard_csv = hazard_asteroids.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Hazard Data CSV", 
            hazard_csv, 
            file_name="hazard_asteroids.csv", 
            mime="text/csv"
        )
    else:
        st.success(f"✅ **All clear!** No asteroids within {distance_threshold_km:,.0f} km distance threshold.")

# =================== TABS ==================
tab_overview, tab_data, tab_viz = st.tabs(["Overview", "Data", "Visualizations"])

with tab_data:
    st.subheader("📋 Asteroid Data")
    if not df.empty:
        st.dataframe(
            df[["date","name","diameter_m","distance_km","velocity_kph","risk_score"]],
            width='stretch'
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="asteroids.csv", mime="text/csv")
    else:
        st.info("No data available.")

with tab_viz:
    if not df.empty:
        st.subheader("📊 Asteroid Analysis")
        
        # Simple scatter plot
        st.markdown("#### 🌌 Diameter vs Distance (colored by velocity)")
        fig1 = px.scatter(
            df,
            x="diameter_m",
            y="distance_km",
            hover_name="name",
            size="diameter_m",
            color="velocity_kph",
            labels={
                "diameter_m": "Diameter (m)",
                "distance_km": "Distance (km)",
                "velocity_kph": "Velocity (kph)",
            },
            color_continuous_scale="Viridis",
            title="Asteroid Size vs Distance"
        )
        fig1.update_yaxes(type="log")
        fig1.update_layout(
            title_x=0.5,
            font=dict(size=12),
            width=800,  # Fixed width for performance
            height=500
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Simple histogram
        st.markdown("#### ⚡ Velocity Distribution")
        x_field = "velocity_kph" if unit_speed == "kph" else None
        df_speed = df.copy()
        if unit_speed == "km/s":
            df_speed["velocity_kmps"] = df_speed["velocity_kph"] / 3600.0
            x_field = "velocity_kmps"
        
        fig2 = px.histogram(
            df_speed,
            x=x_field,
            nbins=15,  # Reduced bins for performance
            title=f"Asteroid Velocity Distribution ({unit_speed})",
            color_discrete_sequence=["#29B6F6"],
            labels={
                x_field: f"Velocity ({unit_speed})",
                "count": "Number of Asteroids"
            }
        )
        fig2.update_layout(
            title_x=0.5,
            font=dict(size=12),
            width=800,
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available for visualization.")

with tab_overview:
    if not df.empty:
        st.subheader("📊 Summary")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Objects", len(df))
        with col2:
            st.metric("Hazardous Objects", len(df[df["is_hazardous_api"]]))
        with col3:
            st.metric("Avg Distance", f"{df['distance_km'].mean():,.0f} km")
        with col4:
            st.metric("Avg Diameter", f"{df['diameter_m'].mean():.1f} m")
        
        # Recent asteroids
        st.subheader("Recently Observed")
        recent_df = df.sort_values(["date", "distance_km"]).head(10)
        st.dataframe(
            recent_df[["date", "name", "diameter_m", "distance_km"]],
            width='stretch'
        )
    else:
        st.info("No asteroid data available. Adjust your filters or date range.")

# =================== FOOTER ==================
st.markdown("---")
st.markdown(
    """
    **🌌 Lightweight Asteroid Tracker** - Optimized for performance and low resource usage.
    
    **Data Source**: NASA NEO Feed API | **Performance**: Optimized for smooth browsing
    """
)
