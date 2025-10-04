# Ultra-Lightweight Asteroid Tracker - Maximum Performance
import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date, timedelta
import json

# =================== CONFIG ===================
st.set_page_config(
    page_title="Asteroid Tracker 🚀",
    page_icon="🌌",
    layout="wide"
)

# Minimal CSS
st.markdown("""
<style>
.block-container { background-color: #0e1525; color: #f5f5f5; }
section[data-testid="stSidebar"] { background-color: #151b2e; }
h1, h2, h3 { color: #e0e0e0; }
</style>
""", unsafe_allow_html=True)

# =================== API SETUP ===================
API_KEY = os.getenv("NASA_API_KEY", "0XJdpfps5K0DV5ccdgrhhFlR3Sn0zIiggSyIXzVP")
today = date.today()

@st.cache_data(ttl=3600)
def fetch_neo_feed(start, end, api_key):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start}&end_date={end}&api_key={api_key}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        # Return minimal demo data
        return {
            "near_earth_objects": {
                str(today): [{
                    "id": "demo_1",
                    "name": "Demo Asteroid",
                    "estimated_diameter": {"meters": {"estimated_diameter_max": 100}},
                    "is_potentially_hazardous_asteroid": False,
                    "close_approach_data": [{
                        "miss_distance": {"kilometers": "1000000"},
                        "relative_velocity": {"kilometers_per_hour": "30000"}
                    }]
                }]
            }
        }

# =================== SIDEBAR ==================
st.sidebar.title("⚙️ Controls")

# Simple date selection
days_ahead = st.sidebar.slider("Days ahead", 1, 7, 3)
start_date = today
end_date = today + timedelta(days=days_ahead)

# Simple filters
min_diameter = st.sidebar.slider("Min diameter (m)", 0, 500, 50)
hazard_distance = st.sidebar.slider("Hazard distance (km)", 100000, 2000000, 1000000)

# =================== FETCH DATA ==================
with st.spinner("Loading..."):
    data = fetch_neo_feed(start_date, end_date, API_KEY)

# Process data
records = []
for d, objs in data.get("near_earth_objects", {}).items():
    for obj in objs:
        try:
            diameter = obj.get("estimated_diameter", {}).get("meters", {}).get("estimated_diameter_max", 0)
            distance = float(obj.get("close_approach_data", [{}])[0].get("miss_distance", {}).get("kilometers", 0))
            velocity = float(obj.get("close_approach_data", [{}])[0].get("relative_velocity", {}).get("kilometers_per_hour", 0))
            hazardous = obj.get("is_potentially_hazardous_asteroid", False)
            
            if diameter >= min_diameter:
                records.append({
                    "name": obj.get("name", "Unknown"),
                    "diameter": diameter,
                    "distance": distance,
                    "velocity": velocity,
                    "hazardous": hazardous,
                    "date": d
                })
        except:
            continue

df = pd.DataFrame(records)

# =================== HEADER ==================
st.markdown("# 🌌 Ultra-Lightweight Asteroid Tracker")
st.markdown("*Fast, efficient asteroid monitoring*")

# =================== METRICS ==================
if not df.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Largest", f"{df['diameter'].max():.0f} m")
    with col2:
        st.metric("Closest", f"{df['distance'].min():,.0f} km")
    with col3:
        st.metric("Count", len(df))

# =================== HAZARD CHECK ==================
if not df.empty:
    hazards = df[df["distance"] < hazard_distance]
    
    if not hazards.empty:
        st.error(f"⚠️ {len(hazards)} hazardous objects detected!")
        st.dataframe(hazards[["name", "diameter", "distance", "velocity"]], width='stretch')
    else:
        st.success("✅ All clear!")

# =================== DATA TABLE ==================
if not df.empty:
    st.subheader("📋 Asteroid Data")
    st.dataframe(df[["name", "diameter", "distance", "velocity", "hazardous"]], width='stretch')
    
    # Simple visualization
    st.subheader("📊 Visualization")
    
    # Scatter plot
    fig = px.scatter(
        df, 
        x="diameter", 
        y="distance",
        color="velocity",
        hover_name="name",
        title="Asteroid Size vs Distance",
        width=600,
        height=400
    )
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "asteroids.csv", "text/csv")
else:
    st.info("No asteroids found. Try adjusting filters.")

# =================== FOOTER ==================
st.markdown("---")
st.markdown("**Performance Optimized** | Data: NASA NEO API")
