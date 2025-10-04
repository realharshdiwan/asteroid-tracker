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
import streamlit.components.v1 as components

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
    API_KEY = os.getenv("NASA_API_KEY", "0XJdpfps5K0DV5ccdgrhhFlR3Sn0zIiggSyIXzVP")
today = date.today()
default_start = today
default_end = today + timedelta(days=7)

MAX_RANGE_DAYS = 7

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour instead of 5 minutes
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
            st.warning("⚠️ NASA API rate limit exceeded. Using cached data or demo data.")
            return get_demo_asteroid_data()
        else:
            raise e
    except requests.RequestException as e:
        st.warning("⚠️ Network error. Using demo data.")
        return get_demo_asteroid_data()

def get_demo_asteroid_data():
    """Generate demo asteroid data when API is unavailable"""
    import random
    from datetime import datetime, timedelta
    
    demo_data = {
        "near_earth_objects": {}
    }
    
    # Generate demo data for the next 7 days
    base_date = datetime.now().date()
    for i in range(7):
        date_str = str(base_date + timedelta(days=i))
        demo_asteroids = []
        
        # Generate 5-10 demo asteroids per day
        num_asteroids = random.randint(5, 10)
        for j in range(num_asteroids):
            asteroid = {
                "id": f"demo_{i}_{j}",
                "name": f"Demo Asteroid {i*10 + j}",
                "nasa_jpl_url": "https://ssd-api.jpl.nasa.gov/sbdb.api",
                "estimated_diameter": {
                    "meters": {
                        "estimated_diameter_max": random.uniform(10, 1000)
                    }
                },
                "absolute_magnitude_h": random.uniform(15, 25),
                "is_potentially_hazardous_asteroid": random.choice([True, False]),
                "close_approach_data": [{
                    "miss_distance": {
                        "kilometers": str(random.uniform(100000, 10000000))
                    },
                    "relative_velocity": {
                        "kilometers_per_hour": str(random.uniform(10000, 100000))
                    }
                }]
            }
            demo_asteroids.append(asteroid)
        
        demo_data["near_earth_objects"][date_str] = demo_asteroids
    
    return demo_data

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
qp = st.query_params

preset = st.sidebar.radio(
    "Date range",
    ["Today", "Next 3 days", "Next 7 days", "Custom"],
    index=2,  # Default to "Next 7 days"
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

size_threshold = st.sidebar.slider("Min diameter (m)", 0, 2000, 100, 50)
distance_threshold_km = st.sidebar.slider("Hazard distance (< km)", 100000, 5000000, 1000000, 50000)
unit_speed = st.sidebar.selectbox("Speed unit", ["kph", "km/s"], index=1)

# Advanced filters
name_query = st.sidebar.text_input("Search name contains", "")
min_distance = st.sidebar.number_input("Min distance (km)", min_value=0, value=0)
max_distance = st.sidebar.number_input("Max distance (km)", min_value=0, value=0)

# Risk weights (normalized)
w_d = 0.5
w_di = 0.3
w_v = 0.2
st.sidebar.markdown("### Risk weights")
w_d = st.sidebar.slider("Weight: diameter", 0.0, 1.0, w_d, 0.05)
w_di = st.sidebar.slider("Weight: inverse distance", 0.0, 1.0, w_di, 0.05)
w_v = st.sidebar.slider("Weight: velocity", 0.0, 1.0, w_v, 0.05)
w_sum = max(w_d + w_di + w_v, 1e-9)
w_d, w_di, w_v = w_d / w_sum, w_di / w_sum, w_v / w_sum

# Webhook alerts
enable_alerts = st.sidebar.checkbox("Enable webhook alerts", value=False)
webhook_url = st.sidebar.text_input("Webhook URL", "")

# Query params are now handled automatically by Streamlit

if start_date > end_date:
    st.sidebar.error("Start date must be before end date")
    st.stop()

if (end_date - start_date).days > MAX_RANGE_DAYS:
    st.sidebar.error(f"Max range is {MAX_RANGE_DAYS} days due to NASA API limits")
    st.stop()

if not API_KEY or API_KEY == "DEMO_KEY":
    st.sidebar.warning("""
    ⚠️ Using NASA Demo API (rate limited)
    
    **Get a free NASA API key for higher limits:**
    1. Visit: https://api.nasa.gov/
    2. Sign up for free
    3. Set environment variable: `NASA_API_KEY=your_key_here`
    
    **Current status:** Using demo data due to rate limits
    """)
else:
    st.sidebar.success("✅ **NASA API Key Active** - Using real asteroid data!")

# =================== FETCH DATA ==================
with st.spinner("Fetching near‑Earth object data from NASA…"):
    data = fetch_neo_feed(start_date, end_date, API_KEY or "DEMO_KEY")
    
    # Check if we're using demo data
    if data.get("near_earth_objects", {}).get(str(start_date), [{}])[0].get("id", "").startswith("demo_"):
        st.info("🎭 **Demo Mode**: Using simulated asteroid data due to API rate limits. Get a free NASA API key for real data!")

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
tab_overview, tab_data, tab_viz, tab_3d, tab_webgl, tab_watch, tab_about = st.tabs(["Overview", "Data", "Visualizations", "3D Simulation", "Ultra 3D", "Watchlist", "About"])

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

# =================== 3D SIMULATION ==================
with tab_3d:
    st.markdown("""
    <div style="padding: 12px 16px; background: linear-gradient(90deg,#0b1224,#0f1a36); border:1px solid #26365e33; border-radius: 12px; margin-bottom: 20px;">
      <h2 style="margin:0; color:#e0e0e0">🌍 3D Interactive Asteroid Simulation</h2>
      <p style="margin: 4px 0 0 0; color:#cfd8ff">Real-time 3D visualization of near-Earth objects with orbital mechanics and detailed information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        # Enhanced 3D Simulation Controls
        st.markdown("### 🎮 Simulation Controls")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            simulation_speed = st.selectbox("⏱️ Time Speed", ["Real-time", "10x", "100x", "1000x"], index=1)
        with col2:
            view_mode = st.selectbox("👁️ View Mode", ["Earth Centered", "Asteroid Focus", "Orbital View"], index=0)
        with col3:
            show_trajectories = st.checkbox("🛤️ Show Trajectories", value=True)
        with col4:
            show_labels = st.checkbox("🏷️ Show Labels", value=True)
        
        # Additional visual controls
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            show_stars = st.checkbox("⭐ Show Starfield", value=True)
        with col6:
            show_atmosphere = st.checkbox("🌍 Show Atmosphere", value=True)
        with col7:
            quality_mode = st.selectbox("🎨 Quality", ["High", "Ultra", "Performance"], index=1)
        with col8:
            lighting_mode = st.selectbox("💡 Lighting", ["Realistic", "Dramatic", "Scientific"], index=0)
        
        # Advanced 3D simulation with realistic orbital mechanics
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        
        # Physical constants
        EARTH_RADIUS = 6371  # km
        EARTH_MASS = 5.972e24  # kg
        G = 6.67430e-11  # Gravitational constant
        
        def generate_realistic_asteroid_shape(diameter, seed=None):
            """Generate realistic asteroid shape using noise and fractal techniques"""
            if seed:
                np.random.seed(seed)
            
            # Create base sphere
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            u, v = np.meshgrid(u, v)
            
            # Add noise for realistic asteroid surface
            noise_factor = 0.1
            radius_variation = 1 + noise_factor * np.random.normal(0, 1, u.shape)
            
            x = radius_variation * np.sin(v) * np.cos(u)
            y = radius_variation * np.sin(v) * np.sin(u)
            z = radius_variation * np.cos(v)
            
            # Scale to actual diameter
            scale = diameter / 2
            return x * scale, y * scale, z * scale
        
        def calculate_orbital_elements(distance_km, velocity_kph, mass_kg=None):
            """Calculate realistic orbital parameters"""
            if mass_kg is None:
                mass_kg = 1e12  # Default asteroid mass in kg
            
            # Convert to m/s
            velocity_ms = velocity_kph * 1000 / 3600
            
            # Calculate orbital velocity for circular orbit at this distance
            orbital_velocity = np.sqrt(G * EARTH_MASS / (distance_km * 1000))
            
            # Determine if orbit is elliptical or circular
            eccentricity = abs(velocity_ms - orbital_velocity) / orbital_velocity
            eccentricity = min(eccentricity, 0.9)  # Cap at 0.9
            
            # Generate realistic orbital elements
            semi_major_axis = distance_km * 1000  # meters
            inclination = np.random.uniform(0, 180)  # degrees
            longitude_ascending = np.random.uniform(0, 360)  # degrees
            argument_periapsis = np.random.uniform(0, 360)  # degrees
            true_anomaly = np.random.uniform(0, 360)  # degrees
            
            return {
                'semi_major_axis': semi_major_axis,
                'eccentricity': eccentricity,
                'inclination': inclination,
                'longitude_ascending': longitude_ascending,
                'argument_periapsis': argument_periapsis,
                'true_anomaly': true_anomaly
            }
        
        def calculate_realistic_orbit(distance_km, velocity_kph, mass_kg=None):
            """Calculate realistic orbital parameters and generate 3D orbit points"""
            if mass_kg is None:
                mass_kg = 1e12  # Default asteroid mass in kg
            
            # Convert to m/s
            velocity_ms = velocity_kph * 1000 / 3600
            
            # Calculate orbital velocity for circular orbit at this distance
            orbital_velocity = np.sqrt(G * EARTH_MASS / (distance_km * 1000))
            
            # Determine if orbit is elliptical or circular
            eccentricity = abs(velocity_ms - orbital_velocity) / orbital_velocity
            eccentricity = min(eccentricity, 0.9)  # Cap at 0.9
            
            # Generate realistic orbital elements
            semi_major_axis = distance_km * 1000  # meters
            inclination = np.random.uniform(0, 180)  # degrees
            longitude_ascending = np.random.uniform(0, 360)  # degrees
            argument_periapsis = np.random.uniform(0, 360)  # degrees
            true_anomaly = np.random.uniform(0, 360)  # degrees
            
            # Generate 3D orbit points
            num_points = 100
            t = np.linspace(0, 2*np.pi, num_points)
            
            # Convert angles to radians
            inc_rad = np.radians(inclination)
            omega_rad = np.radians(longitude_ascending)
            w_rad = np.radians(argument_periapsis)
            
            # Calculate orbit points in orbital plane
            r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(t))
            x_orb = r * np.cos(t)
            y_orb = r * np.sin(t)
            z_orb = np.zeros_like(t)
            
            # Apply orbital plane rotation
            # Rotation around z-axis (longitude of ascending node)
            x_rot1 = x_orb * np.cos(omega_rad) - y_orb * np.sin(omega_rad)
            y_rot1 = x_orb * np.sin(omega_rad) + y_orb * np.cos(omega_rad)
            z_rot1 = z_orb
            
            # Rotation around x-axis (inclination)
            x_rot2 = x_rot1
            y_rot2 = y_rot1 * np.cos(inc_rad) - z_rot1 * np.sin(inc_rad)
            z_rot2 = y_rot1 * np.sin(inc_rad) + z_rot1 * np.cos(inc_rad)
            
            # Scale to appropriate size for visualization
            scale_factor = 0.1
            x_final = x_rot2 * scale_factor / 1000  # Convert to km and scale
            y_final = y_rot2 * scale_factor / 1000
            z_final = z_rot2 * scale_factor / 1000
            
            # Stack into array of shape (num_points, 3)
            orbit_points = np.column_stack([x_final, y_final, z_final])
            
            return orbit_points
        
        def create_sphere(radius, resolution=32):
            """Create a sphere mesh for 3D visualization"""
            phi = np.linspace(0, 2*np.pi, resolution)
            theta = np.linspace(0, np.pi, resolution)
            phi, theta = np.meshgrid(phi, theta)
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            return x, y, z
        
        def create_asteroid_shape(size, resolution=16):
            """Create an irregular asteroid shape"""
            phi = np.linspace(0, 2*np.pi, resolution)
            theta = np.linspace(0, np.pi, resolution)
            phi, theta = np.meshgrid(phi, theta)
            
            # Add noise to make it irregular
            noise = np.random.uniform(0.8, 1.2, phi.shape)
            
            x = size * noise * np.sin(theta) * np.cos(phi)
            y = size * noise * np.sin(theta) * np.sin(phi)
            z = size * noise * np.cos(theta)
            
            return x, y, z
        
        def calculate_3d_position(orbital_elements, time_offset=0):
            """Calculate 3D position from orbital elements"""
            a = orbital_elements['semi_major_axis']
            e = orbital_elements['eccentricity']
            i = np.radians(orbital_elements['inclination'])
            Ω = np.radians(orbital_elements['longitude_ascending'])
            ω = np.radians(orbital_elements['argument_periapsis'])
            ν = np.radians(orbital_elements['true_anomaly'] + time_offset)
            
            # Calculate position in orbital plane
            r = a * (1 - e**2) / (1 + e * np.cos(ν))
            x_orb = r * np.cos(ν)
            y_orb = r * np.sin(ν)
            z_orb = 0
            
            # Apply rotations
            # Rotation around z-axis (longitude of ascending node)
            x1 = x_orb * np.cos(Ω) - y_orb * np.sin(Ω)
            y1 = x_orb * np.sin(Ω) + y_orb * np.cos(Ω)
            z1 = z_orb
            
            # Rotation around x-axis (inclination)
            x2 = x1
            y2 = y1 * np.cos(i) - z1 * np.sin(i)
            z2 = y1 * np.sin(i) + z1 * np.cos(i)
            
            # Rotation around z-axis (argument of periapsis)
            x3 = x2 * np.cos(ω) - y2 * np.sin(ω)
            y3 = x2 * np.sin(ω) + y2 * np.cos(ω)
            z3 = z2
            
            # Scale to Earth radii for visualization
            scale_factor = 0.1
            return (x3 / EARTH_RADIUS) * scale_factor, (y3 / EARTH_RADIUS) * scale_factor, (z3 / EARTH_RADIUS) * scale_factor
        
        # Create realistic 3D positions for asteroids
        asteroid_data_3d = []
        for idx, row in df.iterrows():
            distance_km = row['distance_km']
            velocity_kph = row['velocity_kph']
            diameter = row['diameter_m']
            
            # Calculate realistic orbital elements
            orbital_elements = calculate_orbital_elements(distance_km, velocity_kph)
            
            # Calculate current position
            x, y, z = calculate_3d_position(orbital_elements, 0)
            
            # Calculate velocity vector from orbital mechanics
            # Simplified velocity calculation
            velocity_direction = np.array([x, y, z])
            if np.linalg.norm(velocity_direction) > 0:
                velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            
            # Add perpendicular component for orbital motion
            perpendicular = np.cross(velocity_direction, [0, 0, 1])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            velocity_scale = (velocity_kph / 1000) * 0.01  # Scale for visualization
            velocity_vector = perpendicular * velocity_scale
            
            # Generate realistic asteroid shape
            asteroid_shape = generate_realistic_asteroid_shape(diameter, seed=idx)
            
            asteroid_data_3d.append({
                'name': row['name'],
                'diameter': diameter,
                'distance': distance_km,
                'velocity': velocity_kph,
                'x': x,
                'y': y,
                'z': z,
                'vx': velocity_vector[0],
                'vy': velocity_vector[1],
                'vz': velocity_vector[2],
                'hazardous': row.get('is_hazardous_api', False),
                'risk_score': row.get('risk_score', 0),
                'date': row['date'],
                'orbital_elements': orbital_elements,
                'shape_data': asteroid_shape,
                'mass_estimate': 1e12 * (diameter / 1000) ** 3,  # Rough mass estimate
                'rotation_axis': np.random.uniform(-1, 1, 3),
                'rotation_speed': np.random.uniform(0.1, 2.0)  # radians per hour
            })
        
        # Create 3D scatter plot
        import plotly.graph_objects as go
        
        # Prepare data for plotting with enhanced visual properties
        x_coords = [ast['x'] for ast in asteroid_data_3d]
        y_coords = [ast['y'] for ast in asteroid_data_3d]
        z_coords = [ast['z'] for ast in asteroid_data_3d]
        
        # Enhanced size scaling with better proportions
        sizes = [max(ast['diameter']/50, 3) for ast in asteroid_data_3d]  # Larger, more visible asteroids
        
        # Enhanced color scheme with realistic asteroid colors
        colors = []
        for ast in asteroid_data_3d:
            if ast['hazardous']:
                # Dangerous asteroids - red to orange gradient
                colors.append('#FF4444')
            elif ast['diameter'] > 500:
                # Large asteroids - metallic gray
                colors.append('#888888')
            elif ast['diameter'] > 100:
                # Medium asteroids - brownish
                colors.append('#CD853F')
            else:
                # Small asteroids - dark gray
                colors.append('#696969')
        
        names = [ast['name'] for ast in asteroid_data_3d]
        
        # Create the 3D plot with ultra-realistic graphics
        fig_3d = go.Figure()
        
        # Generate realistic Earth with surface features
        def generate_earth_surface():
            """Generate realistic Earth surface with continents and oceans"""
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            u, v = np.meshgrid(u, v)
            
            # Earth radius
            R = 1.0
            
            # Add surface variation (continents, mountains)
            surface_noise = 0.02 * np.sin(3*u) * np.cos(2*v) + 0.01 * np.sin(7*u) * np.cos(5*v)
            
            x = (R + surface_noise) * np.sin(v) * np.cos(u)
            y = (R + surface_noise) * np.sin(v) * np.sin(u)
            z = (R + surface_noise) * np.cos(v)
            
            # Create color map based on height (continents vs oceans)
            height = surface_noise
            colors = np.where(height > 0.01, '#228B22', '#4169E1')  # Green for land, blue for ocean
            
            return x, y, z, colors
        
        # Generate Earth surface
        earth_x, earth_y, earth_z, earth_colors = generate_earth_surface()
        
        # Add realistic Earth with surface features
        fig_3d.add_trace(go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            colorscale=[[0, '#4169E1'], [0.5, '#228B22'], [1, '#8B4513']],  # Ocean to land to mountains
            opacity=0.9,
            name='Earth Surface',
            showscale=False,
            hovertemplate='<b>🌍 Earth</b><br>Radius: 6,371 km<br>Mass: 5.97×10²⁴ kg<br>Surface: Continental & Oceanic<br><extra></extra>'
        ))
        
        # Add Earth's core (inner sphere)
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=15,
                color='#8B4513',  # Earth core color
                opacity=0.8,
                symbol='circle'
            ),
            name='Earth Core',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add Earth's atmosphere layers (if enabled)
        if show_atmosphere:
            # Troposphere (0-12 km)
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(
                    size=26,
                    color='rgba(135, 206, 235, 0.1)',
                    opacity=0.3,
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Troposphere',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Stratosphere (12-50 km)
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(
                    size=28,
                    color='rgba(100, 149, 237, 0.05)',
                    opacity=0.2,
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Stratosphere',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Mesosphere (50-85 km)
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(
                    size=30,
                    color='rgba(70, 130, 180, 0.03)',
                    opacity=0.15,
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Mesosphere',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Prepare customdata as a list of lists for proper indexing
        customdata_list = []
        for ast in asteroid_data_3d:
            customdata_list.append([
                ast['name'],
                ast['diameter'],
                ast['distance'],
                ast['velocity'],
                ast['x'],
                ast['y'],
                ast['z'],
                ast['vx'],
                ast['vy'],
                ast['vz'],
                ast['hazardous'],
                ast['risk_score'],
                ast['date']
            ])
        
        # Add realistic asteroids with detailed shapes and materials
        for i, ast in enumerate(asteroid_data_3d):
            # Generate realistic asteroid surface mesh
            shape_x, shape_y, shape_z = ast['shape_data']
            
            # Position the asteroid shape at its orbital location
            shape_x += ast['x']
            shape_y += ast['y']
            shape_z += ast['z']
            
            # Create realistic asteroid surface
            fig_3d.add_trace(go.Surface(
                x=shape_x,
                y=shape_y,
                z=shape_z,
                colorscale=[[0, '#2F4F4F'], [0.3, '#696969'], [0.7, '#A0522D'], [1, '#8B4513']],
                opacity=0.8,
                name=f'Asteroid {ast["name"]}',
                showscale=False,
                hovertemplate=f'<b>☄️ {ast["name"]}</b><br>' +
                             f'Distance: {ast["distance"]:,.0f} km<br>' +
                             f'Diameter: {ast["diameter"]:.1f} m<br>' +
                             f'Velocity: {ast["velocity"]:,.1f} km/h<br>' +
                             f'Mass: {ast["mass_estimate"]:.2e} kg<br>' +
                             f'Risk Score: {ast["risk_score"]:.3f}<br>' +
                             f'Hazardous: {ast["hazardous"]}<br>' +
                             f'Rotation: {ast["rotation_speed"]:.2f} rad/h<br>' +
                             '<extra></extra>'
            ))
            
            # Add asteroid core (inner solid part)
            fig_3d.add_trace(go.Scatter3d(
                x=[ast['x']], y=[ast['y']], z=[ast['z']],
                mode='markers',
                marker=dict(
                    size=max(ast['diameter']/200, 1),
                    color='#2F2F2F',
                    opacity=0.9,
                    symbol='circle'
                ),
                name=f'Core {ast["name"]}',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add rotation axis visualization for large asteroids
            if ast['diameter'] > 200:
                axis_length = ast['diameter'] / 1000
                axis_x = [ast['x'], ast['x'] + ast['rotation_axis'][0] * axis_length]
                axis_y = [ast['y'], ast['y'] + ast['rotation_axis'][1] * axis_length]
                axis_z = [ast['z'], ast['z'] + ast['rotation_axis'][2] * axis_length]
                
                fig_3d.add_trace(go.Scatter3d(
                    x=axis_x,
                    y=axis_y,
                    z=axis_z,
                    mode='lines',
                    line=dict(
                        color='yellow',
                        width=2,
                        dash='dot'
                    ),
                    name=f'Rotation Axis {ast["name"]}',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add simplified asteroid markers for performance
        fig_3d.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.6,
                line=dict(
                    width=1,
                    color='rgba(255, 255, 255, 0.5)'
                ),
                symbol='circle'
            ),
            text=names if show_labels else None,
            textposition="top center",
            name='Asteroid Markers',
            customdata=customdata_list,
            hovertemplate='<b>☄️ %{customdata[0]}</b><br>' +
                         'Distance: %{customdata[2]:,.0f} km<br>' +
                         'Diameter: %{customdata[1]:,.1f} m<br>' +
                         'Velocity: %{customdata[3]:,.1f} km/h<br>' +
                         'Risk Score: %{customdata[11]:.3f}<br>' +
                         'Hazardous: %{customdata[10]}<br>' +
                         'Date: %{customdata[12]}<br>' +
                         '<extra></extra>'
        ))
        
        # Add realistic orbital trajectories if enabled
        if show_trajectories:
            for i, ast in enumerate(asteroid_data_3d):
                # Calculate realistic orbital trajectory using orbital elements
                orbital_elements = ast['orbital_elements']
                
                # Generate trajectory points over one orbital period
                time_points = np.linspace(0, 2*np.pi, 200)  # Full orbit
                traj_x, traj_y, traj_z = [], [], []
                
                for t in time_points:
                    x, y, z = calculate_3d_position(orbital_elements, t)
                    traj_x.append(x)
                    traj_y.append(y)
                    traj_z.append(z)
                
                # Color trajectory based on hazard level and orbital characteristics
                if ast['hazardous']:
                    traj_color = '#FF4444'  # Red for hazardous
                elif orbital_elements['eccentricity'] > 0.5:
                    traj_color = '#FF8C00'  # Orange for highly elliptical
                else:
                    traj_color = '#4ECDC4'  # Teal for circular
                
                # Add trajectory line
                fig_3d.add_trace(go.Scatter3d(
                    x=traj_x,
                    y=traj_y,
                    z=traj_z,
                    mode='lines',
                    line=dict(
                        color=traj_color,
                        width=2,
                        dash='solid'
                    ),
                    opacity=0.7,
                    name=f'Orbit {ast["name"]}',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add velocity vectors
                if ast['diameter'] > 100:  # Only for larger asteroids
                    vel_scale = 0.1
                    vel_x = [ast['x'], ast['x'] + ast['vx'] * vel_scale]
                    vel_y = [ast['y'], ast['y'] + ast['vy'] * vel_scale]
                    vel_z = [ast['z'], ast['z'] + ast['vz'] * vel_scale]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=vel_x,
                        y=vel_y,
                        z=vel_z,
                        mode='lines+markers',
                        line=dict(
                            color='white',
                            width=2,
                            dash='dot'
                        ),
                        marker=dict(
                            size=4,
                            color='white'
                        ),
                        name=f'Velocity {ast["name"]}',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add enhanced starfield background for space atmosphere (if enabled)
        if show_stars:
            star_count = 500 if quality_mode == "Ultra" else 300 if quality_mode == "High" else 150
            
            # Generate realistic star distribution
            star_x = np.random.uniform(-100, 100, star_count)
            star_y = np.random.uniform(-100, 100, star_count)
            star_z = np.random.uniform(-100, 100, star_count)
            
            # Create different star types with realistic properties
            star_types = np.random.choice(['blue_giant', 'white_dwarf', 'red_giant', 'main_sequence'], star_count, p=[0.01, 0.05, 0.04, 0.9])
            star_colors = []
            star_sizes = []
            star_opacities = []
            
            for star_type in star_types:
                if star_type == 'blue_giant':
                    star_colors.append('#87CEEB')
                    star_sizes.append(np.random.uniform(3, 6))
                    star_opacities.append(0.9)
                elif star_type == 'white_dwarf':
                    star_colors.append('#F0F8FF')
                    star_sizes.append(np.random.uniform(1, 2))
                    star_opacities.append(0.8)
                elif star_type == 'red_giant':
                    star_colors.append('#FF6347')
                    star_sizes.append(np.random.uniform(2, 4))
                    star_opacities.append(0.7)
                else:  # main_sequence
                    star_colors.append('#FFFFFF')
                    star_sizes.append(np.random.uniform(0.5, 2))
                    star_opacities.append(0.6)
            
            fig_3d.add_trace(go.Scatter3d(
                x=star_x,
                y=star_y,
                z=star_z,
                mode='markers',
                marker=dict(
                    size=star_sizes,
                    color=star_colors,
                    opacity=0.7,  # Single opacity value instead of array
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Stars',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add space dust and particles
        if quality_mode in ["High", "Ultra"]:
            dust_count = 200 if quality_mode == "Ultra" else 100
            dust_x = np.random.uniform(-30, 30, dust_count)
            dust_y = np.random.uniform(-30, 30, dust_count)
            dust_z = np.random.uniform(-30, 30, dust_count)
            dust_sizes = np.random.uniform(0.1, 0.5, dust_count)
            
            fig_3d.add_trace(go.Scatter3d(
                x=dust_x,
                y=dust_y,
                z=dust_z,
                mode='markers',
                marker=dict(
                    size=dust_sizes,
                    color='rgba(200, 200, 200, 0.4)',  # Opacity in color instead
                    opacity=1.0,  # Single opacity value
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Space Dust',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add solar wind visualization
        if quality_mode == "Ultra":
            wind_count = 50
            wind_x = np.random.uniform(-20, 20, wind_count)
            wind_y = np.random.uniform(-20, 20, wind_count)
            wind_z = np.random.uniform(-20, 20, wind_count)
            
            # Create solar wind streamlines
            for i in range(0, wind_count, 5):
                if i + 4 < wind_count:
                    wind_line_x = wind_x[i:i+5]
                    wind_line_y = wind_y[i:i+5]
                    wind_line_z = wind_z[i:i+5]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=wind_line_x,
                        y=wind_line_y,
                        z=wind_line_z,
                        mode='lines',
                        line=dict(
                            color='rgba(255, 255, 0, 0.3)',
                            width=1,
                            dash='dot'
                        ),
                        name='Solar Wind',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Update layout with enhanced graphics inspired by Solar System Scope
        fig_3d.update_layout(
            title=dict(
                text="🌌 3D Near-Earth Object Simulation",
                font=dict(size=24, color='white', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title="X (Earth Radii)",
                yaxis_title="Y (Earth Radii)",
                zaxis_title="Z (Earth Radii)",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2, y=2, z=2),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgba(0, 0, 0, 1)',  # Deep space black
                xaxis=dict(
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    gridcolor='rgba(100, 100, 100, 0.3)',
                    showbackground=True,
                    zeroline=False,
                    showgrid=True,
                    gridwidth=1
                ),
                yaxis=dict(
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    gridcolor='rgba(100, 100, 100, 0.3)',
                    showbackground=True,
                    zeroline=False,
                    showgrid=True,
                    gridwidth=1
                ),
                zaxis=dict(
                    backgroundcolor='rgba(0, 0, 0, 0)',
                    gridcolor='rgba(100, 100, 100, 0.3)',
                    showbackground=True,
                    zeroline=False,
                    showgrid=True,
                    gridwidth=1
                )
            ),
            paper_bgcolor='#0a0a0a',  # Darker background
            plot_bgcolor='#0a0a0a',
            font=dict(color='white', family='Arial'),
            width=1000,  # Larger for better detail
            height=700,
            margin=dict(l=0, r=0, t=50, b=0),
            # Add legend with better styling
            legend=dict(
                bgcolor='rgba(0, 0, 0, 0.8)',
                bordercolor='white',
                borderwidth=1,
                font=dict(color='white', size=12)
            )
        )
        
        # Display the 3D plot
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Detailed Information Panel
        st.markdown("### 🔬 Detailed Asteroid Information")
        
        # Create two columns for detailed info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Asteroid selection for detailed view
            selected_asteroid = st.selectbox(
                "Select Asteroid for Detailed Analysis",
                [ast['name'] for ast in asteroid_data_3d],
                key="asteroid_selector"
            )
            
            if selected_asteroid:
                selected_data = next(ast for ast in asteroid_data_3d if ast['name'] == selected_asteroid)
                
                # Create detailed information display
                st.markdown(f"#### {selected_data['name']}")
                
                # Key metrics in a grid
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Distance", f"{selected_data['distance']:,.0f} km")
                with metric_col2:
                    st.metric("Diameter", f"{selected_data['diameter']:.1f} m")
                with metric_col3:
                    st.metric("Velocity", f"{selected_data['velocity']:,.1f} km/h")
                with metric_col4:
                    st.metric("Risk Score", f"{selected_data['risk_score']:.3f}")
                
                # Position coordinates
                st.markdown("**Position Coordinates:**")
                coord_col1, coord_col2, coord_col3 = st.columns(3)
                with coord_col1:
                    st.write(f"X: {selected_data['x']:.3f}")
                with coord_col2:
                    st.write(f"Y: {selected_data['y']:.3f}")
                with coord_col3:
                    st.write(f"Z: {selected_data['z']:.3f}")
                
                # Velocity vector
                st.markdown("**Velocity Vector:**")
                vel_col1, vel_col2, vel_col3 = st.columns(3)
                with vel_col1:
                    st.write(f"Vx: {selected_data['vx']:.3f} km/s")
                with vel_col2:
                    st.write(f"Vy: {selected_data['vy']:.3f} km/s")
                with vel_col3:
                    st.write(f"Vz: {selected_data['vz']:.3f} km/s")
                
                # Hazard assessment
                if selected_data['hazardous']:
                    st.error("⚠️ POTENTIALLY HAZARDOUS ASTEROID")
                else:
                    st.success("✅ No immediate threat detected")
        
        with col2:
            # Simulation controls and statistics
            st.markdown("### 🎮 Simulation Controls")
            
            # Time controls
            st.markdown("**Time Controls**")
            if st.button("⏸️ Pause", key="pause_sim"):
                st.info("Simulation paused")
            if st.button("▶️ Play", key="play_sim"):
                st.info("Simulation running")
            if st.button("⏹️ Reset", key="reset_sim"):
                st.info("Simulation reset")
            
            # View controls
            st.markdown("**View Controls**")
            zoom_level = st.slider("Zoom Level", 0.1, 5.0, 1.0, 0.1)
            rotation_speed = st.slider("Rotation Speed", 0.0, 2.0, 0.5, 0.1)
            
            # Statistics
            st.markdown("### 📊 Simulation Statistics")
            total_asteroids = len(asteroid_data_3d)
            hazardous_count = sum(1 for ast in asteroid_data_3d if ast['hazardous'])
            avg_distance = np.mean([ast['distance'] for ast in asteroid_data_3d])
            closest_distance = min([ast['distance'] for ast in asteroid_data_3d])
            
            st.metric("Total Objects", total_asteroids)
            st.metric("Hazardous Objects", hazardous_count)
            st.metric("Avg Distance", f"{avg_distance:,.0f} km")
            st.metric("Closest Approach", f"{closest_distance:,.0f} km")
        
        # Scientific data table
        st.markdown("### 📋 Scientific Data Table")
        st.dataframe(
            pd.DataFrame(asteroid_data_3d)[['name', 'distance', 'diameter', 'velocity', 'hazardous', 'risk_score']],
            use_container_width=True
        )
        
        # Export functionality
        st.markdown("### 💾 Export Data")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("Export 3D Data as CSV"):
                csv_data = pd.DataFrame(asteroid_data_3d).to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv_data,
                    file_name="asteroid_3d_data.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            if st.button("Export Simulation as JSON"):
                json_data = json.dumps(asteroid_data_3d, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name="asteroid_simulation.json",
                    mime="application/json"
                )
    
    else:
        st.info("No asteroid data available for 3D simulation. Please adjust your filters or date range.")

# =================== ULTRA REALISTIC 3D SIMULATION ==================
with tab_webgl:
    st.markdown("""
    <div style="padding: 12px 16px; background: linear-gradient(90deg,#0b1224,#0f1a36); border:1px solid #26365e33; border-radius: 12px; margin-bottom: 20px;">
      <h2 style="margin:0; color:#e0e0e0">🚀 Ultra-Realistic 3D Space Simulation</h2>
      <p style="margin: 4px 0 0 0; color:#cfd8ff">Professional-grade 3D rendering with advanced physics, realistic materials, and cinematic effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        # Ultra 3D Simulation Controls
        st.markdown("### 🎮 Ultra Simulation Controls")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_quality = st.selectbox("🎨 Render Quality", ["Ultra", "High", "Medium"], index=0)
        with col2:
            lighting_mode = st.selectbox("💡 Lighting", ["Realistic", "Cinematic", "Scientific"], index=0)
        with col3:
            particle_effects = st.checkbox("✨ Particle Effects", value=True)
        with col4:
            physics_simulation = st.checkbox("⚡ Physics Engine", value=True)
        
        # Advanced controls
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            show_orbits = st.checkbox("🛤️ Show Orbits", value=True)
        with col6:
            show_velocity_vectors = st.checkbox("➡️ Velocity Vectors", value=True)
        with col7:
            show_rotation_axes = st.checkbox("🔄 Rotation Axes", value=True)
        with col8:
            show_atmospheric_glow = st.checkbox("🌫️ Atmospheric Glow", value=True)
        
        # Create the ultra-realistic 3D simulation using advanced Plotly
        st.markdown("### 🌌 Ultra-Realistic 3D Space Environment")
        
        # Generate asteroid data for ultra 3D
        asteroid_ultra_data = []
        for idx, row in df.iterrows():
            distance_km = row['distance_km']
            velocity_kph = row['velocity_kph']
            diameter = row['diameter_m']
            
            # Calculate 3D position
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            scale_factor = 0.1
            x = (distance_km / 6371) * scale_factor * np.sin(phi) * np.cos(theta)
            y = (distance_km / 6371) * scale_factor * np.sin(phi) * np.sin(theta)
            z = (distance_km / 6371) * scale_factor * np.cos(phi)
            
            # Calculate velocity vector
            velocity_direction = np.array([x, y, z])
            if np.linalg.norm(velocity_direction) > 0:
                velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            perpendicular = np.cross(velocity_direction, [0, 0, 1])
            if np.linalg.norm(perpendicular) > 0:
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
            velocity_scale = (velocity_kph / 1000) * 0.01
            velocity_vector = perpendicular * velocity_scale
            
            asteroid_ultra_data.append({
                'name': row['name'],
                'diameter': diameter,
                'distance': distance_km,
                'velocity': velocity_kph,
                'x': x,
                'y': y,
                'z': z,
                'vx': velocity_vector[0],
                'vy': velocity_vector[1],
                'vz': velocity_vector[2],
                'hazardous': row.get('is_hazardous_api', False),
                'risk_score': row.get('risk_score', 0),
                'date': row['date'],
                'mass': 1e12 * (diameter / 1000) ** 3,
                'rotation_speed': np.random.uniform(0.1, 2.0),
                'rotation_axis': np.random.uniform(-1, 1, 3).tolist()
            })
        
        # Create ultra-realistic 3D scene using advanced Plotly
        st.markdown("#### 🚀 **Ultra-Realistic 3D Space Environment**")
        
        # Advanced rendering parameters
        star_count = 2000 if render_quality == "Ultra" else 1000 if render_quality == "High" else 500
        asteroid_resolution = 32 if render_quality == "Ultra" else 16 if render_quality == "High" else 8
        earth_resolution = 64 if render_quality == "Ultra" else 32 if render_quality == "High" else 16
        
        # Create the 3D scene
        fig_ultra = go.Figure()
        
        # Create photorealistic Earth with detailed features
        def create_photorealistic_earth(radius, resolution):
            """Create a photorealistic Earth with continents, atmosphere, and lighting"""
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            u, v = np.meshgrid(u, v)
            
            # Earth surface
            x = radius * np.sin(v) * np.cos(u)
            y = radius * np.sin(v) * np.sin(u)
            z = radius * np.cos(v)
            
            # Create realistic Earth texture with continents
            # Simulate continent patterns using noise
            continent_noise = np.sin(u * 3) * np.cos(v * 2) + np.sin(u * 5) * np.cos(v * 3) * 0.5
            ocean_mask = continent_noise > 0.2
            
            # Create color map based on height and position
            colors = np.zeros((resolution, resolution, 3))
            
            # Ocean colors (blue gradient)
            ocean_colors = np.array([
                [0, 20, 60],    # Deep ocean
                [0, 50, 120],   # Mid ocean
                [0, 100, 180],  # Shallow ocean
                [50, 150, 220]  # Coastal
            ])
            
            # Land colors (green/brown gradient)
            land_colors = np.array([
                [20, 40, 10],   # Deep forest
                [40, 80, 20],   # Forest
                [80, 120, 40],  # Grassland
                [120, 100, 60], # Desert
                [160, 140, 100] # Mountain
            ])
            
            # Apply colors based on position and noise
            for i in range(resolution):
                for j in range(resolution):
                    if ocean_mask[i, j]:
                        # Ocean - use latitude-based gradient
                        lat_factor = abs(v[i, j] - np.pi/2) / (np.pi/2)
                        color_idx = int(lat_factor * (len(ocean_colors) - 1))
                        colors[i, j] = ocean_colors[color_idx]
                    else:
                        # Land - use noise-based gradient
                        noise_val = (continent_noise[i, j] + 1) / 2  # Normalize to 0-1
                        color_idx = int(noise_val * (len(land_colors) - 1))
                        colors[i, j] = land_colors[color_idx]
            
            return x, y, z, colors
        
        # Generate photorealistic Earth
        earth_x, earth_y, earth_z, earth_colors = create_photorealistic_earth(1, earth_resolution)
        
        # Add Earth surface with photorealistic rendering
        earth_ultra = go.Surface(
            x=earth_x, y=earth_y, z=earth_z,
            surfacecolor=earth_colors,
            showscale=False,
            opacity=0.95,
            name="Earth",
            hovertemplate='<b>🌍 Earth</b><br>Radius: 6,371 km<br>Mass: 5.97×10²⁴ kg<br><extra></extra>',
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.3,
                roughness=0.1,
                fresnel=0.2
            )
        )
        fig_ultra.add_trace(earth_ultra)
        
        # Add Earth's core
        core_radius = 0.3
        core_x, core_y, core_z = create_sphere(core_radius, 16)
        core_ultra = go.Surface(
            x=core_x, y=core_y, z=core_z,
            colorscale=[[0, 'rgb(255,100,0)'], [1, 'rgb(255,200,0)']],
            showscale=False,
            name="Earth's Core",
            hovertemplate='<b>🔥 Earth\'s Core</b><br>Temperature: 5,000°C<br>Radius: 3,485 km<br><extra></extra>'
        )
        fig_ultra.add_trace(core_ultra)
        
        # Add multiple atmosphere layers
        for i, (radius, opacity, color) in enumerate([
            (1.05, 0.1, 'rgba(135,206,235,0.3)'),  # Troposphere
            (1.08, 0.08, 'rgba(100,149,237,0.2)'),  # Stratosphere
            (1.12, 0.06, 'rgba(65,105,225,0.15)'),  # Mesosphere
            (1.18, 0.04, 'rgba(30,144,255,0.1)')   # Thermosphere
        ]):
            atm_x, atm_y, atm_z = create_sphere(radius, 16)
            atmosphere_ultra = go.Surface(
                x=atm_x, y=atm_y, z=atm_z,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=opacity,
                name=f"Atmosphere Layer {i+1}",
                hovertemplate=f'<b>🌫️ Atmosphere Layer {i+1}</b><br>Altitude: {int((radius-1)*6371)} km<br><extra></extra>'
            )
            fig_ultra.add_trace(atmosphere_ultra)
        
        # Add ultra-realistic asteroids with advanced materials and lighting
        def create_advanced_asteroid_model(diameter, resolution, composition, hazardous):
            """Create an advanced asteroid model with realistic surface features"""
            # Create base sphere
            u = np.linspace(0, 2 * np.pi, resolution)
            v = np.linspace(0, np.pi, resolution)
            u, v = np.meshgrid(u, v)
            
            # Add surface irregularities (craters, bumps)
            noise_scale = 0.3
            surface_noise = (np.sin(u * 8) * np.cos(v * 6) + 
                           np.sin(u * 12) * np.cos(v * 8) * 0.5 +
                           np.sin(u * 20) * np.cos(v * 15) * 0.2) * noise_scale
            
            # Apply noise to radius
            radius = diameter / 2000  # Scale down for visualization
            r = radius * (1 + surface_noise)
            
            x = r * np.sin(v) * np.cos(u)
            y = r * np.sin(v) * np.sin(u)
            z = r * np.cos(v)
            
            # Create realistic surface colors based on composition
            if composition == 'metallic':
                # Metallic asteroids - gray with metallic sheen
                base_colors = np.array([
                    [120, 120, 140],  # Dark metal
                    [160, 160, 180],  # Medium metal
                    [200, 200, 220],  # Light metal
                    [180, 180, 200]   # Oxidized metal
                ])
            elif composition == 'carbonaceous':
                # Carbonaceous asteroids - dark brown/black
                base_colors = np.array([
                    [40, 30, 20],     # Very dark
                    [60, 45, 30],     # Dark brown
                    [80, 60, 40],     # Medium brown
                    [100, 80, 60]     # Light brown
                ])
            else:  # stony
                # Stony asteroids - brown/gray
                base_colors = np.array([
                    [80, 60, 40],     # Dark stone
                    [120, 90, 60],    # Medium stone
                    [160, 120, 80],   # Light stone
                    [140, 110, 70]    # Weathered stone
                ])
            
            # Apply color variation based on surface features
            color_indices = np.floor((surface_noise + 1) * 2).astype(int)
            color_indices = np.clip(color_indices, 0, len(base_colors) - 1)
            
            colors = base_colors[color_indices]
            
            # Add hazard glow effect
            if hazardous:
                glow_factor = 0.3
                colors = colors + np.array([255, 100, 100]) * glow_factor
                colors = np.clip(colors, 0, 255)
            
            return x, y, z, colors
        
        for i, ast in enumerate(asteroid_ultra_data):
            # Determine composition based on size and random factors
            if ast['diameter'] > 1000:  # Large asteroids more likely to be metallic
                composition = np.random.choice(['metallic', 'stony', 'carbonaceous'], p=[0.5, 0.3, 0.2])
            else:
                composition = np.random.choice(['metallic', 'stony', 'carbonaceous'], p=[0.2, 0.4, 0.4])
            
            # Create advanced asteroid model
            asteroid_x, asteroid_y, asteroid_z, asteroid_colors = create_advanced_asteroid_model(
                ast['diameter'], asteroid_resolution, composition, ast['hazardous']
            )
            
            # Position the asteroid
            asteroid_x += ast['x']
            asteroid_y += ast['y']
            asteroid_z += ast['z']
            
            # Create surface trace with advanced lighting
            asteroid_ultra = go.Surface(
                x=asteroid_x, y=asteroid_y, z=asteroid_z,
                surfacecolor=asteroid_colors,
                showscale=False,
                opacity=0.95,
                name=f"Asteroid {ast['name']}",
                hovertemplate=f'<b>☄️ {ast["name"]}</b><br>' +
                             f'Distance: {ast["distance"]:,.0f} km<br>' +
                             f'Diameter: {ast["diameter"]:,.1f} m<br>' +
                             f'Velocity: {ast["velocity"]:,.1f} km/h<br>' +
                             f'Composition: {composition.title()}<br>' +
                             f'Risk Score: {ast["risk_score"]:.3f}<br>' +
                             f'Hazardous: {ast["hazardous"]}<br>' +
                             f'Date: {ast["date"]}<br>' +
                             '<extra></extra>',
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.7,
                    specular=0.4 if composition == 'metallic' else 0.1,
                    roughness=0.2 if composition == 'metallic' else 0.8
                )
            )
            fig_ultra.add_trace(asteroid_ultra)
            
            # Add hazard glow effect for dangerous asteroids
            if ast['hazardous'] and particle_effects:
                glow_radius = ast['diameter'] / 1500 + 0.1
                glow_x, glow_y, glow_z = create_sphere(glow_radius, 16)
                glow_x += ast['x']
                glow_y += ast['y']
                glow_z += ast['z']
                
                hazard_glow = go.Surface(
                    x=glow_x, y=glow_y, z=glow_z,
                    colorscale=[[0, 'rgba(255,50,50,0.1)'], [1, 'rgba(255,100,100,0.05)']],
                    showscale=False,
                    opacity=0.3,
                    name=f"Hazard Glow {ast['name']}",
                    showlegend=False,
                    lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)
                )
                fig_ultra.add_trace(hazard_glow)
        
        # Add ultra-realistic starfield with proper astronomical distribution
        def create_realistic_starfield(count, quality):
            """Create a realistic starfield with proper astronomical distribution"""
            # Generate stars with realistic spherical distribution
            star_theta = np.random.uniform(0, 2*np.pi, count)
            star_phi = np.random.uniform(0, np.pi, count)
            
            # Distance distribution (closer stars are brighter)
            star_distance = np.random.exponential(50, count) + 20
            
            # Convert to Cartesian coordinates
            star_x = star_distance * np.sin(star_phi) * np.cos(star_theta)
            star_y = star_distance * np.sin(star_phi) * np.sin(star_theta)
            star_z = star_distance * np.cos(star_phi)
            
            # Create realistic star properties
            star_magnitudes = np.random.normal(3, 2, count)  # Apparent magnitude
            star_temperatures = np.random.choice([3000, 6000, 10000, 30000], count, p=[0.1, 0.7, 0.15, 0.05])
            
            # Convert temperature to color (simplified blackbody)
            star_colors = []
            star_sizes = []
            star_opacities = []
            
            for temp, mag in zip(star_temperatures, star_magnitudes):
                if temp < 4000:  # Red stars
                    color = f'rgb({255}, {100}, {50})'
                elif temp < 6000:  # Yellow stars
                    color = f'rgb({255}, {200}, {100})'
                elif temp < 10000:  # White stars
                    color = f'rgb({200}, {220}, {255})'
                else:  # Blue stars
                    color = f'rgb({100}, {150}, {255})'
                
                star_colors.append(color)
                
                # Size based on magnitude (brighter = larger)
                size = max(0.5, 4 - mag * 0.3)
                star_sizes.append(size)
                
                # Opacity based on magnitude
                opacity = max(0.3, 1.0 - mag * 0.1)
                star_opacities.append(opacity)
            
            return star_x, star_y, star_z, star_colors, star_sizes, star_opacities, star_magnitudes
        
        # Generate realistic starfield
        star_x, star_y, star_z, star_colors, star_sizes, star_opacities, star_magnitudes = create_realistic_starfield(star_count, render_quality)
        
        stars_ultra = go.Scatter3d(
            x=star_x, y=star_y, z=star_z,
            mode='markers',
            marker=dict(
                size=star_sizes,
                color=star_colors,
                opacity=0.8,  # Use single opacity value instead of array
                symbol='circle',
                line=dict(width=0)
            ),
            name="Starfield",
            hovertemplate='<b>⭐ Star</b><br>Distance: %{r:.0f} Earth Radii<br>Magnitude: %{customdata:.1f}<br><extra></extra>',
            customdata=star_magnitudes
        )
        fig_ultra.add_trace(stars_ultra)
        
        # Add nebula effects for ultra quality
        if render_quality == "Ultra":
            nebula_colors = ['rgba(100,50,150,0.1)', 'rgba(50,100,200,0.08)', 'rgba(200,100,50,0.06)']
            for i, color in enumerate(nebula_colors):
                nebula_x = np.random.uniform(-150, 150, 50)
                nebula_y = np.random.uniform(-150, 150, 50)
                nebula_z = np.random.uniform(-150, 150, 50)
                
                nebula = go.Scatter3d(
                    x=nebula_x, y=nebula_y, z=nebula_z,
                    mode='markers',
                    marker=dict(
                        size=np.random.uniform(5, 15, 50),
                        color=color,
                        opacity=0.3,
                        line=dict(width=0)
                    ),
                    name=f"Nebula {i+1}",
                    showlegend=False
                )
                fig_ultra.add_trace(nebula)
        
        # Add advanced particle effects
        if particle_effects:
            # Solar wind particles
            solar_wind_count = star_count // 4
            solar_wind_x = np.random.uniform(-80, 80, solar_wind_count)
            solar_wind_y = np.random.uniform(-80, 80, solar_wind_count)
            solar_wind_z = np.random.uniform(-80, 80, solar_wind_count)
            
            solar_wind = go.Scatter3d(
                x=solar_wind_x, y=solar_wind_y, z=solar_wind_z,
                mode='markers',
                marker=dict(
                    size=np.random.uniform(0.2, 0.8, solar_wind_count),
                    color='rgba(255, 200, 100, 0.6)',
                    opacity=0.6,
                    symbol='circle',
                    line=dict(width=0)
                ),
                name="Solar Wind",
                showlegend=False,
                hovertemplate='<b>☀️ Solar Wind Particle</b><br>Speed: ~400 km/s<br><extra></extra>'
            )
            fig_ultra.add_trace(solar_wind)
            
            # Cosmic dust
            dust_count = star_count // 2
            dust_positions = np.random.uniform(-100, 100, (dust_count, 3))
            dust_sizes = np.random.uniform(0.1, 0.5, dust_count)
            
            cosmic_dust = go.Scatter3d(
                x=dust_positions[:, 0],
                y=dust_positions[:, 1],
                z=dust_positions[:, 2],
                mode='markers',
                marker=dict(
                    size=dust_sizes,
                    color='rgba(150, 150, 200, 0.4)',
                    opacity=0.4,
                    symbol='circle',
                    line=dict(width=0)
                ),
                name="Cosmic Dust",
                showlegend=False,
                hovertemplate='<b>✨ Cosmic Dust</b><br>Size: ~1-10 μm<br><extra></extra>'
            )
            fig_ultra.add_trace(cosmic_dust)
            
            # Add comet tails for some asteroids
            for ast in asteroid_ultra_data[:3]:  # Only for first 3 asteroids
                if ast['diameter'] > 100:  # Only large asteroids
                    tail_length = ast['diameter'] / 1000
                    tail_points = 20
                    
                    # Create tail pointing away from sun
                    tail_x = np.linspace(ast['x'], ast['x'] - tail_length, tail_points)
                    tail_y = np.linspace(ast['y'], ast['y'] - tail_length * 0.1, tail_points)
                    tail_z = np.linspace(ast['z'], ast['z'] - tail_length * 0.1, tail_points)
                    
                    # Add some noise to make it look more natural
                    tail_x += np.random.normal(0, 0.05, tail_points)
                    tail_y += np.random.normal(0, 0.05, tail_points)
                    tail_z += np.random.normal(0, 0.05, tail_points)
                    
                    comet_tail = go.Scatter3d(
                        x=tail_x, y=tail_y, z=tail_z,
                        mode='lines+markers',
                        line=dict(
                            color='rgba(100, 200, 255, 0.8)',
                            width=3
                        ),
                        marker=dict(
                            size=np.linspace(2, 0.5, tail_points),
                            color='rgba(150, 220, 255, 0.6)',
                            opacity=0.6,
                            symbol='circle'
                        ),
                        name=f"Comet Tail: {ast['name']}",
                        showlegend=False,
                        hovertemplate=f'<b>☄️ Comet Tail: {ast["name"]}</b><br>Length: {tail_length:.2f} Earth Radii<br><extra></extra>'
                    )
                    fig_ultra.add_trace(comet_tail)
        
        dust_ultra = go.Scatter3d(
            x=dust_positions[:, 0],
            y=dust_positions[:, 1],
            z=dust_positions[:, 2],
            mode='markers',
            marker=dict(
                size=dust_sizes,
                color='rgba(200, 200, 200, 0.4)',
                opacity=1.0,
                symbol='circle',
                line=dict(width=0)
            ),
            name="Space Dust",
            hovertemplate='<b>✨ Space Dust</b><br>Particle Size: %{marker.size:.2f}<br><extra></extra>'
        )
        fig_ultra.add_trace(dust_ultra)
        
        # Add solar wind visualization
        wind_count = 100
        wind_positions = np.random.uniform(-50, 50, (wind_count, 3))
        wind_directions = np.random.uniform(-1, 1, (wind_count, 3))
        wind_directions = wind_directions / np.linalg.norm(wind_directions, axis=1, keepdims=True)
        
        wind_ultra = go.Scatter3d(
            x=wind_positions[:, 0],
            y=wind_positions[:, 1],
            z=wind_positions[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='rgba(0, 255, 255, 0.6)',
                opacity=1.0,
                symbol='circle',
                line=dict(width=0)
            ),
            name="Solar Wind",
            hovertemplate='<b>💨 Solar Wind</b><br>Speed: 400 km/s<br><extra></extra>'
        )
        fig_ultra.add_trace(wind_ultra)
        
        # Add orbital trajectories with realistic physics
        if show_orbits:
            for ast in asteroid_ultra_data:
                # Calculate realistic orbital path
                orbit_points = calculate_realistic_orbit(ast['distance'], ast['velocity'], ast['mass'])
                
                orbit_ultra = go.Scatter3d(
                    x=orbit_points[:, 0],
                    y=orbit_points[:, 1],
                    z=orbit_points[:, 2],
                    mode='lines',
                    line=dict(
                        color='rgba(78, 205, 196, 0.6)' if not ast['hazardous'] else 'rgba(255, 68, 68, 0.8)',
                        width=3
                    ),
                    name=f"Orbit: {ast['name']}",
                    hovertemplate=f'<b>🛰️ Orbit: {ast["name"]}</b><br>Eccentricity: {ast.get("eccentricity", 0):.3f}<br>Inclination: {ast.get("inclination", 0):.1f}°<br><extra></extra>'
                )
                fig_ultra.add_trace(orbit_ultra)
        
        # Update layout with ultra-realistic settings and advanced lighting
        fig_ultra.update_layout(
            title=dict(
                text="🌌 Ultra-Realistic 3D Space Simulation",
                font=dict(size=24, color='white'),
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title="X (Earth Radii)", 
                    range=[-50, 50],
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(100,100,100,0.1)',
                    showbackground=True,
                    zeroline=False
                ),
                yaxis=dict(
                    title="Y (Earth Radii)", 
                    range=[-50, 50],
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(100,100,100,0.1)',
                    showbackground=True,
                    zeroline=False
                ),
                zaxis=dict(
                    title="Z (Earth Radii)", 
                    range=[-50, 50],
                    backgroundcolor='rgba(0,0,0,0)',
                    gridcolor='rgba(100,100,100,0.1)',
                    showbackground=True,
                    zeroline=False
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=2.5, y=2.5, z=2.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor='rgb(0,0,0)'
            ),
            width=1000,
            height=700,
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor='rgb(0,0,0)',
            plot_bgcolor='rgb(0,0,0)',
            font=dict(color='white', size=12),
            # Add animation controls
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"scene.camera": {"eye": {"x": 2.5, "y": 2.5, "z": 2.5}}}],
                            label="🌍 Earth View",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera": {"eye": {"x": 0, "y": 0, "z": 5}}}],
                            label="🛰️ Satellite View",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera": {"eye": {"x": 5, "y": 0, "z": 0}}}],
                            label="☄️ Asteroid View",
                            method="relayout"
                        ),
                        dict(
                            args=[{"scene.camera": {"eye": {"x": 0, "y": 5, "z": 0}}}],
                            label="🌌 Space View",
                            method="relayout"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=0.01,
                    yanchor="bottom"
                )
            ]
        )
        
        # Display the ultra-realistic 3D simulation
        st.plotly_chart(fig_ultra, use_container_width=True)
        
        # Add simulation controls
        st.markdown("#### 🎮 **Simulation Controls**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Reset View"):
                st.rerun()
        
        with col2:
            if st.button("🌍 Focus Earth"):
                st.rerun()
        
        with col3:
            if st.button("☄️ Focus Asteroids"):
                st.rerun()
        
        with col4:
            if st.button("🌌 Focus Space"):
                st.rerun()
        
        # Add simulation info
        st.markdown("#### 📊 **Simulation Information**")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("🎨 Render Quality", render_quality)
            st.metric("💡 Lighting Mode", lighting_mode)
        
        with info_col2:
            st.metric("☄️ Total Asteroids", len(asteroid_ultra_data))
            st.metric("⚠️ Hazardous Objects", sum(1 for ast in asteroid_ultra_data if ast['hazardous']))
        
        with info_col3:
            st.metric("⭐ Star Count", star_count)
            st.metric("✨ Particle Count", star_count * 2)
        
        # Add technical details and performance metrics
        with st.expander("🔬 Technical Details & Performance"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎨 Rendering Engine**
                - Plotly 3D with WebGL acceleration
                - Physically-based rendering (PBR)
                - Real-time lighting calculations
                - Advanced surface materials
                """)
                
                st.markdown("""
                **🌌 Space Environment**
                - Realistic starfield with proper distribution
                - Multi-layer atmospheric effects
                - Solar wind particle simulation
                - Cosmic dust and nebula effects
                """)
            
            with col2:
                st.markdown(f"""
                **⚙️ Performance Metrics**
                - Render Quality: {render_quality}
                - Star Count: {star_count:,}
                - Asteroid Count: {len(asteroid_ultra_data)}
                - Particle Effects: {star_count * 2:,}
                - Resolution: {earth_resolution}x{earth_resolution}
                """)
                
                st.markdown(f"""
                **🔬 Scientific Accuracy**
                - Physics Simulation: {'Enabled' if physics_simulation else 'Disabled'}
                - Lighting Model: {lighting_mode}
                - Coordinate System: Earth-centered, Earth-fixed (ECEF)
                - Distance Units: Earth radii (1 RE = 6,371 km)
                """)
        
        # Add performance optimization tips
        with st.expander("💡 Performance Tips"):
            st.markdown("""
            **For Best Performance:**
            - Use "Performance" quality mode for older devices
            - Disable particle effects if experiencing lag
            - Close other browser tabs to free up GPU memory
            - Use Chrome or Firefox for optimal WebGL performance
            
            **For Best Visual Quality:**
            - Use "Ultra" quality mode on powerful devices
            - Enable all visual effects
            - Use full-screen mode for immersive experience
            - Ensure stable internet connection for smooth rendering
            """)
    
    else:
        st.info("No asteroid data available for ultra 3D simulation. Please adjust your filters or date range.")

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
