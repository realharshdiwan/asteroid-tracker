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
tab_overview, tab_data, tab_viz, tab_3d, tab_watch, tab_about = st.tabs(["Overview", "Data", "Visualizations", "3D Simulation", "Watchlist", "About"])

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
        # 3D Simulation Controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            simulation_speed = st.selectbox("Time Speed", ["Real-time", "10x", "100x", "1000x"], index=1)
        with col2:
            view_mode = st.selectbox("View Mode", ["Earth Centered", "Asteroid Focus", "Orbital View"], index=0)
        with col3:
            show_trajectories = st.checkbox("Show Trajectories", value=True)
        with col4:
            show_labels = st.checkbox("Show Labels", value=True)
        
        # Calculate 3D positions based on distance and create orbital simulation
        import numpy as np
        
        # Earth radius for scale (km)
        EARTH_RADIUS = 6371
        
        # Create 3D positions for asteroids
        asteroid_data_3d = []
        for _, row in df.iterrows():
            # Convert distance to 3D coordinates
            distance_km = row['distance_km']
            
            # Create orbital position (simplified - in reality would need proper orbital elements)
            # For visualization, we'll place asteroids in a sphere around Earth
            theta = np.random.uniform(0, 2*np.pi)  # Azimuthal angle
            phi = np.random.uniform(0, np.pi)      # Polar angle
            
            # Scale distance for visualization (Earth radius = 1 unit)
            scale_factor = 0.1  # Scale down for better visualization
            x = (distance_km / EARTH_RADIUS) * scale_factor * np.sin(phi) * np.cos(theta)
            y = (distance_km / EARTH_RADIUS) * scale_factor * np.sin(phi) * np.sin(theta)
            z = (distance_km / EARTH_RADIUS) * scale_factor * np.cos(phi)
            
            # Calculate relative velocity vector (simplified)
            velocity_kmh = row['velocity_kph']
            velocity_direction = np.random.uniform(-1, 1, 3)
            velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            velocity_vector = velocity_direction * (velocity_kmh / 1000)  # Convert to km/s for scale
            
            asteroid_data_3d.append({
                'name': row['name'],
                'diameter': row['diameter_m'],
                'distance': distance_km,
                'velocity': velocity_kmh,
                'x': x,
                'y': y,
                'z': z,
                'vx': velocity_vector[0],
                'vy': velocity_vector[1],
                'vz': velocity_vector[2],
                'hazardous': row.get('is_hazardous_api', False),
                'risk_score': row.get('risk_score', 0),
                'date': row['date']
            })
        
        # Create 3D scatter plot
        import plotly.graph_objects as go
        
        # Prepare data for plotting
        x_coords = [ast['x'] for ast in asteroid_data_3d]
        y_coords = [ast['y'] for ast in asteroid_data_3d]
        z_coords = [ast['z'] for ast in asteroid_data_3d]
        sizes = [max(ast['diameter']/100, 2) for ast in asteroid_data_3d]  # Scale asteroid sizes
        colors = ['red' if ast['hazardous'] else 'blue' for ast in asteroid_data_3d]
        names = [ast['name'] for ast in asteroid_data_3d]
        
        # Create the 3D plot
        fig_3d = go.Figure()
        
        # Add Earth at center
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=20,
                color='lightblue',
                opacity=0.8,
                symbol='circle'
            ),
            name='Earth',
            text=['Earth'],
            hovertemplate='<b>Earth</b><br>Radius: 6,371 km<br><extra></extra>'
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
        
        # Add asteroids
        fig_3d.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=names if show_labels else None,
            textposition="top center",
            name='Asteroids',
            customdata=customdata_list,
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Distance: %{customdata[2]:,.0f} km<br>' +
                         'Diameter: %{customdata[1]:,.1f} m<br>' +
                         'Velocity: %{customdata[3]:,.1f} km/h<br>' +
                         'Risk Score: %{customdata[11]:.3f}<br>' +
                         'Hazardous: %{customdata[10]}<br>' +
                         'Date: %{customdata[12]}<br>' +
                         '<extra></extra>'
        ))
        
        # Add trajectory lines if enabled
        if show_trajectories:
            for ast in asteroid_data_3d:
                # Create a simple trajectory line (in reality this would be calculated from orbital elements)
                t = np.linspace(0, 1, 50)
                traj_x = [ast['x'] + ast['vx'] * ti for ti in t]
                traj_y = [ast['y'] + ast['vy'] * ti for ti in t]
                traj_z = [ast['z'] + ast['vz'] * ti for ti in t]
                
                fig_3d.add_trace(go.Scatter3d(
                    x=traj_x,
                    y=traj_y,
                    z=traj_z,
                    mode='lines',
                    line=dict(
                        color='gray',
                        width=1,
                        dash='dot'
                    ),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout for scientific appearance
        fig_3d.update_layout(
            title="3D Near-Earth Object Simulation",
            scene=dict(
                xaxis_title="X (Earth Radii)",
                yaxis_title="Y (Earth Radii)",
                zaxis_title="Z (Earth Radii)",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='black',
                xaxis=dict(backgroundcolor='black', gridcolor='gray', showbackground=True),
                yaxis=dict(backgroundcolor='black', gridcolor='gray', showbackground=True),
                zaxis=dict(backgroundcolor='black', gridcolor='gray', showbackground=True)
            ),
            paper_bgcolor='#0e1525',
            plot_bgcolor='#0e1525',
            font=dict(color='white'),
            width=800,
            height=600
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
