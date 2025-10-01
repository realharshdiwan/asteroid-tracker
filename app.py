# asteroid_tracker.py
import requests # type: ignore
import pandas as pd
import streamlit as st # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore

# nasa api endpoint
API_KEY = "ygmXmVdEN4pooz0ze9vsOmhObeJ2uggfDOoETvBM"  # replace w/ your key from https://api.nasa.gov
url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date=2025-08-22&end_date=2025-08-29&api_key={API_KEY}"

# fetch data
resp = requests.get(url)
data = resp.json()

# parse asteroids into dataframe
records = []
for date, objs in data["near_earth_objects"].items():
    for obj in objs:
        name = obj["name"]
        diameter = obj["estimated_diameter"]["meters"]["estimated_diameter_max"]
        approach = obj["close_approach_data"][0]
        distance = float(approach["miss_distance"]["kilometers"])
        velocity = float(approach["relative_velocity"]["kilometers_per_hour"])
        records.append([date, name, diameter, distance, velocity])

df = pd.DataFrame(records, columns=["date", "name", "diameter_m", "distance_km", "velocity_kph"])

# streamlit UI
st.title("🌌 Asteroid Tracker")
st.write("Using NASA NEO API to monitor near-Earth objects.")

# data table
st.subheader("Upcoming Asteroids")
st.dataframe(df)

# plot: distance vs diameter
st.subheader("Asteroid Size vs Distance")
fig, ax = plt.subplots()
ax.scatter(df["diameter_m"], df["distance_km"], c="red", alpha=0.6)
ax.set_xlabel("Diameter (m)")
ax.set_ylabel("Miss Distance (km)")
ax.set_title("Asteroid Danger Map")
st.pyplot(fig)

# highlight potential danger
danger = df[df["distance_km"] < 1000000]  # < 1M km
if not danger.empty:
    st.subheader("⚠️ Potentially Hazardous Objects")
    st.dataframe(danger[["date", "name", "diameter_m", "distance_km"]])
else:
    st.success("No close approaches detected in this window 🚀")
