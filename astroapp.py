
import streamlit as st
import requests
import pandas as pd

st.set_page_config(layout="wide")

class CNEOSAPI:
    BASE_URL = "https://ssd-api.jpl.nasa.gov"

    def get_sentry_data(self):
        """Fetches Sentry risk data."""
        sentry_url = f"{self.BASE_URL}/sentry.api"
        response = requests.get(sentry_url)
        response.raise_for_status()
        return response.json()

    def get_discovery_data(self):
        """Fetches newly discovered objects."""
        discovery_url = "https://cneos.jpl.nasa.gov/api/v1/disc"
        response = requests.get(discovery_url)
        response.raise_for_status()
        return response.json()

    def get_horizons_data(self, des):
        """Fetches Horizons data for a specific object."""
        horizons_url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=''{des}''&OBJ_DATA=''YES''&EPHEM_TYPE=''ELEMENTS''"
        response = requests.get(horizons_url)
        response.raise_for_status()
        return response.text

def main():
    st.title("CNEOS API Explorer")
    api = CNEOSAPI()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentry Risk", "Discovery Hub", "Visualizer", "Horizons", "Eyes on Asteroids"])

    with tab1:
        st.header("Sentry Risk Data")
        try:
            sentry_data = api.get_sentry_data()
            if sentry_data and 'data' in sentry_data:
                df_sentry = pd.DataFrame(sentry_data['data'])
                st.dataframe(df_sentry)
            else:
                st.write("No Sentry data available.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching Sentry data: {e}")

    with tab2:
        st.header("Discovery Hub")
        try:
            discovery_data = api.get_discovery_data()
            if discovery_data and 'data' in discovery_data:
                df_discovery = pd.DataFrame(discovery_data['data'])
                st.dataframe(df_discovery)
            else:
                st.write("No discovery data available.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching discovery data: {e}")

    with tab3:
        st.header("NEO Visualizer")
        st.iframe("https://cneos.jpl.nasa.gov/ov/", height=800)

    with tab4:
        st.header("Horizons")
        des = st.text_input("Enter object designation (e.g., 433):")
        if st.button("Get Horizons Data"):
            if des:
                try:
                    horizons_data = api.get_horizons_data(des)
                    st.text(horizons_data)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching Horizons data: {e}")
            else:
                st.warning("Please enter an object designation.")

    with tab5:
        st.header("Eyes on Asteroids")
        st.iframe("https://eyes.nasa.gov/apps/asteroids/#/home", height=800)

if __name__ == "__main__":
    main()
