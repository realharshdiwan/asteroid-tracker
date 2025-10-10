# Asteroid Tracker Web App

**Asteroid Tracker** is an interactive Streamlit-based web app for **visualizing, analyzing, and monitoring near-Earth asteroids (NEOs)** using real NASA data in near real-time. The app features high-resolution 3D space simulations, live close approach tracking, hazard analysis, and support for multiple scientific APIs.

## 🚀 Features

- **Live asteroid data** from NASA NEO Feed, Horizons, and CNEOS APIs
- **3D visualizations** with Plotly: Orbits, approaches, and simulation of the near-Earth environment
- **Risk analysis**: Palermo and Torino scale display, risk heat maps, customizable alert thresholds
- **Advanced filtering**: By size, distance, velocity, and risk metrics
- **Custom watchlist** management: Track specific asteroids and export/import as JSON
- **Webhook alerts**: Receive warnings about hazardous objects during monitored timeframes
- **Photorealistic visualizations**: Space dust, solar wind, and atmospheric rendering inspired by Solar System Scope
- **Cinematic and scientific display modes**: Adjustable for hardware capability
- **Offline demo mode**: Play safely with sample data if NASA API is unreachable

## 🛠 Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/realharshdiwan/asteroid-tracker.git
   cd asteroid-tracker
   ```

2. **Install dependencies** (create a virtualenv if you wish):
   ```bash
   pip install -r requirements.txt
   ```
   *Or manually:*
   ```bash
   pip install streamlit plotly pandas numpy requests
   ```

3. **Set up a NASA API key** (recommended for higher API limits):
   - Get a free API key from https://api.nasa.gov
   - Set it as an environment variable:  
     `export NASAAPIKEY='your-key-here'`

## ▶️ Usage

To **start the app locally**:
```bash
streamlit run astroapp2.py
```

- You can set the API key via `export NASAAPIKEY=...` (Linux/Mac) or set it in Streamlit Secrets.
- Optional environment variables:
  - `ASTEROIDDBPATH` (SQLite DB location, default: `appdata.db`)

**On first launch**, the app initializes its database and shows sample data if the NASA API is unreachable.

## 🌏 Main Components

- **Sidebar Controls:** Set date range, filtering thresholds, source database, and more.
- **3D Visualization:** Interactive model of Earth/asteroid space.
- **Risk and Statistics Panel:** Palermo/Torino scores, close approaches, and statistical graphics.
- **Watchlist & Alerts:** Save asteroids of interest, export/import watchlists, webhook notifications.
- **Advanced filtering and simulation detail:** Render quality, lighting, physics engine options.

## 📦 Data Sources

- NASA NEO Feed API: Latest asteroid approach data
- NASA Horizons API: Orbital elements and ephemerides
- NASA CNEOS: Impact risk and mission planning
- (Fallback) Sample data for offline/demo use
- [More: See app “About” tab]

## 📜 License

This project is released under the MIT License.

## 🙋‍♂️ Credits

- NASA APIs/JPL CNEOS Team
- Streamlit, Plotly, and the Open Source community

---

**Feel free to adapt this README for your customizations or additional files!** If you want a full example `requirements.txt`, instructions for contributing, or badges, let me know.

