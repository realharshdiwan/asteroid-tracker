## Asteroid Tracker 🚀

Interactive Streamlit dashboard for NASA Near Earth Objects (NEO).

### Features
- Date presets and custom range (max 7 days)
- Advanced filters: name search, min/max distance, min diameter
- Risk score combining size, distance, and speed; percentiles; configurable weights
- Visualizations: danger map, speed histogram, daily timeline
- Watchlist with session persistence and editable table
- Webhook alerts when hazards detected; alert history persisted in SQLite
- CSV download, shareable URL query params
 - JSON import/export for watchlist

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Set NASA API key (recommended):
```bash
export NASA_API_KEY=your_key_here
```
   Or add to `.streamlit/secrets.toml`:
```toml
[default]
NASA_API_KEY = "your_key_here"
```
4. (Optional) configure database path for persistence:
```bash
export ASTEROID_DB_PATH=/path/to/app_data.db
```
3. Run:
```bash
streamlit run app.py
```

### Webhook Alerts
Enable in the sidebar and provide a `POST` endpoint. Payload:
```json
{
  "hazards": [ {"date":"...","name":"...","diameter_m":...,"distance_km":...,"velocity_kph":...,"risk_score":...} ],
  "preset": "Next 7 days",
  "range": ["YYYY-MM-DD","YYYY-MM-DD"]
}
```

Recent alerts are stored in SQLite. You can change weights for the risk score in the sidebar; they are normalized automatically and synced to the URL.

### Notes
- NASA API limits apply; caching is enabled for 5 minutes.
- Distance axis in the danger map uses log scale.

