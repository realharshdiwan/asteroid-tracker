# 🌌 NASA Asteroid Tracker

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Real-time monitoring and visualization dashboard for Near-Earth Objects (NEO) using NASA's API.**

A production-ready, interactive web application for tracking asteroids, analyzing risk scores, and receiving alerts on potentially hazardous space objects.

---

## ✨ Features

### 🎯 Core Functionality
- **Real-time NASA Data**: Fetches live asteroid data from NASA's NEO Feed API
- **Smart Caching**: 5-minute cache to respect API rate limits
- **Flexible Date Ranges**: Presets (Today, 3 days, 7 days) or custom ranges
- **Advanced Filtering**: Search by name, diameter, distance range

### 📊 Risk Analysis
- **Configurable Risk Scoring**: Weighted algorithm combining:
  - Asteroid diameter (default 50%)
  - Inverse distance (default 30%)
  - Velocity (default 20%)
- **Percentile-Based Normalization**: Fair comparison across all objects
- **Hazard Detection**: Automatic flagging of close approaches

### 📈 Visualizations
- **Danger Map**: Interactive scatter plot (diameter vs distance, log scale)
- **Velocity Distribution**: Histogram with unit toggle (kph/km/s)
- **Daily Timeline**: Asteroid count trends over time

### ⭐ Watchlist Management
- **Persistent Storage**: SQLite-backed watchlist
- **Editable Interface**: Add/remove asteroids dynamically
- **JSON Import/Export**: Share watchlists across sessions
- **Cross-Reference**: View watched items in current date window

### 🔔 Webhook Alerts
- **Automatic Notifications**: POST alerts when hazards detected
- **Configurable Thresholds**: Set custom danger distance
- **Alert History**: SQLite-persisted webhook logs
- **Detailed Payload**: Includes risk scores and metadata

### 🎨 User Experience
- **Dark Theme**: Custom CSS with space-inspired design
- **Responsive Layout**: Wide layout with metric cards
- **Shareable URLs**: Query parameters for bookmarking
- **CSV Export**: Download filtered data

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- NASA API key ([Get one free](https://api.nasa.gov/))

### Installation

```bash
# Clone the repository
git clone https://github.com/realharshdiwan/asteroid-tracker.git
cd asteroid-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your NASA_API_KEY

# Run the application
streamlit run app.py
```

Access the dashboard at `http://localhost:8501`

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t asteroid-tracker .

# Run the container
docker run -p 8501:8501 \
  -e NASA_API_KEY=your_api_key_here \
  asteroid-tracker
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Heroku, Streamlit Cloud, and production deployment guides.

---

## 📖 Documentation

- **[Deployment Guide](docs/DEPLOYMENT.md)**: Docker, Heroku, Streamlit Cloud
- **[Webhook API](docs/API.md)**: Integration guide with examples
- **[Contributing](CONTRIBUTING.md)**: Development setup and guidelines

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NASA_API_KEY` | Recommended | `DEMO_KEY` | NASA API key for higher rate limits |
| `ASTEROID_DB_PATH` | No | `./app_data.db` | SQLite database path |
| `ENVIRONMENT` | No | `development` | Set to `production` for stricter validation |

### Streamlit Configuration

Edit `.streamlit/config.toml` for server settings:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
```

---

## 📊 Architecture

```
┌─────────────┐
│   User UI   │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│  Streamlit Frontend     │
│  - Filters & Controls   │
│  - Visualizations       │
│  - Watchlist UI         │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  Business Logic         │
│  - Risk Calculation     │
│  - Data Filtering       │
│  - Webhook Validation   │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│  Data Layer             │
│  - NASA API Client      │
│  - SQLite Database      │
│  - Caching (5min TTL)   │
└─────────────────────────┘
```

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.50+
- **Data Processing**: Pandas 2.0+, NumPy 1.24+
- **Visualizations**: Plotly 5.0+, Matplotlib 3.7+
- **Database**: SQLite3
- **HTTP Client**: Requests 2.28+
- **Testing**: Pytest 7.0+

---

## 📝 Usage Examples

### Basic Filtering
```python
# Set date range to next 7 days
# Set minimum diameter to 200m
# Search for asteroids containing "2024"
```

### Risk Score Customization
```python
# Adjust weights in sidebar:
# - Diameter: 0.6 (60%)
# - Distance: 0.3 (30%)
# - Velocity: 0.1 (10%)
# Weights auto-normalize to sum = 1.0
```

### Webhook Integration
See [docs/API.md](docs/API.md) for complete webhook documentation and examples.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NASA**: For providing the [NEO Feed API](https://api.nasa.gov/)
- **Streamlit**: For the excellent web framework
- **Plotly**: For interactive visualizations

---

## 📧 Contact

**Harsh Diwan**  
GitHub: [@realharshdiwan](https://github.com/realharshdiwan)

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

**Built with ❤️ for space enthusiasts and developers**


