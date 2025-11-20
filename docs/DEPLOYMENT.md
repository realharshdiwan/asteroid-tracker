# Deployment Guide

This guide covers deploying the NASA Asteroid Tracker to various platforms.

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t asteroid-tracker .

# Run the container
docker run -p 8501:8501 \
  -e NASA_API_KEY=your_api_key_here \
  asteroid-tracker
```

Access the app at `http://localhost:8501`

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - NASA_API_KEY=${NASA_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

---

## ☁️ Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Git repository initialized

### Steps

```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Set environment variables
heroku config:set NASA_API_KEY=your_api_key_here

# Deploy
git push heroku main

# Open the app
heroku open
```

### Configuration
The `Procfile` is already configured for Heroku deployment.

---

## 🚀 Streamlit Cloud

### Steps

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path: `app.py`
7. Add secrets in "Advanced settings":
   ```toml
   NASA_API_KEY = "your_api_key_here"
   ```
8. Click "Deploy"

---

## 🖥️ Local Development

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd nasa_hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NASA_API_KEY=your_api_key_here  # On Windows: set NASA_API_KEY=your_api_key_here

# Run the app
streamlit run app.py
```

### Using .env file

Create `.env` file (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env with your API key
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NASA_API_KEY` | Recommended | `DEMO_KEY` | NASA API key for higher rate limits |
| `ASTEROID_DB_PATH` | No | `./app_data.db` | SQLite database path |
| `PORT` | No | `8501` | Server port (auto-set by Heroku) |
| `ENVIRONMENT` | No | `development` | Set to `production` for stricter validation |

---

## 📊 Production Considerations

### Performance
- Enable caching (already configured, 5-minute TTL)
- Use persistent database storage
- Consider CDN for static assets

### Security
- Always use HTTPS in production
- Set `ENVIRONMENT=production` to disable localhost webhooks
- Keep API keys in secrets, never commit to git
- Use `.gitignore` to exclude sensitive files

### Monitoring
- Docker healthcheck is configured (checks `/_stcore/health`)
- Monitor webhook delivery success rates
- Track API rate limit usage

### Scaling
- Streamlit Cloud: Auto-scaling included
- Heroku: Use `heroku ps:scale web=2` for multiple dynos
- Docker: Use orchestration (Kubernetes, Docker Swarm)

---

## 🧪 Testing Deployment

After deployment, verify:

1. ✅ App loads without errors
2. ✅ NASA API data fetches successfully
3. ✅ Filters and visualizations work
4. ✅ Watchlist persists across sessions
5. ✅ Webhook alerts can be configured
6. ✅ CSV export functions correctly

---

## 🐛 Troubleshooting

### "API error: 429"
- Rate limit exceeded. Get a NASA API key or wait for rate limit reset.

### Database errors
- Check `ASTEROID_DB_PATH` is writable
- Ensure SQLite is installed (included in Python)

### Port already in use
- Change port: `streamlit run app.py --server.port=8502`

### Docker healthcheck failing
- Ensure app starts within 5 seconds
- Check logs: `docker logs <container-id>`
