# Webhook API Documentation

The Asteroid Tracker supports webhook notifications when hazardous asteroids are detected.

---

## Configuration

Enable webhooks in the sidebar:
1. Check "Enable webhook alerts"
2. Enter your webhook URL
3. Set hazard distance threshold

---

## Webhook Payload

When hazardous asteroids are detected, a POST request is sent to your webhook URL.

### Request Format

**Method:** `POST`  
**Content-Type:** `application/json`  
**Timeout:** 5 seconds

### Payload Schema

```json
{
  "hazards": [
    {
      "date": "2024-01-15",
      "name": "(2024 AB)",
      "diameter_m": 250.5,
      "distance_km": 850000.0,
      "velocity_kph": 75000.0,
      "risk_score": 0.85
    }
  ],
  "preset": "Next 7 days",
  "range": ["2024-01-15", "2024-01-22"]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `hazards` | Array | List of hazardous asteroids detected |
| `hazards[].date` | String | Close approach date (YYYY-MM-DD) |
| `hazards[].name` | String | Asteroid designation |
| `hazards[].diameter_m` | Number | Estimated diameter in meters |
| `hazards[].distance_km` | Number | Miss distance in kilometers |
| `hazards[].velocity_kph` | Number | Relative velocity in km/h |
| `hazards[].risk_score` | Number | Calculated risk score (0-1) |
| `preset` | String | Date range preset used |
| `range` | Array | Start and end dates [start, end] |

---

## Response Handling

Your webhook endpoint should:
- Return HTTP 2xx status code for success
- Respond within 5 seconds
- Handle duplicate alerts (same asteroid may appear in multiple queries)

### Example Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "received",
  "alert_id": "abc123"
}
```

---

## Security

### URL Validation

The application validates webhook URLs:
- ✅ Must use `http://` or `https://` protocol
- ✅ Localhost allowed in development mode only
- ❌ Invalid protocols rejected
- ❌ Empty URLs rejected

### Production Mode

Set `ENVIRONMENT=production` to:
- Disable localhost webhooks
- Enforce stricter validation

### Best Practices

1. **Use HTTPS** for webhook endpoints
2. **Validate payload** signature if implementing authentication
3. **Rate limit** your webhook endpoint
4. **Log failures** for debugging
5. **Implement retry logic** on your end if needed

---

## Example Implementations

### Python (Flask)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def asteroid_webhook():
    data = request.json
    hazards = data.get('hazards', [])
    
    # Process hazards
    for asteroid in hazards:
        print(f"Alert: {asteroid['name']} - Risk: {asteroid['risk_score']}")
        # Send email, SMS, etc.
    
    return jsonify({"status": "received"}), 200

if __name__ == '__main__':
    app.run(port=5000)
```

### Node.js (Express)

```javascript
const express = require('express');
const app = express();

app.use(express.json());

app.post('/webhook', (req, res) => {
  const { hazards, preset, range } = req.body;
  
  hazards.forEach(asteroid => {
    console.log(`Alert: ${asteroid.name} - Risk: ${asteroid.risk_score}`);
    // Send notification
  });
  
  res.json({ status: 'received' });
});

app.listen(5000, () => console.log('Webhook server running'));
```

---

## Testing Webhooks

### Using webhook.site

1. Go to [webhook.site](https://webhook.site)
2. Copy the unique URL
3. Paste into Asteroid Tracker webhook URL field
4. Trigger an alert by setting low hazard threshold
5. View received payload on webhook.site

### Using ngrok for Local Testing

```bash
# Start your local webhook server
python webhook_server.py

# In another terminal, expose it
ngrok http 5000

# Use the ngrok URL in Asteroid Tracker
https://abc123.ngrok.io/webhook
```

---

## Alert History

All webhook alerts are logged in the SQLite database (`app_data.db`).

### Query Alert History

```python
import sqlite3

conn = sqlite3.connect('app_data.db')
cursor = conn.cursor()
cursor.execute("SELECT ts, payload FROM alerts ORDER BY ts DESC LIMIT 10")

for timestamp, payload in cursor.fetchall():
    print(f"{timestamp}: {payload}")

conn.close()
```

---

## Troubleshooting

### "Invalid webhook URL" error
- Ensure URL starts with `http://` or `https://`
- Check for typos in the URL
- Localhost only works in development mode

### "Failed to post webhook alert" error
- Verify webhook endpoint is accessible
- Check endpoint returns 2xx status code
- Ensure endpoint responds within 5 seconds
- Check firewall/network settings

### No alerts received
- Verify hazard threshold is set appropriately
- Check that asteroids in date range meet threshold
- Confirm webhook URL is correct
- Test endpoint with curl or Postman

### Duplicate alerts
- Normal behavior if querying overlapping date ranges
- Implement deduplication on your end using asteroid name + date
