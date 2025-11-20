"""
Test suite for Asteroid Tracker application.

Run with: pytest tests/
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRiskCalculation:
    """Test risk scoring functionality."""
    
    def test_risk_score_calculation(self):
        """Test that risk scores are calculated correctly."""
        # Sample data
        data = {
            'diameter_m': [100, 500, 1000],
            'distance_km': [1000000, 500000, 100000],
            'velocity_kph': [50000, 75000, 100000]
        }
        df = pd.DataFrame(data)
        
        # Calculate percentiles
        df['p_diameter'] = df['diameter_m'].rank(pct=True)
        df['p_velocity'] = df['velocity_kph'].rank(pct=True)
        df['p_distance_inv'] = (1 - df['distance_km'].rank(pct=True))
        
        # Default weights
        w_d, w_di, w_v = 0.5, 0.3, 0.2
        df['risk_score'] = (
            w_d * df['p_diameter'] + w_di * df['p_distance_inv'] + w_v * df['p_velocity']
        )
        
        # Assertions
        assert len(df) == 3
        assert all(df['risk_score'] >= 0)
        assert all(df['risk_score'] <= 1)
        # Largest, closest, fastest should have highest risk
        assert df.iloc[2]['risk_score'] > df.iloc[0]['risk_score']
    
    def test_weight_normalization(self):
        """Test that weights are normalized correctly."""
        w_d, w_di, w_v = 0.5, 0.3, 0.2
        w_sum = w_d + w_di + w_v
        w_d_norm = w_d / w_sum
        w_di_norm = w_di / w_sum
        w_v_norm = w_v / w_sum
        
        assert abs((w_d_norm + w_di_norm + w_v_norm) - 1.0) < 1e-9


class TestDataFiltering:
    """Test data filtering functionality."""
    
    def test_diameter_filter(self):
        """Test filtering by minimum diameter."""
        data = {
            'diameter_m': [50, 150, 250],
            'distance_km': [1000000, 1000000, 1000000],
            'velocity_kph': [50000, 50000, 50000]
        }
        df = pd.DataFrame(data)
        
        threshold = 100
        filtered = df[df['diameter_m'] >= threshold]
        
        assert len(filtered) == 2
        assert all(filtered['diameter_m'] >= threshold)
    
    def test_distance_range_filter(self):
        """Test filtering by distance range."""
        data = {
            'diameter_m': [100, 100, 100],
            'distance_km': [500000, 1000000, 2000000],
            'velocity_kph': [50000, 50000, 50000]
        }
        df = pd.DataFrame(data)
        
        min_dist = 750000
        max_dist = 1500000
        filtered = df[(df['distance_km'] >= min_dist) & (df['distance_km'] <= max_dist)]
        
        assert len(filtered) == 1
        assert filtered.iloc[0]['distance_km'] == 1000000
    
    def test_name_search(self):
        """Test name substring search."""
        data = {
            'name': ['(2024 AB)', '(2024 CD)', '(2025 EF)'],
            'diameter_m': [100, 100, 100],
            'distance_km': [1000000, 1000000, 1000000],
            'velocity_kph': [50000, 50000, 50000]
        }
        df = pd.DataFrame(data)
        
        query = '2024'
        filtered = df[df['name'].str.contains(query, case=False, na=False)]
        
        assert len(filtered) == 2


class TestWebhookValidation:
    """Test webhook URL validation."""
    
    def test_valid_https_url(self):
        """Test that valid HTTPS URLs pass validation."""
        from app import validate_webhook_url
        assert validate_webhook_url("https://example.com/webhook") == True
    
    def test_valid_http_url(self):
        """Test that valid HTTP URLs pass validation."""
        from app import validate_webhook_url
        assert validate_webhook_url("http://example.com/webhook") == True
    
    def test_invalid_url_no_protocol(self):
        """Test that URLs without protocol fail validation."""
        from app import validate_webhook_url
        assert validate_webhook_url("example.com/webhook") == False
    
    def test_empty_url(self):
        """Test that empty URLs fail validation."""
        from app import validate_webhook_url
        assert validate_webhook_url("") == False
        assert validate_webhook_url(None) == False
    
    def test_localhost_in_development(self):
        """Test that localhost is allowed in development."""
        import os
        from app import validate_webhook_url
        
        # Set development environment
        os.environ['ENVIRONMENT'] = 'development'
        assert validate_webhook_url("http://localhost:8000/webhook") == True
        
        # Clean up
        if 'ENVIRONMENT' in os.environ:
            del os.environ['ENVIRONMENT']


class TestDatabaseOperations:
    """Test SQLite database operations."""
    
    def test_watchlist_persistence(self):
        """Test that watchlist can be saved and retrieved."""
        import sqlite3
        import tempfile
        from pathlib import Path
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        try:
            # Initialize database
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS watchlist (name TEXT PRIMARY KEY)")
            conn.commit()
            
            # Add items
            items = ['(2024 AB)', '(2024 CD)']
            cur.executemany("INSERT INTO watchlist(name) VALUES(?)", [(n,) for n in items])
            conn.commit()
            
            # Retrieve items
            cur.execute("SELECT name FROM watchlist ORDER BY name ASC")
            retrieved = [r[0] for r in cur.fetchall()]
            
            conn.close()
            
            assert retrieved == sorted(items)
        finally:
            # Cleanup
            Path(db_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
