import sqlite3
from pathlib import Path
from typing import Optional

def _get_db_path(root: Optional[Path] = None) -> Path:
    if root is None:
        from prometheus.api.app import forecasts_root
        root = Path(forecasts_root())
    return root / "prometheus.db"

def get_connection(root: Optional[Path] = None) -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    db_path = _get_db_path(root)
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db(root: Optional[Path] = None) -> None:
    """Initialize the database schema."""
    conn = get_connection(root)
    
    # Forecasts catalog
    conn.execute("""
    CREATE TABLE IF NOT EXISTS forecasts (
        forecast_date TEXT PRIMARY KEY,
        bundle_hash TEXT,
        features_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # District timeseries statistics
    conn.execute("""
    CREATE TABLE IF NOT EXISTS district_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        district_id TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        horizon INTEGER NOT NULL,
        mean_prob REAL,
        max_prob REAL,
        UNIQUE(district_id, forecast_date, horizon)
    );
    """)
    
    # Verification metrics
    conn.execute("""
    CREATE TABLE IF NOT EXISTS verification_metrics (
        forecast_date TEXT PRIMARY KEY,
        observe_date TEXT NOT NULL,
        n INTEGER,
        n_pos INTEGER,
        base_rate REAL,
        mean_forecast REAL,
        pr_auc REAL,
        brier REAL,
        top10_capture REAL,
        fss REAL,
        rev REAL,
        valid BOOLEAN
    );
    """)
    
    # API query logs
    conn.execute("""
    CREATE TABLE IF NOT EXISTS query_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        endpoint TEXT,
        query_params TEXT,
        client_ip TEXT
    );
    """)
    
    # Create indices
    conn.execute("CREATE INDEX IF NOT EXISTS idx_district_date ON district_stats(district_id, forecast_date);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_district_horizon ON district_stats(district_id, horizon);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_verify_date ON verification_metrics(forecast_date);")
    
    conn.commit()
    conn.close()
