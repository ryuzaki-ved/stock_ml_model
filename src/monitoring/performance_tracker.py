"""
Track model performance over time
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
import sqlite3

class PerformanceTracker:
    """Track predictions and actual outcomes"""
    
    def __init__(self, db_path: str = "predictions.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                prediction INTEGER,
                confidence FLOAT,
                signal TEXT,
                predicted_at TIMESTAMP,
                actual_outcome INTEGER DEFAULT NULL,
                actual_return FLOAT DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT NULL
            )
        """)
        self.conn.commit()
    
    def log_predictions(self, predictions: List):
        """Log predictions to database"""
        records = [(
            p.symbol,
            p.prediction,
            p.confidence,
            p.signal,
            p.timestamp
        ) for p in predictions]
        
        self.conn.executemany("""
            INSERT INTO predictions (symbol, prediction, confidence, signal, predicted_at)
            VALUES (?, ?, ?, ?, ?)
        """, records)
        self.conn.commit()
        
        logger.info(f"Logged {len(predictions)} predictions")
    
    def update_actual_outcomes(self):
        """Update predictions with actual outcomes"""
        # Get predictions from 1-2 days ago (to check outcome)
        query = """
            SELECT id, symbol, predicted_at, prediction
            FROM predictions
            WHERE actual_outcome IS NULL
            AND predicted_at < datetime('now', '-1 day')
            AND predicted_at > datetime('now', '-30 days')
        """
        
        df = pd.read_sql(query, self.conn)
        
        if len(df) == 0:
            return
        
        # Fetch actual price movements
        from ..data.collectors import NSECollector
        collector = NSECollector()
        
        for _, row in df.iterrows():
            # Get actual price movement
            # ... implementation ...
            pass
    
    def get_metrics(self, days: int = 30) -> Dict:
        """Calculate performance metrics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = """
            SELECT *
            FROM predictions
            WHERE predicted_at > ?
            AND actual_outcome IS NOT NULL
        """
        
        df = pd.read_sql(query, self.conn, params=(cutoff_date,))
        
        if len(df) == 0:
            return {
                'accuracy': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'total_predictions': 0
            }
        
        accuracy = (df['prediction'] == df['actual_outcome']).mean()
        win_rate = (df['actual_return'] > 0).mean()
        sharpe = df['actual_return'].mean() / df['actual_return'].std() * np.sqrt(252) if df['actual_return'].std() > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'total_predictions': len(df),
            'period_start': df['predicted_at'].min(),
            'period_end': df['predicted_at'].max()
        }