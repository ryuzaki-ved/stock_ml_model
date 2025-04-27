"""
Track model performance over time
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
import sqlite3
import numpy as np

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
        
        # Fetch actual price movements (simplified: next-day close vs current close)
        try:
            from ..data.collectors import NSECollector
            import asyncio
            collector = NSECollector()
            
            updates = []
            for _, row in df.iterrows():
                symbol = row['symbol']
                start_date = pd.to_datetime(row['predicted_at']).strftime('%Y-%m-%d')
                end_date = (pd.to_datetime(row['predicted_at']) + timedelta(days=2)).strftime('%Y-%m-%d')
                
                async def fetch_symbol():
                    return await collector.fetch([symbol], start_date, end_date)
                
                try:
                    prices = asyncio.run(fetch_symbol())
                except RuntimeError:
                    # Already running event loop (e.g., in async context) — fallback to direct loop
                    loop = asyncio.get_event_loop()
                    prices = loop.run_until_complete(fetch_symbol())
                
                if len(prices) < 2:
                    continue
                prices = prices.sort_values('date')
                current_close = prices.iloc[0]['close']
                next_close = prices.iloc[-1]['close']
                actual_return = (next_close / current_close) - 1.0
                if actual_return > 0:
                    actual_outcome = 1
                elif actual_return < 0:
                    actual_outcome = -1
                else:
                    actual_outcome = 0
                updates.append((actual_outcome, float(actual_return), datetime.now(), int(row['id'])))
            
            if updates:
                self.conn.executemany(
                    """
                    UPDATE predictions
                    SET actual_outcome = ?, actual_return = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    updates
                )
                self.conn.commit()
                logger.info(f"Updated actual outcomes for {len(updates)} predictions")
        except Exception as e:
            logger.error(f"Failed to update actual outcomes: {e}")
    
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

    def get_history(self, symbol: str = None, days: int = 30):
        """Return historical predictions (optionally filtered by symbol)"""
        cutoff_date = datetime.now() - timedelta(days=days)
        params = [cutoff_date]
        base_query = """
            SELECT symbol, prediction, confidence, signal, predicted_at, actual_outcome, actual_return
            FROM predictions
            WHERE predicted_at > ?
        """
        if symbol:
            base_query += " AND symbol = ?"
            params.append(symbol)
        base_query += " ORDER BY predicted_at DESC"
        df = pd.read_sql(base_query, self.conn, params=tuple(params))
        records = df.to_dict(orient='records') if len(df) else []
        return records