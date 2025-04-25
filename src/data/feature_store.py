"""
Feature engineering and feature store
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import talib
from loguru import logger

class FeatureEngineer:
    """Create features from raw data"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features"""
        logger.info("Creating features...")
        
        df = df.sort_values(['symbol', 'date'])
        
        # Price-based features
        df = self._add_technical_indicators(df)
        
        # Volume-based features
        df = self._add_volume_features(df)
        
        # Delivery-based features
        df = self._add_delivery_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Lag features
        df = self._add_lag_features(df)
        
        logger.info(f"Created {len(df.columns)} features")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib"""
        for symbol, group in df.groupby('symbol'):
            # RSI
            df.loc[group.index, 'rsi_14'] = talib.RSI(group['close'].values, timeperiod=14)
            
            # MACD
            macd, signal, hist = talib.MACD(group['close'].values)
            df.loc[group.index, 'macd'] = macd
            df.loc[group.index, 'macd_signal'] = signal
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(group['close'].values)
            df.loc[group.index, 'bb_upper'] = upper
            df.loc[group.index, 'bb_lower'] = lower
            df.loc[group.index, 'bb_position'] = (group['close'] - lower) / (upper - lower)
            
            # ATR (volatility) – guard if high/low missing
            if 'high' in group.columns and 'low' in group.columns:
                df.loc[group.index, 'atr_14'] = talib.ATR(
                    group['high'].values, 
                    group['low'].values, 
                    group['close'].values, 
                    timeperiod=14
                )
            else:
                df.loc[group.index, 'atr_14'] = np.nan
            
            # Moving averages
            df.loc[group.index, 'sma_20'] = talib.SMA(group['close'].values, timeperiod=20)
            df.loc[group.index, 'ema_12'] = talib.EMA(group['close'].values, timeperiod=12)
            
            # Price momentum
            df.loc[group.index, 'momentum_10'] = talib.MOM(group['close'].values, timeperiod=10)
            
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""
        for symbol, group in df.groupby('symbol'):
            # Volume ratios
            df.loc[group.index, 'volume_sma_20'] = group['volume'].rolling(20).mean()
            df.loc[group.index, 'volume_ratio'] = group['volume'] / group['volume'].rolling(20).mean()
            
            # OBV (On Balance Volume)
            df.loc[group.index, 'obv'] = talib.OBV(group['close'].values, group['volume'].values)
            
            # Volume spike detection
            vol_mean = group['volume'].rolling(20).mean()
            vol_std = group['volume'].rolling(20).std()
            df.loc[group.index, 'volume_spike'] = (group['volume'] - vol_mean) / vol_std
            
        return df
    
    def _add_delivery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Delivery percentage based features"""
        for symbol, group in df.groupby('symbol'):
            # Moving average of delivery %
            df.loc[group.index, 'delivery_ma_5'] = group['delivery_percentage'].rolling(5).mean()
            df.loc[group.index, 'delivery_ma_20'] = group['delivery_percentage'].rolling(20).mean()
            
            # Delivery trend
            df.loc[group.index, 'delivery_trend'] = (
                group['delivery_percentage'] - group['delivery_percentage'].rolling(20).mean()
            )
            
            # High delivery flag (>70%)
            df.loc[group.index, 'high_delivery'] = (group['delivery_percentage'] > 70).astype(int)
            
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Is it near month/quarter end?
        df['days_to_month_end'] = df['date'].dt.days_in_month - df['date'].dt.day
        df['near_month_end'] = (df['days_to_month_end'] <= 5).astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features (past values)"""
        lag_cols = ['close', 'volume', 'delivery_percentage', 'rsi_14']
        lags = [1, 2, 3, 5]
        
        for col in lag_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('symbol')[col].shift(lag)
        
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.02) -> pd.DataFrame:
        """Create target variable"""
        # Future return
        df['future_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
        
        # Classification target: 1 if return > threshold, -1 if < -threshold, 0 otherwise
        df['target'] = 0
        df.loc[df['future_return'] > threshold, 'target'] = 1
        df.loc[df['future_return'] < -threshold, 'target'] = -1
        
        # Regression target
        df['target_return'] = df['future_return']
        
        return df

class FeatureStore:
    """Store and version features"""
    
    def __init__(self, db_path: str = "features.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self):
        """Create feature store tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date DATE,
                feature_version TEXT,
                features JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def save_features(self, df: pd.DataFrame, version: str):
        """Save features to store"""
        df['feature_version'] = version
        df.to_sql('features', self.conn, if_exists='append', index=False)
        logger.info(f"Saved {len(df)} feature records (version: {version})")
    
    def load_features(self, symbols: List[str], start_date: str, end_date: str, version: str = None):
        """Load features from store"""
        query = """
            SELECT * FROM features 
            WHERE symbol IN ({})
            AND date BETWEEN ? AND ?
        """.format(','.join(['?'] * len(symbols)))
        
        params = symbols + [start_date, end_date]
        
        if version:
            query += " AND feature_version = ?"
            params.append(version)
        
        df = pd.read_sql(query, self.conn, params=params)
        logger.info(f"Loaded {len(df)} feature records")
        return df