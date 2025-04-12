"""
Data collectors for various sources
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import aiohttp
from loguru import logger

class BaseCollector(ABC):
    """Abstract base class for data collectors"""
    
    @abstractmethod
    async def fetch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass

class NSECollector(BaseCollector):
    """Collect data from NSE"""
    
    def __init__(self, api_key: str = None):
        self.base_url = "https://www.nseindia.com/api"
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        
    async def fetch(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV + delivery data"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_symbol(session, symbol, start_date, end_date) 
                    for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        df = pd.concat([r for r in results if isinstance(r, pd.DataFrame)])
        logger.info(f"Fetched {len(df)} records for {len(symbols)} symbols")
        return df
    
    async def _fetch_symbol(self, session, symbol, start_date, end_date):
        """Fetch data for single symbol"""
        url = f"{self.base_url}/historical/cm/equity"
        params = {'symbol': symbol, 'from': start_date, 'to': end_date}
        
        try:
            async with session.get(url, headers=self.headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_response(data, symbol)
                else:
                    logger.warning(f"Failed to fetch {symbol}: {resp.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def _parse_response(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse NSE API response"""
        df = pd.DataFrame(data.get('data', []))
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['CH_TIMESTAMP'])
        df.rename(columns={
            'CH_OPENING_PRICE': 'open',
            'CH_CLOSING_PRICE': 'close',
            'CH_TOT_TRADED_QTY': 'volume',
            'DELIV_QTY': 'delivery_qty',
            'DELIV_PER': 'delivery_percentage'
        }, inplace=True)
        return df[['symbol', 'date', 'open', 'close', 'volume', 
                   'delivery_qty', 'delivery_percentage']]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate collected data"""
        required_cols = ['symbol', 'date', 'open', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            logger.error("Missing required columns")
            return False
        
        if df.isnull().sum().sum() > len(df) * 0.1:  # >10% missing
            logger.warning("Too many missing values")
            return False
        
        if (df['close'] <= 0).any():
            logger.error("Invalid price values found")
            return False
        
        return True

class FIICollector(BaseCollector):
    """Collect FII/DII data"""
    
    async def fetch(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch FII/DII data from NSE"""
        # Implementation for FII data collection
        pass
    
    def validate(self, df: pd.DataFrame) -> bool:
        # Validation logic
        pass

class DataPipeline:
    """Orchestrate data collection from multiple sources"""
    
    def __init__(self):
        self.nse_collector = NSECollector()
        self.fii_collector = FIICollector()
        
    async def collect_all(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Collect from all sources concurrently"""
        logger.info(f"Starting data collection for {len(symbols)} symbols")
        
        price_data, fii_data = await asyncio.gather(
            self.nse_collector.fetch(symbols, start_date, end_date),
            self.fii_collector.fetch(start_date, end_date)
        )
        
        return {
            'prices': price_data,
            'fii': fii_data
        }