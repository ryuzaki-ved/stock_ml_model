"""
Detect data and concept drift
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from loguru import logger
import json
import os

class DriftDetector:
    """Detect distribution drift in features"""
    
    def __init__(self, reference_data_path: str = "data/reference_distribution.parquet"):
        self.reference_data_path = reference_data_path
        try:
            self.reference_data = pd.read_parquet(reference_data_path)
            logger.info("Loaded reference data for drift detection")
        except:
            self.reference_data = None
            logger.warning("No reference data found")
    
    async def check_drift(self, current_data: pd.DataFrame, threshold: float = 0.05):
        """Check for drift using KS test"""
        
        if self.reference_data is None:
            logger.warning("Cannot check drift without reference data")
            return
        
        drift_results = {}
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.reference_data.columns:
                # KS test
                statistic, pvalue = ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_detected = pvalue < threshold
                
                drift_results[col] = {
                    'statistic': float(statistic),
                    'pvalue': float(pvalue),
                    'drift_detected': drift_detected
                }
                
                if drift_detected:
                    logger.warning(f"Drift detected in {col}: p-value={pvalue:.4f}")
        
        # Save results
        os.makedirs('logs', exist_ok=True)
        with open('logs/drift_results.json', 'w') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'results': drift_results
            }, f, indent=2)
        
        # Alert if significant drift
        drifted_features = [k for k, v in drift_results.items() if v['drift_detected']]
        if len(drifted_features) > 5:
            await self._send_alert(f"Significant drift detected in {len(drifted_features)} features")
    
    async def _send_alert(self, message: str):
        """Send alert via Slack/email"""
        logger.critical(f"ALERT: {message}")
        # Implementation for Slack webhook or email