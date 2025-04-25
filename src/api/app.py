"""
FastAPI application for serving predictions
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import mlflow
from loguru import logger

from .schemas import PredictionRequest, PredictionResponse, PerformanceMetrics
from ..monitoring.performance_tracker import PerformanceTracker
from ..monitoring.drift_detector import DriftDetector

app = FastAPI(
    title="Stock ML Platform API",
    description="Production ML system for stock predictions",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load ML model from MLflow"""
    global model, feature_cols, performance_tracker, drift_detector
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/stock_predictor/production"
    
    try:
        # Prefer LightGBM flavor when available
        try:
            import mlflow.lightgbm
            model = mlflow.lightgbm.load_model(model_uri)
        except Exception:
            # Fallback to sklearn flavor
            model = mlflow.sklearn.load_model(model_uri)
        # Load feature columns from LightGBM Booster if available
        try:
            feature_cols = list(model.feature_name())
        except Exception:
            # Fallback: try attribute used by some wrappers
            feature_cols = list(getattr(model, 'feature_name_', []))
        
        performance_tracker = PerformanceTracker()
        drift_detector = DriftDetector()
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=List[PredictionResponse])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Make predictions for given symbols
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Collect latest data for symbols
        from ..data.collectors import DataPipeline
        pipeline = DataPipeline()
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")
        
        data = await pipeline.collect_all(request.symbols, start_date, end_date)
        
        # Feature engineering
        from ..data.feature_store import FeatureEngineer
        engineer = FeatureEngineer({})
        df = engineer.create_features(data['prices'])
        
        # Get latest data for each symbol
        latest_df = df.groupby('symbol').tail(1)
        
        # Make predictions
        X = latest_df[feature_cols]
        predictions = model.predict(X)
        # LightGBM multiclass returns probs per class in order [0,1,2]
        pred_classes = np.argmax(predictions, axis=1) - 1
        pred_probs = np.max(predictions, axis=1)
        
        # Format response
        responses = []
        for idx, row in latest_df.iterrows():
            responses.append(PredictionResponse(
                symbol=row['symbol'],
                prediction=int(pred_classes[idx]),
                confidence=float(pred_probs[idx]),
                signal="BUY" if pred_classes[idx] == 1 else ("SELL" if pred_classes[idx] == -1 else "HOLD"),
                timestamp=datetime.now()
            ))
        
        # Background tasks
        background_tasks.add_task(drift_detector.check_drift, latest_df)
        background_tasks.add_task(performance_tracker.log_predictions, responses)
        
        logger.info(f"Generated predictions for {len(responses)} symbols")
        
        return responses
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance", response_model=PerformanceMetrics)
async def get_performance(days: int = 30):
    """Get model performance metrics"""
    metrics = performance_tracker.get_metrics(days)
    return metrics

@app.get("/predictions/history")
async def get_prediction_history(symbol: str = None, days: int = 30):
    """Get historical predictions"""
    history = performance_tracker.get_history(symbol, days)
    return history

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    background_tasks.add_task(retrain_model)
    return {"message": "Retraining triggered"}

async def retrain_model():
    """Retrain model with latest data"""
    logger.info("Starting model retraining...")
    # Implementation for retraining
    pass