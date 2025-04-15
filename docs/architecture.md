# System Architecture

## Overview
Production-grade ML system for stock prediction with end-to-end pipeline from data collection to monitoring.

## Components

### 1. Data Pipeline
- **Collectors**: Async data collection from NSE, BSE, FII sources
- **Feature Store**: Versioned feature storage with SQLite
- **Validation**: Data quality checks at ingestion

### 2. ML Pipeline
- **Training**: MLflow-tracked experiments with cross-validation
- **Backtesting**: Walk-forward validation to avoid look-ahead bias
- **Model Registry**: Versioned models with promotion workflow

### 3. API Layer
- **FastAPI**: High-performance async API
- **Rate Limiting**: Redis-backed rate limiting
- **Authentication**: JWT-based auth (optional)

### 4. Monitoring
- **Performance Tracking**: Real-time accuracy, Sharpe ratio
- **Drift Detection**: KS-test based drift detection
- **Alerting**: Slack/email alerts for degradation

### 5. Frontend
- **React Dashboard**: Real-time predictions and performance
- **Visualizations**: Recharts-based interactive charts

## Data Flow