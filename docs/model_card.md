# Model Card: Stock Prediction Model

## Model Details
- **Model Type**: LightGBM Classifier
- **Version**: 1.0.0
- **Training Date**: 2024-01-15
- **Framework**: LightGBM 4.0.0

## Intended Use
- **Primary Use**: Generate daily buy/sell/hold signals for Indian stocks
- **Out-of-Scope**: Not for automated trading without human oversight

## Training Data
- **Sources**: NSE historical data, FII/DII reports
- **Time Period**: 2019-2024 (5 years)
- **Symbols**: Nifty 500 constituents
- **Samples**: ~500,000 daily records

## Features (Top 10)
1. RSI (14-day)
2. Delivery percentage trend
3. Volume ratio
4. MACD histogram
5. Bollinger Band position
6. FII net buying
7. ATR (volatility)
8. Price momentum
9. Day of week
10. Near month-end flag

## Performance Metrics

### Cross-Validation (5-fold time series split)
- **Accuracy**: 62.3% ± 2.1%
- **F1 Score**: 0.59 ± 0.03
- **Sharpe Ratio**: 1.45

### Walk-Forward Backtest (2023-2024)
- **Total Return**: 23.4%
- **Sharpe Ratio**: 1.52
- **Max Drawdown**: -12.3%
- **Win Rate**: 58.2%

### Benchmark Comparison
- **Nifty 50 Return (same period)**: 18.7%
- **Alpha**: +4.7%

## Limitations
1. **Market Regimes**: Performance degrades during extreme volatility
2. **Low Liquidity**: Not tested on illiquid stocks
3. **Corporate Actions**: Doesn't account for splits, bonuses
4. **Macro Events**: Limited exposure to black swan events

## Ethical Considerations
- Model outputs should not be sole basis for investment decisions
- Past performance does not guarantee future results
- Users should understand financial risks

## Monitoring
- **Drift Detection**: Weekly KS-test on feature distributions
- **Performance Tracking**: Daily accuracy, monthly Sharpe ratio
- **Retraining**: Triggered when accuracy drops below 55% for 2 weeks