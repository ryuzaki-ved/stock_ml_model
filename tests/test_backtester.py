import pandas as pd
import numpy as np
from src.training.backtester import Backtester


class DummyModel:
    def predict(self, X):
        # Produce simple class probabilities favoring up moves when momentum positive
        momentum = X.get('momentum_10', pd.Series(np.zeros(len(X))))
        probs = np.zeros((len(X), 3))
        probs[:, 1] = 0.2  # class 0
        probs[:, 2] = (momentum > 0).astype(float) * 0.7 + 0.1  # class 1 (BUY)
        probs[:, 0] = 1 - probs[:, 1] - probs[:, 2]  # class -1
        return probs


def make_synthetic_df(n_days=60, n_symbols=3):
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    symbols = [f'SYM{i}' for i in range(n_symbols)]
    rows = []
    for s in symbols:
        price = 100.0
        for d in dates:
            ret = np.random.normal(0.0005, 0.01)
            price *= (1 + ret)
            rows.append({
                'symbol': s,
                'date': d,
                'close': price,
                'momentum_10': ret * 100
            })
    df = pd.DataFrame(rows)
    return df


def test_backtester_runs_and_returns_metrics():
    df = make_synthetic_df()
    feature_cols = ['momentum_10']
    model = DummyModel()
    cfg = {
        'initial_capital': 100000,
        'transaction_cost': 0.0005,
        'train_window_days': 20,
        'test_window_days': 10,
        'retrain_frequency_days': 10,
        'confidence_threshold': 0.5
    }
    bt = Backtester(cfg)
    summary = bt.run(df, model, feature_cols)
    assert 'total_return' in summary
    assert 'sharpe_ratio' in summary
    assert 'max_drawdown' in summary
    assert 'final_portfolio_value' in summary
    # sanity checks
    assert isinstance(summary['total_return'], float)
    assert isinstance(summary['final_portfolio_value'], (float, int))

