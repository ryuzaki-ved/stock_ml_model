import os
import sqlite3
import pandas as pd
import numpy as np


def ensure_dirs():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/predictions', exist_ok=True)
    os.makedirs('data/external', exist_ok=True)


def generate_prices_rows(num_days: int = 260, symbols: list | None = None, start_price: float = 100.0):
    if symbols is None:
        # ~50k rows: 260 days * 200 symbols = 52,000 rows
        symbols = [f"SYM{i:03d}" for i in range(200)]
    dates = pd.date_range('2023-01-01', periods=num_days, freq='B')  # business days
    rows = []
    rng = np.random.default_rng(42)
    for s in symbols:
        price = start_price * (1 + rng.normal(0, 0.02))
        for d in dates:
            ret = rng.normal(0.0005, 0.02)
            open_price = price * (1 + rng.normal(0, 0.005))
            high_price = open_price * (1 + abs(rng.normal(0, 0.01)))
            low_price = open_price * (1 - abs(rng.normal(0, 0.01)))
            close_price = max(low_price, min(high_price, open_price * (1 + ret)))
            volume = int(abs(rng.normal(1_000_000, 300_000)))
            delivery_pct = float(max(10.0, min(90.0, rng.normal(55.0, 15.0))))
            delivery_qty = int(volume * delivery_pct / 100.0)
            rows.append({
                'symbol': s,
                'date': d,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'delivery_qty': delivery_qty,
                'delivery_percentage': delivery_pct,
            })
            price = close_price
    return rows


def create_large_prices_csv(out_path='data/raw/prices.csv', min_rows: int = 50000):
    rows = generate_prices_rows()
    df = pd.DataFrame(rows)
    if len(df) < min_rows:
        # If somehow below, extend symbols
        extra = generate_prices_rows(symbols=[f"XSYM{i:03d}" for i in range(200, 260)])
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    df.to_csv(out_path, index=False)
    return df


def create_features_db(db_path='features.db'):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date DATE,
            feature_version TEXT,
            features JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol, date)")
    conn.commit()
    conn.close()


def create_predictions_db(db_path='predictions.db'):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
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
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, predicted_at)")
    conn.commit()
    conn.close()


def write_parquet(df: pd.DataFrame, out_path='data/processed/prices.parquet'):
    try:
        df.to_parquet(out_path, index=False)
    except Exception as e:
        # pandas may need pyarrow/fastparquet; user can install later
        print(f"Skipping parquet write (install pyarrow/fastparquet): {e}")


def write_external_data(dates: pd.Series, out_path='data/external/fii_dii.csv'):
    rng = np.random.default_rng(123)
    fii = rng.normal(0, 500.0, size=len(dates)).round(2)
    dii = rng.normal(0, 400.0, size=len(dates)).round(2)
    ext = pd.DataFrame({
        'date': dates,
        'fii_net_buy_sell_cr': fii,
        'dii_net_buy_sell_cr': dii
    })
    ext.to_csv(out_path, index=False)


def write_predictions_sample(df: pd.DataFrame, out_path='data/predictions/predictions_sample.csv'):
    # Make a tiny sample predictions dump aligned with schema
    subset = df[['symbol', 'date', 'close']].head(100).copy()
    subset.rename(columns={'date': 'predicted_at'}, inplace=True)
    subset['prediction'] = np.random.choice([-1, 0, 1], size=len(subset))
    subset['confidence'] = np.random.uniform(0.4, 0.9, size=len(subset)).round(3)
    subset['signal'] = subset['prediction'].map({1: 'BUY', 0: 'HOLD', -1: 'SELL'})
    subset['actual_outcome'] = np.random.choice([-1, 0, 1], size=len(subset))
    subset['actual_return'] = np.random.normal(0.001, 0.02, size=len(subset)).round(4)
    subset.to_csv(out_path, index=False)


def main():
    ensure_dirs()
    # Create a large synthetic prices dataset (>50k rows)
    df = create_large_prices_csv()
    create_features_db()
    create_predictions_db()
    write_parquet(df)
    write_external_data(df['date'].drop_duplicates().sort_values())
    write_predictions_sample(df)
    print("Data setup complete.")


if __name__ == '__main__':
    main()

