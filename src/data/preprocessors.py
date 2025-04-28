import pandas as pd


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure correct dtypes
    df['date'] = pd.to_datetime(df['date'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'delivery_qty', 'delivery_percentage']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Drop impossible values
    df = df[df['close'] > 0]
    df = df.dropna(subset=['symbol', 'date'])
    return df

