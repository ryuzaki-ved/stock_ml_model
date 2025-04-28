import pandas as pd


def validate_price_schema(df: pd.DataFrame) -> bool:
    required = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}
    return required.issubset(set(df.columns))


def check_missing_ratio(df: pd.DataFrame, threshold: float = 0.1) -> bool:
    return df.isnull().sum().sum() <= threshold * len(df)


def validate_prices(df: pd.DataFrame) -> bool:
    if not validate_price_schema(df):
        return False
    if not check_missing_ratio(df):
        return False
    if (df['close'] <= 0).any():
        return False
    return True

