"""Price loading and download helpers."""

import pandas as pd


def historical_adjusted_close(symbol: str) -> pd.Series:
    """Download adjusted daily closing prices for a single symbol."""
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ValueError(
            "yfinance is required for downloads; install it or use --read-prices-file"
        ) from exc
    data = yf.download(symbol, start="1900-01-01", auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price history returned for {symbol!r}")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise ValueError(f"Yahoo Finance returned no adjusted close for {symbol}")
        prices = data["Adj Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices[symbol] if symbol in prices.columns else prices.iloc[:, 0]
    else:
        if "Adj Close" not in data.columns:
            raise ValueError(f"Yahoo Finance returned no adjusted close for {symbol}")
        prices = data["Adj Close"]
    return pd.to_numeric(prices, errors="coerce").dropna().sort_index()

def read_price_file(path: str) -> pd.DataFrame:
    """Read a date-indexed price CSV written by this script."""
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date").sort_index()

def prices_from_file(frame: pd.DataFrame, symbol: str, allow_adj_close: bool = False):
    """Return a ticker column, or the legacy single-symbol Adj Close column."""
    column = symbol if symbol in frame.columns else (
        "Adj Close" if allow_adj_close and "Adj Close" in frame.columns else None
    )
    if column is None:
        return None
    return pd.to_numeric(frame[column], errors="coerce").dropna().sort_index()
