"""Moving-average signal construction and moving-average caches."""

import pandas as pd


def precompute_moving_averages(signal_prices, lengths):
    """Precompute each requested simple moving average once per signal symbol."""
    unique_lengths = sorted(set(int(length) for length in lengths))
    return {
        symbol: {
            length: prices.rolling(length, min_periods=length).mean()
            for length in unique_lengths
        }
        for symbol, prices in signal_prices.items()
    }


def crossover_dates_from_mas(fast_ma, slow_ma, threshold=0.0):
    """Return crossover dates from already-computed moving-average series."""
    difference = (fast_ma / slow_ma - 1 - threshold).dropna()
    prior_sign = 0
    signals = []
    for date, value in difference.items():
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prior_sign and sign != prior_sign:
            signals.append((date, "bullish" if sign > 0 else "bearish"))
        prior_sign = sign
    return signals


def crossover_dates(prices: pd.Series, fast_length: int, slow_length: int, threshold: float = 0.0):
    """Return dates where MA1/MA2 - 1 crosses the given threshold."""
    fast = prices.rolling(fast_length, min_periods=fast_length).mean()
    slow = prices.rolling(slow_length, min_periods=slow_length).mean()
    return crossover_dates_from_mas(fast, slow, threshold)
