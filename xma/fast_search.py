"""Exact block-vectorized evaluation of large moving-average search grids."""

import itertools
import math

import numpy as np
import pandas as pd


_TRADING_DAYS = 252.0
_SQRT_TRADING_DAYS = math.sqrt(_TRADING_DAYS)


def _turnover(weights):
    """Vectorized equivalent of diff().abs().fillna(abs(weight))."""
    previous = np.empty_like(weights)
    previous[:, 0] = np.nan
    previous[:, 1:] = weights[:, :-1]
    difference = np.abs(weights - previous)
    return np.where(np.isfinite(difference), difference, np.abs(weights))


def _benchmark_row(trade_symbol, benchmark_returns, cash_daily, start, end):
    """Build one buy-and-hold summary row for a common evaluation window."""
    observations = len(benchmark_returns)
    years = observations / _TRADING_DAYS
    growth = np.prod(1.0 + benchmark_returns)
    cagr = growth ** (1.0 / years) - 1.0
    volatility = np.std(benchmark_returns, ddof=1) * _SQRT_TRADING_DAYS
    sharpe = (
        (np.mean(benchmark_returns) - cash_daily) * _TRADING_DAYS / volatility
        if volatility > 0.0 else float("nan")
    )
    return {
        "type": "Buy-and-hold", "signal": "—", "fast": "—", "slow": "—",
        "threshold": None, "down_pos": None, "down_symbol": "—",
        "trade": trade_symbol.upper(), "return": float(cagr),
        "volatility": float(volatility), "sharpe": float(sharpe),
        "beta": 1.0, "alpha": 0.0, "trades_per_year": 0.0,
        "avg_position": 1.0, "cash_weight": 0.0, "deviation": None,
        "start": start, "end": end, "trading_days": observations,
        "years": years,
    }


def _evaluate_signal_trade(
    signal_symbol,
    trade_symbol,
    args,
    fast_group,
    slow_group,
    thresholds,
    signal_prices,
    trade_prices,
    down_prices,
    ma_cache,
    keep_returns,
):
    """Evaluate one signal/traded-asset grid exactly in vectorized blocks."""
    common_dates = signal_prices.index.intersection(trade_prices.index).sort_values()
    if down_prices is not None:
        common_dates = common_dates.intersection(down_prices.index).sort_values()
    if common_dates.empty:
        return None

    asset_return_series = trade_prices.reindex(common_dates).pct_change()
    asset_returns = asset_return_series.to_numpy(dtype=float)
    down_returns = (
        down_prices.reindex(common_dates).pct_change().to_numpy(dtype=float)
        if down_prices is not None else None
    )

    effective_min = getattr(args, "effective_trade_date_min", args.trade_date_min)
    date_mask = np.ones(len(common_dates), dtype=bool)
    if effective_min is not None:
        date_mask &= np.asarray(common_dates >= effective_min)
    if args.trade_date_max is not None:
        date_mask &= np.asarray(common_dates <= args.trade_date_max)
    date_mask &= np.isfinite(asset_returns)
    if down_returns is not None:
        date_mask &= np.isfinite(down_returns)
    if np.count_nonzero(date_mask) < 2:
        return None

    evaluation_dates = common_dates[date_mask]
    benchmark_returns = asset_returns[date_mask]
    cash_daily = (1.0 + args.cash_rate) ** (1.0 / _TRADING_DAYS) - 1.0
    observations = len(evaluation_dates)
    years = observations / _TRADING_DAYS

    lengths = sorted(set(fast_group) | set(slow_group))
    cached = ma_cache[signal_symbol]
    ma_matrix = np.vstack([
        cached[length].reindex(common_dates).to_numpy(dtype=float)
        for length in lengths
    ])
    length_index = {length: index for index, length in enumerate(lengths)}
    last_ma = {length: float(cached[length].iloc[-1]) for length in lengths}

    combinations = list(itertools.product(fast_group, slow_group, thresholds))
    rows = []
    block_size = args.search_block_size
    shift = 1 + args.lag
    benchmark_mean = float(np.mean(benchmark_returns))
    benchmark_variance = float(np.var(benchmark_returns, ddof=1))
    benchmark_excess_mean = (benchmark_mean - cash_daily) * _TRADING_DAYS

    for offset in range(0, len(combinations), block_size):
        block = combinations[offset:offset + block_size]
        fast_lengths = np.fromiter((item[0] for item in block), dtype=int)
        slow_lengths = np.fromiter((item[1] for item in block), dtype=int)
        block_thresholds = np.fromiter((item[2] for item in block), dtype=float)
        fast_indices = np.fromiter(
            (length_index[int(length)] for length in fast_lengths), dtype=int
        )
        slow_indices = np.fromiter(
            (length_index[int(length)] for length in slow_lengths), dtype=int
        )
        fast_ma = ma_matrix[fast_indices]
        slow_ma = ma_matrix[slow_indices]
        valid_signal = np.isfinite(fast_ma) & np.isfinite(slow_ma)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = fast_ma / slow_ma - 1.0
        raw_up = deviation > block_thresholds[:, None]
        up_state = np.where(valid_signal, raw_up.astype(float), np.nan)
        shifted_up = np.full_like(up_state, np.nan)
        if shift < up_state.shape[1]:
            shifted_up[:, shift:] = up_state[:, :-shift]

        positions = np.where(
            np.isfinite(shifted_up),
            np.where(shifted_up > 0.5, 1.0, args.down_pos),
            np.nan,
        )
        if down_returns is not None:
            down_weights = np.where(
                np.isfinite(shifted_up),
                np.where(shifted_up > 0.5, 0.0, 1.0 - args.down_pos),
                np.nan,
            )
        else:
            down_weights = None

        cash_weights = (
            1.0 - positions - down_weights
            if down_weights is not None else 1.0 - positions
        )
        strategy_returns = (
            positions * asset_returns[None, :]
            + cash_weights * cash_daily
        )
        if down_weights is not None:
            strategy_returns += down_weights * down_returns[None, :]

        if args.cost:
            turnover = _turnover(positions)
            if down_weights is not None:
                turnover += _turnover(down_weights)
            strategy_returns -= args.cost * turnover

        evaluated_returns = strategy_returns[:, date_mask]
        evaluated_positions = positions[:, date_mask]
        evaluated_cash = cash_weights[:, date_mask]
        if not (
            np.all(np.isfinite(evaluated_returns))
            and np.all(np.isfinite(evaluated_positions))
            and np.all(np.isfinite(evaluated_cash))
        ):
            # The fast path relies on a common window in which every tested
            # system has a valid shifted signal. Let the caller use the scalar
            # implementation if unusual missing data violate that assumption.
            return None

        growth = np.prod(1.0 + evaluated_returns, axis=1)
        cagr = np.power(growth, 1.0 / years) - 1.0
        means = np.mean(evaluated_returns, axis=1)
        volatility = np.std(evaluated_returns, axis=1, ddof=1) * _SQRT_TRADING_DAYS
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = (means - cash_daily) * _TRADING_DAYS / volatility
        sharpe = np.where(volatility > 0.0, sharpe, np.nan)

        if benchmark_variance > 0.0:
            centered_returns = evaluated_returns - means[:, None]
            centered_benchmark = benchmark_returns - benchmark_mean
            covariance = (
                centered_returns @ centered_benchmark / (observations - 1.0)
            )
            beta = covariance / benchmark_variance
        else:
            beta = np.full(len(block), np.nan)
        alpha = (means - cash_daily) * _TRADING_DAYS - beta * benchmark_excess_mean

        changes = np.count_nonzero(
            np.abs(np.diff(evaluated_positions, axis=1)) > 1.0e-12,
            axis=1,
        )
        changes += (np.abs(evaluated_positions[:, 0]) > 1.0e-12).astype(int)
        changes_per_year = changes / years
        average_position = np.mean(evaluated_positions, axis=1)
        average_cash = np.mean(evaluated_cash, axis=1)

        for index, (fast, slow, threshold) in enumerate(block):
            final_deviation = last_ma[fast] / last_ma[slow] - 1.0
            row = {
                "type": "Strategy", "signal": signal_symbol.upper(),
                "fast": fast, "slow": slow, "threshold": threshold,
                "down_pos": args.down_pos,
                "down_symbol": args.down_symbol.upper() if args.down_symbol else "—",
                "trade": trade_symbol.upper(), "return": float(cagr[index]),
                "volatility": float(volatility[index]),
                "sharpe": float(sharpe[index]), "beta": float(beta[index]),
                "alpha": float(alpha[index]),
                "trades_per_year": float(changes_per_year[index]),
                "avg_position": float(average_position[index]),
                "cash_weight": float(average_cash[index]),
                "deviation": float(final_deviation),
                "start": evaluation_dates[0], "end": evaluation_dates[-1],
                "trading_days": observations, "years": years,
            }
            if keep_returns:
                row["_returns"] = pd.Series(
                    evaluated_returns[index].copy(), index=evaluation_dates
                )
            rows.append(row)

    rows.append(
        _benchmark_row(
            trade_symbol,
            benchmark_returns,
            cash_daily,
            evaluation_dates[0],
            evaluation_dates[-1],
        )
    )
    return rows


def vectorized_summary_rows(
    args,
    signal_symbols,
    fast_group,
    slow_group,
    thresholds,
    trade_symbols,
    signal_prices,
    trade_prices_by_symbol,
    down_prices,
):
    """Return exact common-window summary rows, or None to request fallback."""
    if not args.common_window or not args.terse or not trade_symbols:
        return None

    keep_returns = bool(args.deflate)
    rows = []
    for signal_symbol in signal_symbols:
        for trade_symbol in trade_symbols:
            pair_rows = _evaluate_signal_trade(
                signal_symbol,
                trade_symbol,
                args,
                fast_group,
                slow_group,
                thresholds,
                signal_prices[signal_symbol],
                trade_prices_by_symbol[trade_symbol],
                down_prices,
                args._ma_length_cache,
                keep_returns,
            )
            if pair_rows is None:
                return None
            rows.extend(pair_rows)
    return rows
