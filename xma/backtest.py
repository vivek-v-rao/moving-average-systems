"""Strategy construction and backtesting."""

import pandas as pd

from .reporting import print_annual_returns
from .signals import crossover_dates_from_mas


def transaction_cost(position, cost_rate, other_position=None):
    """Return per-session cost from absolute changes in asset weights."""
    turnover = position.diff().abs().fillna(position.abs())
    if other_position is not None:
        turnover = turnover.add(
            other_position.diff().abs().fillna(other_position.abs()), fill_value=0.0
        )
    return cost_rate * turnover

def report_system(symbol, fast, slow, threshold, trade_symbol, down_symbol, args, prices,
                  trade_prices, down_prices, summary_rows):
    print(f"\n=== {symbol.upper()} MA1({fast}) / MA2({slow}), threshold {threshold:.2%} ===")
    cache_key = (symbol, fast, slow, threshold)
    length_cache = getattr(args, "_ma_length_cache", {}).get(symbol, {})
    fast_ma = length_cache.get(fast)
    slow_ma = length_cache.get(slow)
    if fast_ma is None:
        fast_ma = prices.rolling(fast, min_periods=fast).mean()
    if slow_ma is None:
        slow_ma = prices.rolling(slow, min_periods=slow).mean()
    signal_cache = getattr(args, "_signal_event_cache", None)
    if signal_cache is None:
        signal_cache = {}
        args._signal_event_cache = signal_cache
    ratio_difference = fast_ma.iloc[-1] / slow_ma.iloc[-1] - 1
    if args.days > 0 and not args.terse:
        all_signals = signal_cache.get(cache_key)
        if all_signals is None:
            all_signals = crossover_dates_from_mas(fast_ma, slow_ma, threshold)
            signal_cache[cache_key] = all_signals
        cutoff = prices.index[max(0, len(prices) - args.days)]
        signal_positions = prices.index.get_indexer(
            pd.DatetimeIndex([date for date, _ in all_signals])
        )

        print(f"Crossovers (last {args.days} trading days):")
        visible_signal_indices = [
            index for index, (date, _) in enumerate(all_signals) if date >= cutoff
        ]
        if not visible_signal_indices:
            print("No crossovers in the requested display window.")
        else:
            for signal_index in visible_signal_indices:
                date, direction = all_signals[signal_index]
                position = signal_positions[signal_index]
                previous = signal_positions[signal_index - 1] if signal_index else None
                gap = "n/a" if previous is None else str(position - previous)
                print(f"{date:%Y-%m-%d} {direction} {gap} trading days since previous signal")

        current_date = prices.index[-1]
        current_direction = "bullish" if ratio_difference > threshold else (
            "bearish" if ratio_difference < 0 else "neutral"
        )
        if all_signals:
            last_date, last_direction = all_signals[-1]
            active = len(prices.index) - 1 - signal_positions[-1]
            active_text = f"{active} trading days since {last_direction} signal on {last_date:%Y-%m-%d}"
        else:
            active_text = "no crossover found in downloaded history"
        print(
            f"Current data date: {current_date:%Y-%m-%d}; MA1({fast})/MA2({slow}) - 1: "
            f"{ratio_difference:.6f}; current signal: {current_direction} ({active_text})"
        )

    if trade_prices is None:
        return
    # Signals are observed at the close. They affect holdings from the next
    # session; --lag adds further full trading sessions of delay.
    common_dates = prices.index.intersection(trade_prices.index).sort_values()
    if down_prices is not None:
        common_dates = common_dates.intersection(down_prices.index).sort_values()
    up_state = ((fast_ma / slow_ma - 1) > threshold).where(
        fast_ma.notna() & slow_ma.notna()
    ).astype("boolean")
    up_state = up_state.reindex(common_dates).ffill().shift(1 + args.lag)
    state = up_state.map({True: 1.0, False: args.down_pos})
    position = state
    asset_returns = trade_prices.reindex(common_dates).pct_change()
    down_weight = (
        up_state.map({True: 0.0, False: 1.0 - args.down_pos})
        if down_prices is not None else 0.0
    )
    cash_weight_series = 1.0 - state - down_weight
    cash_daily = (1.0 + args.cash_rate) ** (1.0 / 252.0) - 1.0
    strategy_returns = state * asset_returns + cash_weight_series * cash_daily
    if down_prices is not None:
        down_asset_returns = down_prices.reindex(common_dates).pct_change()
        strategy_returns = strategy_returns + down_weight * down_asset_returns
    strategy_returns = strategy_returns - transaction_cost(
        state, args.cost, down_weight if down_prices is not None else None
    )
    strategy_returns = strategy_returns.dropna()
    effective_min = getattr(args, "effective_trade_date_min", args.trade_date_min)
    if effective_min is not None:
        strategy_returns = strategy_returns.loc[strategy_returns.index >= effective_min]
    if args.trade_date_max is not None:
        strategy_returns = strategy_returns.loc[strategy_returns.index <= args.trade_date_max]
    if strategy_returns.empty:
        print("Strategy backtest: not enough overlapping price history after MA warmup.")
        return

    years = len(strategy_returns) / 252.0
    equity = (1.0 + strategy_returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    strategy_cagr = equity.iloc[-1] ** (1.0 / years) - 1.0
    strategy_vol = strategy_returns.std(ddof=1) * (252.0 ** 0.5)
    strategy_sharpe = (
        (strategy_returns - cash_daily).mean() * 252.0 / strategy_vol
        if strategy_vol > 0 else float("nan")
    )
    drawdown = equity / equity.cummax() - 1.0

    benchmark_returns = asset_returns.reindex(strategy_returns.index)
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    benchmark_cagr = benchmark_equity.iloc[-1] ** (1.0 / years) - 1.0
    benchmark_vol = benchmark_returns.std(ddof=1) * (252.0 ** 0.5)
    benchmark_sharpe = (
        (benchmark_returns - cash_daily).mean() * 252.0 / benchmark_vol
        if benchmark_vol > 0 else float("nan")
    )
    mirror_position = up_state.map({True: 0.0, False: 1.0})
    mirror_returns = (
        mirror_position * asset_returns + (1.0 - mirror_position) * cash_daily
    ).reindex(strategy_returns.index).dropna()
    mirror_returns = mirror_returns - transaction_cost(
        mirror_position.reindex(mirror_returns.index), args.cost
    )
    mirror_vol = mirror_returns.std(ddof=1) * (252.0 ** 0.5)
    mirror_sharpe = (
        (mirror_returns - cash_daily).mean() * 252.0 / mirror_vol
        if mirror_vol > 0 else float("nan")
    )

    def beta_alpha(returns):
        aligned_benchmark = benchmark_returns.reindex(returns.index)
        benchmark_variance = aligned_benchmark.var(ddof=1)
        beta = (
            returns.cov(aligned_benchmark) / benchmark_variance
            if benchmark_variance > 0 else float("nan")
        )
        alpha = (
            (returns - cash_daily).mean() * 252.0
            - beta * (aligned_benchmark - cash_daily).mean() * 252.0
        )
        return beta, alpha

    strategy_beta, strategy_alpha = beta_alpha(strategy_returns)
    mirror_beta, mirror_alpha = beta_alpha(mirror_returns)

    evaluated_position = position.reindex(strategy_returns.index)
    evaluated_sessions = len(evaluated_position)
    position_changes = int((evaluated_position.diff().abs() > 1e-12).sum())
    if evaluated_sessions and abs(evaluated_position.iloc[0]) > 1e-12:
        position_changes += 1
    changes_per_year = position_changes / years if years else float("nan")
    average_position = evaluated_position.mean() if evaluated_sessions else float("nan")
    average_cash_weight = cash_weight_series.reindex(strategy_returns.index).mean()
    evaluated_up = up_state.reindex(strategy_returns.index)
    up_fraction = evaluated_up.eq(True).mean()
    down_fraction = 1.0 - up_fraction

    def asset_regime_stats(mask):
        returns = benchmark_returns.loc[mask]
        if returns.empty:
            return float("nan"), float("nan")
        regime_equity = (1.0 + returns).cumprod()
        regime_years = len(returns) / 252.0
        annual_return = regime_equity.iloc[-1] ** (1.0 / regime_years) - 1.0
        annual_vol = returns.std(ddof=1) * (252.0 ** 0.5)
        return annual_return, annual_vol

    up_return, up_vol = asset_regime_stats(evaluated_up == True)
    down_return, down_vol = asset_regime_stats(evaluated_up == False)
    print(
        f"Strategy: position 1 in {trade_symbol.upper()} when signal is up and "
        f"{args.down_pos:g} when down"
        + (f" with {down_symbol.upper()} weight {1.0 - args.down_pos:g}" if down_symbol else "")
        + f"; any residual cash earns {args.cash_rate:.2%} annually."
    )
    if args.cost:
        print(f"Trading cost: {args.cost:.3%} per unit of absolute position turnover.")
    print(
        f"Execution: next session plus {args.lag} lag session(s); period "
        f"{strategy_returns.index[0]:%Y-%m-%d} to {strategy_returns.index[-1]:%Y-%m-%d}."
    )
    print(
        f"Total return: {total_return:.2%}; CAGR: {strategy_cagr:.2%}; "
        f"max drawdown: {drawdown.min():.2%}; buy-and-hold {trade_symbol.upper()}: "
        f"{(1.0 + benchmark_returns).prod() - 1.0:.2%}"
    )
    print(
        f"Position changes: {position_changes} ({changes_per_year:.2f}/year); "
        f"average position: {average_position:.2f}; cash weight: {average_cash_weight:.2f}; "
        f"signal up: {up_fraction:.1%}; down: {down_fraction:.1%}"
    )
    print("Annualized performance:")
    print(
        f"{'System / regime':<24} {'Return':>11} {'Volatility':>12} "
        f"{'Sharpe':>9} {'Beta':>8} {'Alpha':>10}"
    )
    print(
        f"{'Strategy':<24} {strategy_cagr:>10.2%} {strategy_vol:>11.2%} "
        f"{strategy_sharpe:>9.2f} {strategy_beta:>8.2f} {strategy_alpha:>9.2%}"
    )
    print(
        f"{'Buy-and-hold':<24} {benchmark_cagr:>10.2%} {benchmark_vol:>11.2%} "
        f"{benchmark_sharpe:>9.2f} {1.0:>8.2f} {0.0:>9.2%}"
    )
    print(
        f"{'Asset, signal up':<24} {up_return:>10.2%} {up_vol:>11.2%} "
        f"{strategy_sharpe:>9.2f} {strategy_beta:>8.2f} {strategy_alpha:>9.2%}"
    )
    print(
        f"{'Asset, signal down':<24} {down_return:>10.2%} {down_vol:>11.2%} "
        f"{mirror_sharpe:>9.2f} {mirror_beta:>8.2f} {mirror_alpha:>9.2%}"
    )
    print("Alpha is annualized relative to buy-and-hold of the traded symbol; regime rows show asset return/volatility and the corresponding strategy metrics.")
    if args.annual:
        print_annual_returns(
            strategy_returns, benchmark_returns, evaluated_position,
            args.cash_rate,
        )

    summary_rows.append({
        "type": "Strategy", "signal": symbol.upper(), "fast": fast, "slow": slow,
        "threshold": threshold,
        "down_pos": args.down_pos, "down_symbol": down_symbol.upper() if down_symbol else "—",
        "trade": trade_symbol.upper(), "return": strategy_cagr,
        "volatility": strategy_vol, "sharpe": strategy_sharpe,
        "beta": strategy_beta, "alpha": strategy_alpha,
        "trades_per_year": changes_per_year, "avg_position": average_position, "cash_weight": average_cash_weight,
        "deviation": ratio_difference,
        "start": strategy_returns.index[0], "end": strategy_returns.index[-1],
        "trading_days": len(strategy_returns), "years": years,
        "_returns": strategy_returns.copy(),
    })
    summary_rows.append({
        "type": "Buy-and-hold", "signal": "—", "fast": "—", "slow": "—",
        "threshold": None, "down_pos": None, "down_symbol": "—",
        "trade": trade_symbol.upper(), "return": benchmark_cagr,
        "volatility": benchmark_vol, "sharpe": benchmark_sharpe,
        "beta": 1.0, "alpha": 0.0, "trades_per_year": 0.0, "avg_position": 1.0, "cash_weight": 0.0,
        "deviation": None,
        "start": strategy_returns.index[0], "end": strategy_returns.index[-1],
        "trading_days": len(strategy_returns), "years": years,
    })

def report_average_system(trade_symbol, down_symbol, args, signal_symbols, fast_group, slow_group, thresholds,
                          signal_prices, trade_prices, down_prices, summary_rows):
    """Backtest equal average exposure across every tested signal system."""
    asset = trade_prices[trade_symbol]
    dates = asset.index
    for ticker in signal_symbols:
        dates = dates.intersection(signal_prices[ticker].index)
    if down_prices is not None:
        dates = dates.intersection(down_prices.index)
    dates = dates.sort_values()
    component_positions = {}
    component_down_positions = {}
    for ticker in signal_symbols:
        prices = signal_prices[ticker]
        length_cache = getattr(args, "_ma_length_cache", {}).get(ticker, {})
        for fast in fast_group:
            fast_ma = length_cache.get(fast)
            if fast_ma is None:
                fast_ma = prices.rolling(fast, min_periods=fast).mean()
            for slow in slow_group:
                slow_ma = length_cache.get(slow)
                if slow_ma is None:
                    slow_ma = prices.rolling(slow, min_periods=slow).mean()
                for threshold in thresholds:
                    up_state = ((fast_ma / slow_ma - 1) > threshold).where(
                    fast_ma.notna() & slow_ma.notna()
                    )
                    state = up_state.map({True: 1.0, False: args.down_pos})
                    key = f"{ticker} MA1({fast})/MA2({slow}) threshold {threshold:g}"
                    component_positions[key] = state.reindex(dates).ffill().shift(1 + args.lag)
                    if down_prices is not None:
                        component_down_positions[key] = (
                            up_state.map({True: 0.0, False: 1.0 - args.down_pos})
                            .reindex(dates).ffill().shift(1 + args.lag)
                        )

    positions = pd.DataFrame(component_positions, index=dates).dropna(how="any")
    exposure = positions.mean(axis=1)
    down_exposure = (
        pd.DataFrame(component_down_positions, index=dates).reindex(exposure.index).mean(axis=1)
        if down_prices is not None else pd.Series(0.0, index=exposure.index)
    )
    asset_returns = asset.reindex(dates).pct_change().reindex(exposure.index)
    cash_daily = (1.0 + args.cash_rate) ** (1.0 / 252.0) - 1.0
    cash_weight_series = 1.0 - exposure - down_exposure
    system_returns = exposure * asset_returns + cash_weight_series * cash_daily
    if down_prices is not None:
        down_asset_returns = down_prices.reindex(dates).pct_change().reindex(exposure.index)
        system_returns = system_returns + down_exposure * down_asset_returns
    system_returns = system_returns - transaction_cost(
        exposure, args.cost, down_exposure if down_prices is not None else None
    )
    system_returns = system_returns.dropna()
    effective_min = getattr(args, "effective_trade_date_min", args.trade_date_min)
    if effective_min is not None:
        system_returns = system_returns.loc[system_returns.index >= effective_min]
    if args.trade_date_max is not None:
        system_returns = system_returns.loc[system_returns.index <= args.trade_date_max]
    if system_returns.empty:
        print(f"Average system for {trade_symbol}: no returns in the selected date window.")
        return

    exposure = exposure.reindex(system_returns.index)
    cash_weight_series = cash_weight_series.reindex(system_returns.index)
    asset_returns = asset_returns.reindex(system_returns.index)
    years = len(system_returns) / 252.0
    equity = (1.0 + system_returns).cumprod()
    annual_return = equity.iloc[-1] ** (1.0 / years) - 1.0
    volatility = system_returns.std(ddof=1) * (252.0 ** 0.5)
    sharpe = (
        (system_returns - cash_daily).mean() * 252.0 / volatility
        if volatility > 0 else float("nan")
    )
    benchmark_variance = asset_returns.var(ddof=1)
    beta = system_returns.cov(asset_returns) / benchmark_variance if benchmark_variance > 0 else float("nan")
    alpha = (
        (system_returns - cash_daily).mean() * 252.0
        - beta * (asset_returns - cash_daily).mean() * 252.0
    )
    changes = int((exposure.diff().abs() > 1e-12).sum())
    if exposure.iloc[0] > 0:
        changes += 1
    actions_per_year = changes / years
    average_exposure = exposure.mean()
    average_cash_weight = cash_weight_series.mean()
    fully_flat_fraction = (exposure == 0).mean()
    total_return = equity.iloc[-1] - 1.0
    max_drawdown = (equity / equity.cummax() - 1.0).min()

    print(f"\n=== Average system trading {trade_symbol.upper()} ===")
    print(
        f"Equal average exposure across {len(component_positions)} systems; "
        f"cash earns {args.cash_rate:.2%} annually."
        + (f" Down-state weight {down_symbol.upper()} = 1 - down position." if down_symbol else "")
    )
    print(
        f"Execution: next session plus {args.lag} lag session(s); period "
        f"{system_returns.index[0]:%Y-%m-%d} to {system_returns.index[-1]:%Y-%m-%d}."
    )
    print(
        f"Total return: {total_return:.2%}; CAGR: {annual_return:.2%}; "
        f"max drawdown: {max_drawdown:.2%}"
    )
    print(
        f"Exposure rebalances: {changes} ({actions_per_year:.2f}/year); "
        f"average position: {average_exposure:.2f}; average cash weight: {average_cash_weight:.2f}; "
        f"net exposure zero: {fully_flat_fraction:.1%}"
    )
    print(
        f"Annualized return: {annual_return:.2%}; volatility: {volatility:.2%}; "
        f"Sharpe: {sharpe:.2f}; beta: {beta:.2f}; alpha: {alpha:.2%}"
    )
    if args.annual:
        print_annual_returns(
            system_returns, asset_returns, exposure, args.cash_rate, "Average"
        )

    summary_rows.append({
        "type": "*Average*", "signal": "Average", "fast": "—", "slow": "—",
        "threshold": "All",
        "down_pos": args.down_pos, "down_symbol": down_symbol.upper() if down_symbol else "—",
        "trade": trade_symbol.upper(), "return": annual_return,
        "volatility": volatility, "sharpe": sharpe, "beta": beta, "alpha": alpha,
        "trades_per_year": actions_per_year, "avg_position": average_exposure,
        "cash_weight": average_cash_weight, "deviation": None,
        "start": system_returns.index[0], "end": system_returns.index[-1],
        "trading_days": len(system_returns), "years": years,
        "_returns": system_returns.copy(),
    })
