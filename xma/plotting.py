"""Cumulative-return plotting."""

import pandas as pd

from .backtest import transaction_cost


def plot_cumulative_returns(args, signal_symbols, fast_group, slow_group, thresholds,
                           trade_symbols, down_symbol, signal_prices,
                           trade_prices_by_symbol, down_prices):
    """Display cumulative return charts comparing each traded asset's systems."""
    import matplotlib.pyplot as plt

    for trade_symbol in trade_symbols:
        asset = trade_prices_by_symbol[trade_symbol]
        common_dates = asset.index
        for ticker in signal_symbols:
            common_dates = common_dates.intersection(signal_prices[ticker].index)
        if down_prices is not None:
            common_dates = common_dates.intersection(down_prices.index)
        common_dates = common_dates.sort_values()
        if common_dates.empty:
            print(f"No common price dates available to plot {trade_symbol}.")
            continue

        asset_returns = asset.reindex(common_dates).pct_change()
        down_returns = down_prices.reindex(common_dates).pct_change() if down_prices is not None else None
        system_returns = {}
        component_positions = {}
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
                        ).astype("boolean")
                        up_state = up_state.reindex(common_dates).ffill().shift(1 + args.lag)
                        position = up_state.map({True: 1.0, False: args.down_pos})
                        down_weight = (
                            up_state.map({True: 0.0, False: 1.0 - args.down_pos})
                            if down_prices is not None else 0.0
                        )
                        cash_weight = 1.0 - position - down_weight
                        cash_daily = (1.0 + args.cash_rate) ** (1.0 / 252.0) - 1.0
                        returns = position * asset_returns + cash_weight * cash_daily
                        if down_returns is not None:
                            returns = returns + down_weight * down_returns
                        returns = returns - transaction_cost(
                            position, args.cost, down_weight if down_returns is not None else None
                        )
                        label = f"{ticker} MA1({fast})/MA2({slow}) {threshold:.2%}"
                        system_returns[label] = returns.dropna()
                        component_positions[label] = position

        if args.average:
            position_frame = pd.DataFrame(component_positions, index=common_dates).dropna(how="any")
            avg_position = position_frame.mean(axis=1)
            # Rebuild average down-symbol exposure from the averaged component states.
            if down_prices is not None:
                down_exposure = pd.Series(0.0, index=position_frame.index)
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
                                up = ((fast_ma / slow_ma - 1) > threshold).where(
                                    fast_ma.notna() & slow_ma.notna()
                                ).astype("boolean")
                                up = up.reindex(common_dates).ffill().shift(1 + args.lag)
                                down_exposure = down_exposure.add(
                                    up.map({True: 0.0, False: 1.0 - args.down_pos})
                                    .reindex(position_frame.index), fill_value=0.0
                                )
                down_exposure /= len(component_positions)
            else:
                down_exposure = pd.Series(0.0, index=position_frame.index)
            avg_position = avg_position.reindex(position_frame.index)
            avg_cash = 1.0 - avg_position - down_exposure
            avg_returns = avg_position * asset_returns.reindex(avg_position.index) + avg_cash * cash_daily
            if down_returns is not None:
                avg_returns = avg_returns + down_exposure * down_returns.reindex(avg_position.index)
            avg_returns = avg_returns - transaction_cost(
                avg_position, args.cost,
                down_exposure if down_returns is not None else None,
            )
            system_returns["Average system"] = avg_returns.dropna()

        valid_curves = {label: series for label, series in system_returns.items() if not series.empty}
        if not valid_curves:
            print(f"No valid strategy returns available to plot {trade_symbol}.")
            continue
        plot_dates = valid_curves[next(iter(valid_curves))].index
        for series in valid_curves.values():
            plot_dates = plot_dates.intersection(series.index)
        if args.effective_trade_date_min is not None:
            plot_dates = plot_dates[plot_dates >= args.effective_trade_date_min]
        if args.trade_date_max is not None:
            plot_dates = plot_dates[plot_dates <= args.trade_date_max]
        if plot_dates.empty:
            print(f"No common strategy return dates available to plot {trade_symbol}.")
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        benchmark = asset_returns.reindex(plot_dates).dropna()
        ax.plot(benchmark.index, (1.0 + benchmark).cumprod(), label=f"Buy-and-hold {trade_symbol}", linewidth=2)
        for label, series in valid_curves.items():
            selected = series.reindex(plot_dates).dropna()
            ax.plot(selected.index, (1.0 + selected).cumprod(), label=label)
        ax.set_title(f"Cumulative returns: {trade_symbol.upper()}")
        ax.set_ylabel("Growth of $1")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
    if plt.get_fignums():
        plt.show()
