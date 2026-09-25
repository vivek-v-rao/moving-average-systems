"""Text and table reporting for backtest results."""

import math

import numpy as np

from .deflation import analyze_deflation_for_trade, subset_galwey_count


def print_deflation_report(rows, trade_symbols, simulations, seed, top_count=5):
    """Print effective-trial and Sharpe deflation diagnostics by traded asset."""
    analyses = {}
    for trade_index, trade in enumerate(trade_symbols):
        trade_rows = [row for row in rows if row["trade"] == trade.upper()]
        analysis = analyze_deflation_for_trade(
            trade_rows, simulations, seed + trade_index
        )
        if analysis is not None:
            analyses[trade.upper()] = analysis
    if not analyses:
        print("\nDeflation diagnostics: no finite strategy returns are available.")
        return

    print("\nDeflation diagnostics (Sharpe ratios are annualized):")
    print(
        "  EffN-B = Bailey-Lopez de Prado average-correlation implied trials; "
        "EffN-G = Galwey eigenvalue effective tests; EffN-max = the independent "
        "Gaussian trial count with the same expected maximum as the full "
        "estimated return-correlation matrix."
    )
    print(
        "  SR0 is the standard DSR search hurdle using EffN-B and the "
        "cross-trial Sharpe SD. Haircut = observed Sharpe - SR0."
    )
    header = (
        f"{'Trade':<8} {'Raw':>5} {'Obs':>7} {'MeanCorr':>9} {'EffN-B':>8} "
        f"{'EffN-G':>8} {'EffN-max':>9} {'MeanSR':>7} {'SR sd':>7} {'SR0':>7} "
        f"{'BestSR':>7} {'Haircut':>8} {'DSR':>7} {'Max-p':>8} {'EB SR':>7}"
    )
    print(header)
    for trade, analysis in analyses.items():
        best = analysis["best"]
        print(
            f"{trade:<8} {analysis['raw_count']:>5d} "
            f"{analysis['common_observations']:>7d} "
            f"{analysis['mean_correlation']:>9.3f} "
            f"{analysis['bailey_count']:>8.2f} "
            f"{analysis['galwey_count']:>8.2f} "
            f"{analysis['max_equivalent_count']:>9.2f} "
            f"{analysis['family_mean']:>7.3f} "
            f"{analysis['sharpe_sd']:>7.3f} "
            f"{analysis['selection_hurdle']:>7.3f} "
            f"{best['sharpe']:>7.3f} "
            f"{best.get('_haircut_sharpe', float('nan')):>8.3f} "
            f"{best.get('_dsr', float('nan')):>7.3f} "
            f"{best.get('_max_test_p', float('nan')):>8.4f} "
            f"{best.get('_eb_sharpe', float('nan')):>7.3f}"
        )

    for trade, analysis in analyses.items():
        print(f"\n{trade}: highest-Sharpe candidates after search adjustment")
        print(
            f"{'Signal':<8} {'MA1':>5} {'MA2':>5} {'Threshold':>10} "
            f"{'Sharpe':>7} {'PSR':>7} {'DSR':>7} {'Max-p':>8} "
            f"{'Haircut':>8} {'EB SR':>7}"
        )
        ranked = sorted(
            analysis["candidates"], key=lambda row: float(row["sharpe"]), reverse=True
        )
        display_rows = ranked if top_count == 0 else ranked[:top_count]
        for row in display_rows:
            print(
                f"{row['signal']:<8} {str(row['fast']):>5} {str(row['slow']):>5} "
                f"{format_threshold(row['threshold']):>10} "
                f"{row['sharpe']:>7.3f} {row.get('_psr', float('nan')):>7.3f} "
                f"{row.get('_dsr', float('nan')):>7.3f} "
                f"{row.get('_max_test_p', float('nan')):>8.4f} "
                f"{row.get('_haircut_sharpe', float('nan')):>8.3f} "
                f"{row.get('_eb_sharpe', float('nan')):>7.3f}"
            )

        ordinary = [row for row in analysis["candidates"] if row["type"] == "Strategy"]
        signals = list(dict.fromkeys(row["signal"] for row in ordinary))
        if ordinary and (len(signals) > 1 or len(ordinary) > len(signals)):
            print(f"{trade}: Galwey redundancy decomposition")
            for signal in signals:
                signal_rows = [row for row in ordinary if row["signal"] == signal]
                raw, effective = subset_galwey_count(signal_rows)
                if raw > 1:
                    print(
                        f"  MA/threshold variants within {signal}: "
                        f"raw {raw}, effective {effective:.2f}"
                    )
            groups = {}
            for row in ordinary:
                key = (row["fast"], row["slow"], row["threshold"])
                groups.setdefault(key, []).append(row)
            signal_effective = []
            signal_raw = []
            for group_rows in groups.values():
                raw, effective = subset_galwey_count(group_rows)
                if raw > 1 and math.isfinite(effective):
                    signal_raw.append(raw)
                    signal_effective.append(effective)
            if signal_effective:
                print(
                    "  Signal assets at fixed MA/threshold: "
                    f"raw {max(signal_raw)}, mean effective {np.mean(signal_effective):.2f}, "
                    f"range {np.min(signal_effective):.2f}-{np.max(signal_effective):.2f}"
                )

    print(
        "\nInterpretation: DSR is a multiple-testing-adjusted Sharpe significance "
        "diagnostic, Max-p is a correlation-aware familywise p-value under a "
        "Gaussian max-test approximation, and EB SR is an empirical-Bayes "
        "shrinkage estimate toward the mean Sharpe of the tested family. None "
        "is a substitute for genuinely untouched out-of-sample data."
    )

def print_annual_returns(
    system_returns, benchmark_returns, position, cash_rate, system_label="Strategy"
):
    """Print yearly returns and annualized daily return statistics."""
    benchmark_returns = benchmark_returns.reindex(system_returns.index)
    position = position.reindex(system_returns.index)
    cash_daily = (1.0 + cash_rate) ** (1.0 / 252.0) - 1.0
    print("Calendar-year returns (mean and volatility annualized; Sharpe over cash rate):")
    print(
        f"{'Year':>4} {system_label:>10} {'Buy&Hold':>10} {'Diff(pp)':>9} "
        f"{'Avg Pos':>9} {'Sys Mean':>9} {'Sys Vol':>9} {'Sys Sharpe':>10} "
        f"{'B&H Mean':>9} {'B&H Vol':>9} {'B&H Sharpe':>10}"
    )
    for year, sys_year in system_returns.groupby(system_returns.index.year):
        bh_year = benchmark_returns.loc[sys_year.index]
        sys_return = (1.0 + sys_year).prod() - 1.0
        bh_return = (1.0 + bh_year).prod() - 1.0
        sys_mean = sys_year.mean() * 252.0
        bh_mean = bh_year.mean() * 252.0
        sys_vol = sys_year.std(ddof=1) * (252.0 ** 0.5)
        bh_vol = bh_year.std(ddof=1) * (252.0 ** 0.5)
        sys_sharpe = (sys_year.mean() - cash_daily) * 252.0 / sys_vol if sys_vol > 0 else float("nan")
        bh_sharpe = (bh_year.mean() - cash_daily) * 252.0 / bh_vol if bh_vol > 0 else float("nan")
        average_position = position.reindex(sys_year.index).mean()
        print(
            f"{year:>4} {sys_return:>10.2%} {bh_return:>10.2%} "
            f"{(sys_return - bh_return) * 100:>9.2f} {average_position:>9.1%} "
            f"{sys_mean:>9.2%} {sys_vol:>9.2%} {sys_sharpe:>10.2f} "
            f"{bh_mean:>9.2%} {bh_vol:>9.2%} {bh_sharpe:>10.2f}"
        )

def format_threshold(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return str(value)

def print_summary_table(rows, cost_rate=0.0, best=False, systems_tested=None):
    """Print all systems, or the highest-Sharpe system per traded symbol."""
    trade_order = list(dict.fromkeys(row["trade"] for row in rows))
    if best:
        summary = []
        for trade in trade_order:
            candidates = [
                row for row in rows
                if row["trade"] == trade and row["type"] in ("Strategy", "*Average*")
            ]
            if not candidates:
                continue
            selected = max(
                candidates,
                key=lambda row: row["sharpe"] if math.isfinite(row["sharpe"]) else float("-inf"),
            )
            selected_window = (
                selected["start"], selected["end"], selected["trading_days"]
            )
            benchmark = next(
                (
                    row for row in rows
                    if row["trade"] == trade and row["type"] == "Buy-and-hold"
                    and (row["start"], row["end"], row["trading_days"]) == selected_window
                ),
                None,
            )
            if benchmark is not None and selected["return"] <= benchmark["return"]:
                summary.append(benchmark)
            else:
                summary.append(selected)
        print(
            "\nBest strategy selected by in-sample Sharpe ratio; showing it only "
            "when its annualized return exceeds buy-and-hold."
        )
        if systems_tested:
            counts = ", ".join(
                f"{trade}: {systems_tested.get(trade, 0)}"
                for trade in trade_order
            )
            print(f"Systems tested per traded asset: {counts}")
    else:
        seen_benchmarks = set()
        summary = []
        for row in rows:
            if row["type"] == "Buy-and-hold":
                if row["trade"] in seen_benchmarks:
                    continue
                seen_benchmarks.add(row["trade"])
            summary.append(row)
    summary.sort(
        key=lambda row: (
            trade_order.index(row["trade"]),
            0 if row["type"] == "Buy-and-hold" else 1,
        )
    )
    finish_summary_table(summary, cost_rate, leading_newline=not best)

def finish_summary_table(summary, cost_rate=0.0, leading_newline=True):
    strategy_rows = [row for row in summary if row["type"] in ("Strategy", "*Average*")]
    threshold_values = sorted({row["threshold"] for row in strategy_rows if row["type"] == "Strategy"})
    show_threshold_column = len(threshold_values) > 1
    window_rows = list(strategy_rows)
    for row in summary:
        if row["type"] == "Buy-and-hold" and not any(
            system_row["trade"] == row["trade"] for system_row in strategy_rows
        ):
            window_rows.append(row)
    windows = {}
    for row in window_rows:
        key = (row["start"], row["end"], row["trading_days"], row["years"])
        windows.setdefault(key, []).append(row)
    if len(windows) == 1:
        start, end, trading_days, years = next(iter(windows))
        print(
            f"{'\n' if leading_newline else ''}Summary table: {start:%Y-%m-%d} to {end:%Y-%m-%d} | "
            f"{trading_days:,} trading days | {years:.2f} years | 252 trading days/year | "
            f"cost {cost_rate:.3%}/unit turnover"
            + (f" | MA deviation threshold: {threshold_values[0]:.2%}" if len(threshold_values) == 1 else "")
        )
    else:
        title = (
            f"{'\n' if leading_newline else ''}Summary table: system date ranges (252 trading days/year) | "
            f"cost {cost_rate:.3%}/unit turnover"
        )
        if len(threshold_values) == 1:
            title += f" | MA deviation threshold: {threshold_values[0]:.2%}"
        print(title)
        for (start, end, trading_days, years), window_rows in windows.items():
            systems = ", ".join(
                (
                    f"Average system / {row['trade']}"
                    if row["type"] == "*Average*"
                    else f"Buy-and-hold / {row['trade']}"
                    if row["type"] == "Buy-and-hold"
                    else f"{row['signal']} MA1({row['fast']})/MA2({row['slow']}) "
                    f"threshold {row['threshold']} / {row['trade']}"
                )
                for row in window_rows
            )
            print(
                f"  {start:%Y-%m-%d} to {end:%Y-%m-%d} | {trading_days:,} trading days | "
                f"{years:.2f} years | {systems}"
            )
    show_down_symbol = any(row.get("down_symbol") not in (None, "—", "") for row in summary)
    show_down_pos = any(row.get("down_pos") not in (None, 0, 0.0) for row in summary)
    header = f"{'Type':<13} {'Signal':<8} {'MA1':>5} {'MA2':>5}"
    if show_threshold_column:
        header += f" {'Threshold':>10}"
    if show_down_pos:
        header += f" {'DownPos':>8}"
    header += f" {'Trade':<8}"
    if show_down_symbol:
        header += f" {'DownSym':<8}"
    header += (
        f" {'Return':>9} {'Vol':>9} {'Sharpe':>7} {'Beta':>7} {'Alpha':>9} "
        f"{'Changes/yr':>10} {'AvgPos':>9} {'CashWt':>9} {'MA Dev':>9}"
    )
    print(header)
    for row in summary:
        deviation = "—" if row["deviation"] is None else f"{row['deviation']:.2%}"
        down_pos = "—" if row["down_pos"] is None else f"{row['down_pos']:g}"
        line = (
            f"{row['type']:<13} {row['signal']:<8} {str(row['fast']):>5} "
            f"{str(row['slow']):>5}"
        )
        if show_threshold_column:
            line += f" {format_threshold(row['threshold']):>10}"
        if show_down_pos:
            line += f" {down_pos:>8}"
        line += f" {row['trade']:<8}"
        if show_down_symbol:
            line += f" {row['down_symbol']:<8}"
        line += (
            f" {row['return']:>9.2%} "
            f"{row['volatility']:>9.2%} {row['sharpe']:>7.2f} {row['beta']:>7.2f} "
            f"{row['alpha']:>9.2%} {row['trades_per_year']:>10.2f} "
            f"{row['avg_position']:>9.2f} {row['cash_weight']:>9.2f} {deviation:>9}"
        )
        print(line)
