"""Command-line parsing and top-level orchestration."""

import argparse
import contextlib
import io
import itertools
import math
import sys
import time
from datetime import date

import pandas as pd

from .backtest import report_average_system, report_system
from .fast_search import vectorized_summary_rows
from .data import historical_adjusted_close, prices_from_file, read_price_file
from .plotting import plot_cumulative_returns
from .reporting import print_deflation_report, print_summary_table
from .signals import precompute_moving_averages


def iso_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date.fromisoformat(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from exc

def parse_length_groups(tokens):
    """Parse two MA groups, expanding inclusive start:stop[:step] ranges."""
    groups = []
    current = []
    in_brackets = False
    for token in tokens:
        token = token.replace(",", " ")
        starts = "[" in token
        ends = "]" in token
        if starts:
            if in_brackets or current:
                raise ValueError("each bracketed moving-average group must be separate")
            in_brackets = True
            token = token.replace("[", "")
        if ends:
            token = token.replace("]", "")
        current.extend(part for part in token.split() if part)
        if ends:
            if not in_brackets:
                raise ValueError("found ']' without a matching '['")
            groups.append(current)
            current = []
            in_brackets = False
        elif not in_brackets and not starts:
            groups.append(current)
            current = []
    if in_brackets:
        raise ValueError("missing closing ']' in moving-average group")
    if current:
        groups.append(current)
    if len(groups) != 2 or any(not group for group in groups):
        raise ValueError("provide a first and second moving-average group, such as '[1 5] [21 63]'")
    parsed_groups = []
    for group in groups:
        values = []
        for token in group:
            parts = token.split(":")
            if len(parts) == 1:
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise ValueError("moving-average lengths must be integers or ranges") from exc
                continue
            if len(parts) not in (2, 3):
                raise ValueError(
                    f"invalid moving-average range {token!r}; use start:stop[:step]"
                )
            try:
                start, stop = int(parts[0]), int(parts[1])
                step = int(parts[2]) if len(parts) == 3 else (1 if stop >= start else -1)
            except ValueError as exc:
                raise ValueError(
                    f"invalid moving-average range {token!r}; bounds and step must be integers"
                ) from exc
            if step == 0:
                raise ValueError(f"moving-average range {token!r} cannot have a zero step")
            if (stop - start) * step < 0:
                raise ValueError(
                    f"step in moving-average range {token!r} moves away from the stop value"
                )
            inclusive_stop = stop + (1 if step > 0 else -1)
            values.extend(range(start, inclusive_stop, step))
        parsed_groups.append(values)
    return parsed_groups

def parse_initial_symbol_group(tokens):
    """Read one ticker or a bracketed ticker group from initial CLI tokens."""
    if not tokens:
        raise ValueError("provide at least one signal symbol")
    if "[" not in tokens[0]:
        return [tokens[0]], tokens[1:]
    symbols = []
    for index, token in enumerate(tokens):
        if index == 0:
            token = token.replace("[", "")
        closes = "]" in token
        token = token.replace("]", "").replace(",", " ")
        symbols.extend(part for part in token.split() if part)
        if closes:
            return symbols, tokens[index + 1:]
    raise ValueError("missing closing ']' in signal-symbol group")

def parse_symbol_group(tokens, option_name):
    """Read one ticker or a bracketed/list of tickers from an option."""
    if not tokens:
        raise ValueError(f"{option_name} needs at least one symbol")
    if "[" not in tokens[0]:
        if any("[" in token or "]" in token for token in tokens):
            raise ValueError(f"malformed bracketed symbol list for {option_name}")
        return tokens
    symbols = []
    in_brackets = True
    for index, token in enumerate(tokens):
        if index == 0:
            token = token.replace("[", "")
        closes = "]" in token
        token = token.replace("]", "").replace(",", " ")
        symbols.extend(part for part in token.split() if part)
        if closes:
            if index != len(tokens) - 1:
                raise ValueError(f"unexpected values after bracketed {option_name} list")
            in_brackets = False
            break
    if in_brackets:
        raise ValueError(f"missing closing ']' in {option_name} symbol list")
    if not symbols:
        raise ValueError(f"{option_name} symbol list is empty")
    return symbols

def parse_real_values(tokens, option_name):
    """Parse real numbers from option tokens, optionally enclosed in brackets."""
    values = []
    for token in tokens:
        token = token.replace("[", " ").replace("]", " ").replace(",", " ")
        values.extend(token.split())
    try:
        result = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{option_name} values must be real numbers") from exc
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{option_name} needs finite real values")
    return result

def main() -> int:
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Print moving-average crossovers and optionally backtest them."
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="VALUE",
        help="signal ticker(s), then MA1 and MA2 lengths in that order; use inclusive ranges start:stop[:step] or bracketed groups",
    )
    parser.add_argument("--read-prices-file", metavar="PATH", help="read signal-symbol adjusted closes from a CSV")
    parser.add_argument("--write-prices-file", metavar="PATH", help="write signal-symbol adjusted closes to a CSV")
    parser.add_argument("--days", type=int, default=252, metavar="N", help="signal lookback in trading days (default: 252; <= 0 prints summaries only)")
    parser.add_argument("--madev-thresh", nargs="+", metavar="VALUE", default=["0"], help="MA-deviation threshold(s), for example '[-0.01 0 0.01]' (default: 0)")
    parser.add_argument("--trade", nargs="+", metavar="SYMBOL", help="traded ticker(s) to hold when the signal is bullish")
    parser.add_argument("--down-symbol", metavar="SYMBOL", help="asset held when the signal is down, with weight 1 - --down-pos")
    parser.add_argument("--trade-date-min", type=iso_date, metavar="YYYY-MM-DD", help="inclusive backtest start date")
    parser.add_argument("--trade-date-max", type=iso_date, metavar="YYYY-MM-DD", help="inclusive backtest end date")
    parser.add_argument("--common-window", action="store_true", help="use the default shared backtest window")
    parser.add_argument("--individual-windows", action="store_true", help="give each system its own available backtest window instead of the default common window")
    parser.add_argument("--summary-table", action="store_true", help="print a compact comparison table at the end")
    parser.add_argument("--best", action="store_true", help="print the highest-Sharpe system per traded asset (in-sample; implies --summary-table)")
    parser.add_argument("--terse", action="store_true", help="print only the summary table (implies --summary-table)")
    parser.add_argument("--annual", action="store_true", help="print calendar-year returns for each system and buy-and-hold")
    parser.add_argument("--average", action="store_true", help="also test the equal average position across all systems for each traded symbol")
    parser.add_argument(
        "--deflate", action="store_true",
        help="estimate effective strategy count and print selection-bias-adjusted Sharpe diagnostics",
    )
    parser.add_argument(
        "--deflate-sims", type=int, default=20000, metavar="N",
        help="Gaussian max-test simulations used by --deflate (default: 20000)",
    )
    parser.add_argument(
        "--deflate-seed", type=int, default=12345, metavar="N",
        help="random seed for reproducible --deflate simulations (default: 12345)",
    )
    parser.add_argument(
        "--deflate-top", type=int, default=5, metavar="N",
        help="number of highest-Sharpe candidates to show per traded asset with --deflate; 0 shows all (default: 5)",
    )
    parser.add_argument("--plot", action="store_true", help="display cumulative returns for buy-and-hold and tested systems")
    parser.add_argument("--time", action="store_true", help="print elapsed times for data access, writing, calculations/display, and overall")
    parser.add_argument(
        "--search-block-size", type=int, default=512, metavar="N",
        help="number of strategies evaluated per NumPy block in terse common-window searches (default: 512)",
    )
    parser.add_argument("--lag", type=int, default=0, metavar="N", help="additional trading-session delay (default: 0)")
    parser.add_argument("--cash-rate", type=float, default=0.03, metavar="RATE", help="annual cash interest as a decimal (default: 0.03)")
    parser.add_argument("--cost", type=float, default=0.0, metavar="RATE", help="trading cost per unit of absolute asset-position turnover (default: 0)")
    parser.add_argument("--down-pos", type=float, default=0.0, metavar="VALUE", help="target traded-asset position when signal is down (default: 0; up position is 1)")
    args = parser.parse_args()
    args.show_full_summary = args.summary_table
    if args.terse or args.best or args.deflate:
        args.summary_table = True

    try:
        signal_symbols, length_tokens = parse_initial_symbol_group(args.inputs)
        fast_group, slow_group = parse_length_groups(length_tokens)
        thresholds = parse_real_values(args.madev_thresh, "--madev-thresh")
        trade_symbols = parse_symbol_group(args.trade, "--trade") if args.trade else []
    except ValueError as exc:
        parser.error(str(exc))
    if args.common_window and args.individual_windows:
        parser.error("--common-window and --individual-windows cannot be used together")
    if args.common_window and not trade_symbols:
        parser.error("--common-window requires --trade")
    if args.individual_windows and not trade_symbols:
        parser.error("--individual-windows requires --trade")
    # Shared dates are the default for comparisons across tested systems.
    # Without a traded asset, signal-only crossover output has no backtest window.
    args.common_window = bool(trade_symbols) and not args.individual_windows
    systems = list(itertools.product(fast_group, slow_group))
    if any(length <= 0 for pair in systems for length in pair):
        parser.error("moving-average lengths must be positive integers")
    if args.lag < 0:
        parser.error("--lag must be zero or a positive integer")
    if args.cash_rate <= -1:
        parser.error("--cash-rate must be greater than -1")
    if not math.isfinite(args.cost) or args.cost < 0:
        parser.error("--cost must be a finite nonnegative rate")
    if not math.isfinite(args.down_pos):
        parser.error("--down-pos must be a finite real number")
    if args.trade_date_min and args.trade_date_max and args.trade_date_min > args.trade_date_max:
        parser.error("--trade-date-min must be on or before --trade-date-max")
    if (args.trade_date_min or args.trade_date_max) and not trade_symbols:
        parser.error("--trade-date-min and --trade-date-max require --trade")
    if args.summary_table and not trade_symbols:
        parser.error("--summary-table requires --trade")
    if args.average and not trade_symbols:
        parser.error("--average requires --trade")
    if args.deflate and not trade_symbols:
        parser.error("--deflate requires --trade")
    if args.deflate_sims < 100:
        parser.error("--deflate-sims must be at least 100")
    if args.deflate_top < 0:
        parser.error("--deflate-top must be zero or a positive integer")
    if args.search_block_size <= 0:
        parser.error("--search-block-size must be a positive integer")
    if args.down_symbol and not trade_symbols:
        parser.error("--down-symbol requires --trade")
    if args.plot and not trade_symbols:
        parser.error("--plot requires --trade")
    try:
        data_start = time.perf_counter()
        price_file_data = read_price_file(args.read_prices_file) if args.read_prices_file else None
        price_cache = {}

        def get_prices(ticker, allow_adj_close=False):
            if ticker in price_cache:
                return price_cache[ticker]
            loaded = (
                prices_from_file(price_file_data, ticker, allow_adj_close)
                if price_file_data is not None else None
            )
            price_cache[ticker] = loaded if loaded is not None else historical_adjusted_close(ticker)
            return price_cache[ticker]

        signal_prices = {
            ticker: get_prices(
                ticker,
                allow_adj_close=(len(signal_symbols) == 1 and ticker == signal_symbols[0]),
            )
            for ticker in signal_symbols
        }
        trade_prices_by_symbol = {
            ticker: get_prices(ticker) for ticker in trade_symbols
        }
        down_prices = get_prices(args.down_symbol) if args.down_symbol else None
        data_elapsed = time.perf_counter() - data_start
        all_prices = {**trade_prices_by_symbol, **signal_prices}
        if args.down_symbol:
            all_prices[args.down_symbol] = down_prices
        if len(all_prices) > 1:
            combined_dates = all_prices[next(iter(all_prices))].index
            for prices in list(all_prices.values())[1:]:
                combined_dates = combined_dates.union(prices.index)
            combined_dates = combined_dates.sort_values()
            for ticker, prices in all_prices.items():
                in_range = combined_dates[
                    (combined_dates >= prices.index.min())
                    & (combined_dates <= prices.index.max())
                ]
                missing_dates = in_range.difference(prices.index)
                for date in missing_dates:
                    print(
                        f"Warning: no price for {ticker} on {date:%Y-%m-%d}, "
                        "although another requested symbol has data for that date",
                        file=sys.stderr,
                    )
        if args.trade_date_min is not None:
            for ticker, prices in signal_prices.items():
                for fast, slow in systems:
                    available = prices.loc[prices.index <= args.trade_date_min]
                    required = max(fast, slow)
                    if len(available) < required:
                        raise ValueError(
                            f"{ticker} MA1({fast})/MA2({slow}) has only {len(available)} "
                            f"observations on or before {args.trade_date_min:%Y-%m-%d}; "
                            f"at least {required} are needed to compute the signal"
                        )
        args.effective_trade_date_min = args.trade_date_min
        if args.common_window:
            longest_ma = max(max(fast_group), max(slow_group))
            for ticker, prices in signal_prices.items():
                if len(prices) < longest_ma:
                    raise ValueError(
                        f"common window needs at least {longest_ma} observations for "
                        f"signal symbol {ticker} to compute the longest moving average"
                    )
            all_price_series = list(signal_prices.values()) + list(trade_prices_by_symbol.values())
            if down_prices is not None:
                all_price_series.append(down_prices)
            common_dates = all_price_series[0].index
            for series in all_price_series[1:]:
                common_dates = common_dates.intersection(series.index)
            common_dates = common_dates.sort_values()
            first_signal_dates = [prices.index[longest_ma - 1] for prices in signal_prices.values()]
            first_common_signal_date = max(first_signal_dates)
            dates_with_signal = common_dates[common_dates >= first_common_signal_date]
            first_held_offset = 1 + args.lag
            if len(dates_with_signal) <= first_held_offset:
                raise ValueError(
                    "not enough overlapping signal and traded-symbol history to form "
                    "the common window after applying --lag"
                )
            common_start = dates_with_signal[first_held_offset]
            if args.effective_trade_date_min is None or common_start > args.effective_trade_date_min:
                args.effective_trade_date_min = common_start
            if args.trade_date_max is not None and args.effective_trade_date_min > args.trade_date_max:
                raise ValueError("common window starts after --trade-date-max")
        if args.write_prices_file:
            write_start = time.perf_counter()
            if len(all_prices) == 1:
                output_frame = next(iter(all_prices.values())).rename("Adj Close").to_frame()
            else:
                output_frame = pd.concat(
                    {ticker: series for ticker, series in all_prices.items()}, axis=1
                )
            output_frame.to_csv(args.write_prices_file, index_label="Date")
            write_elapsed = time.perf_counter() - write_start
        else:
            write_elapsed = None
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    results_start = time.perf_counter()
    if args.read_prices_file and not args.terse:
        print(f"Read available ticker prices from {args.read_prices_file}")
    if args.write_prices_file and not args.terse:
        print(
            f"Saved adjusted closing prices for "
            f"{len(set(signal_symbols + trade_symbols + ([args.down_symbol] if args.down_symbol else [])))} "
            f"unique symbols to {args.write_prices_file}"
        )
    args._ma_length_cache = precompute_moving_averages(
        signal_prices, set(fast_group) | set(slow_group)
    )
    args._signal_event_cache = {}
    summary_rows = vectorized_summary_rows(
        args, signal_symbols, fast_group, slow_group, thresholds, trade_symbols,
        signal_prices, trade_prices_by_symbol, down_prices,
    )
    if summary_rows is None:
        summary_rows = []
        for signal_symbol, fast, slow, threshold, trade_symbol in itertools.product(
            signal_symbols, fast_group, slow_group, thresholds, trade_symbols or [None]
        ):
            report = lambda: report_system(
                signal_symbol, fast, slow, threshold, trade_symbol, args.down_symbol, args,
                signal_prices[signal_symbol],
                trade_prices_by_symbol.get(trade_symbol) if trade_symbol else None,
                down_prices,
                summary_rows,
            )
            if args.terse:
                with contextlib.redirect_stdout(io.StringIO()):
                    report()
            else:
                report()
                print()
    if args.average:
        for trade_symbol in trade_symbols:
            report = lambda: report_average_system(
                trade_symbol, args.down_symbol, args, signal_symbols, fast_group, slow_group, thresholds,
                signal_prices, trade_prices_by_symbol, down_prices, summary_rows,
            )
            if args.terse:
                with contextlib.redirect_stdout(io.StringIO()):
                    report()
            else:
                report()
                print()
    if args.summary_table:
        if args.best and args.show_full_summary:
            print_summary_table(summary_rows, args.cost)
        tested_count = (
            len(signal_symbols) * len(fast_group) * len(slow_group) * len(thresholds)
            + int(args.average)
        )
        systems_tested = {trade: tested_count for trade in trade_symbols} if args.best else None
        print_summary_table(
            summary_rows, args.cost, best=args.best, systems_tested=systems_tested
        )
    if args.deflate:
        print_deflation_report(
            summary_rows, trade_symbols, args.deflate_sims,
            args.deflate_seed, args.deflate_top,
        )
    if args.plot:
        try:
            plot_cumulative_returns(
                args, signal_symbols, fast_group, slow_group, thresholds,
                trade_symbols, args.down_symbol, signal_prices,
                trade_prices_by_symbol, down_prices,
            )
        except ImportError as exc:
            print("Error: --plot requires matplotlib to be installed", file=sys.stderr)
            return 1
    if args.time:
        results_elapsed = time.perf_counter() - results_start
        overall_elapsed = time.perf_counter() - overall_start
        print("\nElapsed time in seconds:")
        print(f"{'Data read/download':<36} {data_elapsed:>8.3f}")
        if write_elapsed is not None:
            print(f"{'Data write':<36} {write_elapsed:>8.3f}")
        print(f"{'Calculations and result display':<36} {results_elapsed:>8.3f}")
        print(f"{'Overall':<36} {overall_elapsed:>8.3f}")
    return 0
