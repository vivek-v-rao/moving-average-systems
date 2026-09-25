# xma_signal

`xma_signal` is a focused Python research tool for **moving-average crossover strategy searches**.

It is designed for a common quantitative-research problem:

> Search a large grid of moving-average rules exactly, identify the strongest in-sample systems, and then ask how much confidence should be placed in the winner after accounting for the fact that many correlated strategies were tested.

The program is intentionally narrower than a general-purpose backtesting framework. It works with daily price series and moving-average signals rather than arbitrary indicators, order types, event streams, or broker models. In exchange, it makes this particular research workflow simple, transparent, and fast.

Its two main goals are:

1. **Fast exact exhaustive search.** Required moving averages are computed once, then large strategy grids are evaluated in vectorized NumPy blocks. No coarse grid, random search, Bayesian optimization, or local optimizer is used: every requested parameter combination is still tested.
2. **Multiple-testing-aware evaluation.** `--deflate` studies the dependence among the tested strategies, estimates several effective numbers of trials, and reports Deflated Sharpe and related diagnostics rather than treating the best in-sample Sharpe as if it had been selected in isolation.

The program can:

- generate moving-average crossover signals for one or more signal assets;
- test thousands of fast/slow moving-average combinations in one run;
- use different assets for the signal and the traded position;
- test multiple traded assets at the same time;
- apply one or more moving-average deviation thresholds;
- model cash, a defensive/down-state asset, execution lag, and proportional transaction costs;
- compare strategies with buy-and-hold;
- report annualized return, volatility, Sharpe ratio, beta, alpha, drawdown, turnover, and exposure statistics;
- select the highest in-sample Sharpe strategy for each traded asset;
- average positions across all tested systems;
- plot cumulative returns;
- read and write reproducible price CSV files; and
- use `--deflate` to study selection bias, correlated strategy searches, and shrinkage of in-sample Sharpe ratios.

## Why this project?

A moving-average grid such as:

```bash
python xma_signal.py SPY 1:50 2:200 --trade SPY --terse --best
```

contains:

```text
50 x 199 = 9,950
```

MA1/MA2 combinations.

A naive implementation can repeatedly recompute the same rolling averages and perform one pandas backtest at a time. `xma_signal` instead computes the distinct moving averages once and evaluates strategies in blocks using NumPy arrays.

The more important issue is statistical rather than computational. If 9,950 strategies are searched and the one with the highest Sharpe ratio is reported, the winner is selected partly because of favorable sampling noise. At the same time, those 9,950 strategies are not 9,950 independent experiments: nearby moving-average lengths often generate highly correlated position and return series.

`xma_signal` is built to examine both sides of that problem:

```text
large exact search
       |
       v
best in-sample strategy
       |
       +--> How fast can the full grid be evaluated?
       |
       +--> How redundant are the tested strategies?
       |
       +--> How much should the winning Sharpe be discounted?
```

This makes the program useful as a compact research tool even though its strategy family is deliberately narrow.

## Scope

This project is best thought of as a **moving-average strategy research engine**, not as a full trading simulator.

It does model:

- adjusted daily prices;
- moving-average crossover states;
- signal-at-close / trade-next-session timing;
- optional additional execution lag;
- cash or a defensive down-state asset;
- proportional turnover costs;
- one or more signal assets;
- one or more traded assets;
- large exact parameter grids; and
- statistical diagnostics for strategy selection.

It does **not** currently model:

- intraday bars or tick data;
- limit, stop, or partial-fill orders;
- bid/ask spread dynamics;
- market impact;
- portfolio margin;
- borrow availability;
- taxes;
- corporate-action mechanics beyond what is already reflected in adjusted prices; or
- arbitrary user-defined strategy logic.

If those features are central to a project, a general event-driven or portfolio backtesting framework will usually be more appropriate.

## Requirements

The core program requires:

- Python 3
- NumPy
- pandas

For automatic historical-price downloads it also uses:

- yfinance

For `--plot` it additionally uses:

- matplotlib

A simple installation is:

```bash
python -m pip install numpy pandas yfinance matplotlib
```

`yfinance` is imported only when a requested price series must be downloaded, so it is not required when every requested symbol is supplied by `--read-prices-file`.

## Quick start

Display recent crossovers for a 50-day / 200-day SPY moving-average system:

```bash
python xma_signal.py SPY 50 200
```

Use IEF as the signal asset and trade SPY:

```bash
python xma_signal.py IEF 20 200 --trade SPY --summary-table
```

Test several slow moving-average lengths for each of three signal assets and trade SPY:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 "[20 50 100 150 200]" --trade SPY --summary-table
```

Test a grid of 20 slow moving-average lengths for three signal assets, select the best in-sample system, and run the multiple-testing diagnostics:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse --best --deflate
```

The quoted bracket notation is convenient for commands containing lists. Comma-separated forms are also accepted inside the brackets.

## Command-line structure

The general form is:

```text
python xma_signal.py SIGNALS MA1_LENGTHS MA2_LENGTHS [options]
```

`SIGNALS` is one symbol or a bracketed group of symbols. `MA1_LENGTHS` and `MA2_LENGTHS` may each be a single integer, a bracketed list, or an inclusive range.

For example:

```bash
python xma_signal.py "[SPY IEF TLT]" "[1 5 10]" 20:200:20 --trade SPY
```

runs every combination of:

- 3 signal assets;
- 3 MA1 lengths; and
- 10 MA2 lengths.

That is 90 ordinary candidate strategies for SPY before adding any threshold dimension or an `--average` system.

### Length syntax

Single value:

```text
50
```

Inclusive range:

```text
10:200:10
```

which expands to 10, 20, ..., 200.

A range without an explicit step uses a step of one:

```text
2:20
```

Bracketed list:

```text
"[5 10 20 50]"
```

The program tests the Cartesian product of the MA1 and MA2 groups. It does not require MA1 to be shorter than MA2, so choose the supplied groups accordingly if that is part of the strategy definition you want to study.

## Signal definition

For a signal price series, the program computes simple moving averages:

```text
MA1 = rolling mean over the first length
MA2 = rolling mean over the second length
```

and the moving-average deviation:

```text
MA deviation = MA1 / MA2 - 1
```

With the default threshold of zero, the backtest is in the up state when:

```text
MA1 / MA2 - 1 > 0
```

or equivalently when MA1 is above MA2.

A nonzero threshold changes the rule to:

```text
MA1 / MA2 - 1 > threshold
```

For example:

```bash
python xma_signal.py SPY 20 200 --trade SPY --madev-thresh 0.01
```

requires MA1 to exceed MA2 by more than 1% before the strategy enters its up state.

Several thresholds can be tested in one search:

```bash
python xma_signal.py IEF 20 200 --trade SPY --madev-thresh "[-0.01 0 0.01]" --summary-table
```

Threshold values are decimal returns, so `0.01` means 1%.

## Signal asset versus traded asset

The asset producing the signal does not have to be the asset being traded.

For example:

```bash
python xma_signal.py TLT 20 200 --trade SPY
```

uses the relationship between TLT's 20-day and 200-day moving averages to determine the position in SPY.

Multiple signal assets can be tested against the same traded asset:

```bash
python xma_signal.py "[SPY IEF TLT]" 20 200 --trade SPY --summary-table
```

Multiple traded assets can also be evaluated in the same run:

```bash
python xma_signal.py "[IEF TLT]" 20 200 --trade "[SPY IEF TLT]" --summary-table
```

Each signal/MA/threshold specification is tested separately for each traded asset.

## Position and execution rules

Signals are calculated using closing prices. A signal observed at a close affects holdings beginning with the **next trading session**, which prevents same-close look-ahead in the backtest.

`--lag N` adds another `N` complete trading sessions of delay. For example:

```bash
python xma_signal.py SPY 50 200 --trade SPY --lag 1
```

uses the signal two sessions after it is observed: the normal next-session delay plus one additional lag session.

### Up and down positions

In the up state, the traded-asset weight is always 1.0.

In the down state, the traded-asset weight is controlled by `--down-pos`. The default is zero:

```text
up state:   traded asset = 1.0
down state: traded asset = 0.0
```

With no `--down-symbol`, any uninvested weight is cash. Cash earns the annual rate supplied by `--cash-rate`, whose default is 0.03 (3%).

For example, this keeps a 25% SPY position in the down state and holds the other 75% in cash:

```bash
python xma_signal.py SPY 50 200 --trade SPY --down-pos 0.25
```

### Defensive/down-state asset

`--down-symbol` can replace the residual down-state cash allocation with another asset.

For example:

```bash
python xma_signal.py SPY 50 200 --trade SPY --down-symbol TLT
```

has approximately these target weights:

```text
Signal up:    100% SPY,   0% TLT
Signal down:    0% SPY, 100% TLT
```

Combining `--down-symbol` with `--down-pos` allows partial exposure to both assets in the down state:

```bash
python xma_signal.py SPY 50 200 --trade SPY --down-symbol TLT --down-pos 0.25
```

which gives a down-state target of 25% SPY and 75% TLT.

## Transaction costs

`--cost RATE` subtracts a cost proportional to absolute portfolio-weight turnover.

For example:

```bash
python xma_signal.py SPY 50 200 --trade SPY --cost 0.001
```

uses a cost of 0.1% per unit of absolute position turnover.

When a down-state asset is used, turnover in both the primary traded asset and the down-state asset contributes to the cost.

## Backtest windows

When one or more traded assets are supplied, the default is a **common comparison window**. The program finds dates shared by all relevant price series and waits until the longest requested moving average can be calculated for every signal asset. This makes comparisons across candidate systems more meaningful.

The default can be stated explicitly with:

```text
--common-window
```

Use:

```text
--individual-windows
```

to allow each strategy to use its own available history instead.

The backtest interval can also be restricted explicitly:

```bash
python xma_signal.py IEF 20 200 --trade SPY --trade-date-min 2010-01-01 --trade-date-max 2025-12-31
```

The endpoints are inclusive.

For strategy searches and especially for `--deflate`, the common-window default is generally the easier basis for interpreting differences among systems, because the candidates are evaluated over comparable observations.

## Output and performance statistics

For a backtested strategy the program reports quantities including:

- total return;
- CAGR;
- annualized volatility;
- Sharpe ratio over the configured cash rate;
- maximum drawdown;
- beta relative to buy-and-hold of the traded asset;
- annualized alpha relative to buy-and-hold;
- position changes per year;
- average traded-asset position;
- average cash weight; and
- the fraction of time the signal is up or down.

The summary table includes the principal annualized statistics for comparing many systems.

### Summary table

```bash
python xma_signal.py "[IEF TLT]" 20 200 --trade SPY --summary-table
```

### Terse output

`--terse` suppresses the detailed per-system reports and prints the summary output only:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse
```

`--terse` implies `--summary-table`.

### Large exact searches

Large `--terse` searches use an accelerated exact-search path when the default common backtest window is active. The optimization changes how the calculations are performed, not which strategies are tested. Every signal/MA1/MA2/threshold/traded-asset combination in the requested Cartesian product is still evaluated.

For example:

```bash
python xma_signal.py SPY 1:50 2:200 --trade SPY --terse --best --time
```

tests all 9,950 MA1/MA2 parameter pairs. The program does not use a coarse grid, random search, Bayesian optimization, or local optimization to skip candidates.

The accelerated path works in two stages:

1. Every distinct moving-average length required for a signal asset is computed once and cached. A search such as `1:50` by `2:200` therefore needs moving averages for lengths 1 through 200, rather than recomputing the same rolling averages thousands of times.
2. Candidate states, positions, transaction costs, and returns are evaluated in NumPy blocks. Summary statistics for all strategies in a block are calculated vectorially.

The block size can be changed with `--search-block-size`:

```bash
python xma_signal.py SPY 1:50 2:200 --trade SPY --terse --best --search-block-size 512
```

The default is 512. Larger blocks can reduce Python overhead but require more memory because several arrays of approximately `block size x number of dates` are held at once. The default is intended to be a conservative speed/memory compromise.

The vectorized path is used for terse common-window searches. Detailed per-system output and `--individual-windows` retain the scalar backtest path because those modes can require system-specific histories and reporting. Both paths use the same precomputed moving-average cache.

### Illustrative performance

The acceleration is intended to preserve the exact exhaustive result while reducing repeated work.

During development testing on a daily price dataset with roughly 4,500 observations:

| Search | Candidate systems | Scalar path | Accelerated path | Approx. speedup |
| --- | ---: | ---: | ---: | ---: |
| Dense MA grid | 990 | 13.6 s | 2.7 s | 5.0x |
| `1:50 x 2:200` | 9,950 | not separately timed | about 5 s | — |

These are illustrative timings from the development environment, not universal performance claims. Runtime depends on CPU, Python/NumPy versions, number of dates, transaction-cost settings, number of signal and traded assets, threshold count, and block size.

The important invariant is **exactness**: the accelerated path evaluates the same requested strategy set and was regression-checked against the scalar implementation for ordinary searches, multiple assets, thresholds, transaction costs, execution lag, `--down-symbol`, `--average`, and `--deflate`.

You can benchmark your own machine with:

```bash
python xma_signal.py SPY 1:50 2:200 --trade SPY --terse --best --time
```

To explore the speed/memory tradeoff:

```bash
python xma_signal.py SPY 1:50 2:200 --trade SPY --terse --best --time --search-block-size 1024
```

### Best system

`--best` selects the highest in-sample Sharpe candidate for each traded asset:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade "[SPY TLT]" --terse --best
```

`--best` also implies `--summary-table`.

The final best-system table shows the selected strategy only when its annualized return exceeds the corresponding buy-and-hold return; otherwise it shows the buy-and-hold row. The selection itself is still based on in-sample Sharpe ratio.

Because a highest-Sharpe rule is selected *after* searching many candidates, its in-sample Sharpe is subject to selection bias. `--deflate` is intended to help quantify that problem.

### Annual results

Use `--annual` to print calendar-year results for every system:

```bash
python xma_signal.py SPY 50 200 --trade SPY --annual
```

### Timing

Use `--time` to report elapsed time for data access and calculations:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse --time
```

## Averaging systems

`--average` adds a strategy whose exposure is the equal average of the positions generated by all ordinary candidate systems in the run.

For example:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 "[50 100 150 200]" --trade SPY --average --summary-table
```

If 12 ordinary systems are being tested, the average system's SPY exposure on a date is the mean of those 12 systems' SPY target positions on that date.

This is different from choosing the best system. It treats the tested rules as an ensemble and diversifies across their signals.

When `--average` and `--deflate` are both requested, the average system is also included among the candidate return streams used by the deflation analysis.

## Multiple testing and `--deflate`

> **Worked example:** For a line-by-line interpretation of a real 597-strategy SPY/IEF/TLT search, including `EffN-B`, `EffN-G`, `EffN-max`, `SR0`, `DSR`, `Max-p`, `Haircut`, `EB SR`, and the redundancy decomposition, see [Interpreting `--deflate` results](INTERPRETING_DEFLATION.md).

A strategy search creates a statistical selection problem. Suppose a researcher tests many systems and reports the one with the highest Sharpe ratio. Even if all systems have modest or zero true performance, random sampling variation makes the maximum observed Sharpe larger as more strategies are tried.

Simply counting every candidate as an independent experiment is often too conservative for a moving-average search. Nearby MA lengths are highly related, and signals generated from correlated assets can also be similar. `--deflate` therefore measures dependence directly from the realized daily return streams of the tested strategies.

Run:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse --best --deflate
```

`--deflate` implies `--summary-table`.

The analysis aligns the candidate strategy returns over a common set of dates and constructs their empirical return-correlation matrix. It then reports several diagnostics rather than pretending that one definition of an effective number of trials is uniquely correct.

### Raw

`Raw` is the literal number of candidate return streams included in the analysis.

For three signal assets and 20 MA2 lengths with one MA1 length and one threshold:

```text
3 x 1 x 20 x 1 = 60 strategies
```

### MeanCorr

`MeanCorr` is the mean pairwise correlation among candidate daily strategy return streams over their common observations.

A large value indicates substantial redundancy in the search.

### EffN-B

`EffN-B` is an average-correlation approximation to the implied number of independent trials:

```text
EffN-B = rho + (1 - rho) * M
```

where `M` is the raw number of candidates and `rho` is their mean pairwise correlation.

It has the useful limiting behavior that perfectly correlated candidates act like approximately one trial, while uncorrelated candidates act like approximately `M` trials.

This is the effective count used by the program's standard DSR search hurdle.

### EffN-G

`EffN-G` is Galwey's eigenvalue-based effective number of correlated tests. If the strategy correlation matrix has eigenvalues `lambda_i`, the implementation computes:

```text
EffN-G = (sum(sqrt(lambda_i)))^2 / sum(lambda_i)
```

It measures the effective dimensionality of the family using the full eigenvalue spectrum rather than only the average correlation.

### EffN-max

`EffN-max` is designed specifically around the fact that the search selects a maximum.

The program:

1. estimates the complete strategy-return correlation matrix;
2. simulates correlated standard-normal test statistics with that matrix;
3. computes the expected maximum simulated statistic; and
4. finds the number of independent Gaussian trials that would have the same expected maximum.

The result is an "equivalent independent trial count" for maximum-selection behavior.

Simulation controls are:

```text
--deflate-sims N
--deflate-seed N
```

The defaults are 20,000 simulations and seed 12345. Increasing the simulation count reduces Monte Carlo noise at the cost of additional computation.

For example:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse --deflate --deflate-sims 100000 --deflate-seed 12345
```

### SR0 and Haircut

The tested Sharpe ratios have a cross-sectional standard deviation. Searching more effectively independent trials raises the Sharpe ratio that can occur merely because the researcher selected the best result.

`SR0` is the program's DSR-style search hurdle. It combines:

- the dispersion of Sharpe ratios across the tested family; and
- the expected maximum associated with `EffN-B`.

`Haircut` is simply:

```text
observed Sharpe - SR0
```

It is a useful way to show how much of the observed Sharpe remains after subtracting the search hurdle, but it should **not** be interpreted as a forecast of future Sharpe.

### PSR

For each displayed candidate, `PSR` is a probabilistic Sharpe diagnostic relative to a benchmark Sharpe of zero. Its sampling calculation uses the candidate's number of observations, skewness, and kurtosis rather than assuming that the sample Sharpe has the simplest Gaussian-return variance.

### DSR

`DSR` applies the same Sharpe sampling framework but compares the observed Sharpe with `SR0` rather than with zero.

Conceptually, it asks whether a candidate's Sharpe remains unusually high after accounting for the fact that it emerged from a search across multiple correlated strategies.

A high DSR is evidence against the hypothesis that selection alone explains the observed Sharpe, but it is not an out-of-sample performance forecast.

### Max-p

`Max-p` is a correlation-aware familywise p-value based on the simulated maximum test statistic. It estimates how often the maximum statistic across the entire correlated family would equal or exceed the candidate's observed Sharpe test statistic under the Gaussian max-test approximation.

Unlike a simple Bonferroni adjustment, it uses the estimated dependence structure among the strategies.

### EB SR

`EB SR` is an empirical-Bayes-style shrinkage estimate of the candidate Sharpe.

The program uses the whole cross-section of tested strategies to estimate:

- the family mean Sharpe;
- Sharpe estimation error for each candidate;
- covariance of estimation errors using the strategy-return correlation matrix; and
- the amount of cross-strategy Sharpe dispersion that remains after estimated sampling noise is removed.

Candidate Sharpes are then shrunk toward the family mean. Extreme in-sample Sharpes receive more shrinkage when the estimated signal-to-noise ratio across strategies is low.

Of the reported statistics, `EB SR` is closest to an attempt to adjust the expected performance of **each** tested strategy using information from the entire strategy family. It is still an in-sample model-based estimate and should not be treated as a replacement for genuine out-of-sample evidence.

### Redundancy decomposition

When the search contains enough variation, `--deflate` also reports a Galwey effective-count decomposition.

For example, with:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --terse --deflate
```

it can separately summarize:

- redundancy among MA/threshold variants while holding the signal asset fixed; and
- redundancy among SPY, IEF, and TLT signals while holding the MA/threshold specification fixed.

This helps distinguish two different reasons why the raw number of tested strategies can overstate the effective search size.

## Interpreting a strategy search

The deflation statistics answer different questions:

| Statistic | Main question |
| --- | --- |
| Raw | How many strategies were literally evaluated? |
| MeanCorr | How similar are their daily return streams on average? |
| EffN-B | What independent-trial count is implied by average correlation? |
| EffN-G | What is the effective dimensionality of the correlation matrix? |
| EffN-max | How many independent trials give a similar expected maximum? |
| PSR | How strong is this Sharpe relative to a zero-Sharpe benchmark? |
| DSR | How strong is this Sharpe after a multiple-search hurdle? |
| Max-p | How surprising is this candidate after accounting for the maximum over the correlated family? |
| Haircut | How far is observed Sharpe above the DSR search hurdle? |
| EB SR | How far should this candidate's Sharpe be shrunk toward the family mean under the empirical-Bayes model? |

No single statistic makes data mining disappear. Results can still be optimistic because of choices made before the program was run: which assets to test, which date range to use, which strategy family to study, which cost assumptions to choose, and which variants were discarded during earlier research.

For that reason, the strongest validation remains performance on genuinely untouched data or a carefully designed rolling/out-of-sample procedure.

## Price data

### Automatic download

If no price file supplies a requested ticker, the program downloads adjusted daily closing prices using yfinance.

For example:

```bash
python xma_signal.py IEF 50 200 --trade SPY --summary-table
```

will obtain the necessary series automatically unless they are already available through a supplied price file.

### Reading a CSV

Use:

```text
--read-prices-file PATH
```

The expected format is a `Date` column plus one adjusted-close column per ticker, for example:

```csv
Date,SPY,IEF,TLT
2024-01-02,472.65,93.41,98.12
2024-01-03,468.79,94.02,99.35
```

Then run, for example:

```bash
python xma_signal.py "[IEF TLT]" 20 200 --trade SPY --read-prices-file prices.csv --summary-table
```

If a requested symbol is missing from the CSV, the program attempts to download that symbol with yfinance.

For legacy single-symbol files, an `Adj Close` column is accepted for the single signal symbol.

### Writing a CSV

Use:

```text
--write-prices-file PATH
```

to save the adjusted closes loaded for the run:

```bash
python xma_signal.py "[SPY IEF TLT]" 50 200 --trade SPY --write-prices-file prices.csv --summary-table
```

This is useful for making subsequent runs reproducible and avoiding repeated downloads.

## Plotting

Use `--plot` to display cumulative growth-of-$1 curves on a logarithmic vertical scale:

```bash
python xma_signal.py "[IEF TLT]" 50 200 --trade SPY --plot
```

The chart includes buy-and-hold plus the tested strategies. If `--average` is present, the average system is included as well.

`matplotlib` is required only for plotting.

## Important command-line options

| Option | Meaning |
| --- | --- |
| `--trade SYMBOL [...]` | Asset or assets traded when a signal is up |
| `--down-pos VALUE` | Primary traded-asset weight when the signal is down; default 0 |
| `--down-symbol SYMBOL` | Asset receiving weight `1 - down_pos` in the down state |
| `--cash-rate RATE` | Annual cash rate as a decimal; default 0.03 |
| `--cost RATE` | Cost per unit of absolute position turnover |
| `--lag N` | Additional full trading sessions of execution delay |
| `--madev-thresh VALUE [...]` | One or more MA1/MA2 deviation thresholds |
| `--trade-date-min DATE` | Inclusive first backtest date |
| `--trade-date-max DATE` | Inclusive last backtest date |
| `--individual-windows` | Let systems use their own available histories |
| `--summary-table` | Print compact comparison table |
| `--best` | Select highest in-sample Sharpe candidate per traded asset |
| `--terse` | Suppress detailed per-system output |
| `--annual` | Print calendar-year performance |
| `--average` | Add equal-average exposure across all tested systems |
| `--deflate` | Analyze correlated multiple testing and Sharpe selection bias |
| `--deflate-sims N` | Number of Gaussian max-test simulations; default 20,000 |
| `--deflate-seed N` | Random seed for `--deflate`; default 12345 |
| `--deflate-top N` | Number of top-Sharpe candidates displayed by `--deflate`; 0 means all |
| `--plot` | Plot cumulative returns |
| `--read-prices-file PATH` | Read adjusted closes from CSV |
| `--write-prices-file PATH` | Save loaded adjusted closes to CSV |
| `--time` | Print elapsed-time breakdown |
| `--search-block-size N` | Strategies per NumPy block in accelerated terse searches; default 512 |
| `--days N` | Number of recent sessions for crossover display; `<= 0` suppresses it |

Run:

```bash
python xma_signal.py --help
```

for the authoritative option list for the installed version.

## Project structure

The executable is intentionally small. Most functionality is divided into modules by responsibility:

```text
xma_signal.py
xma/
    __init__.py
    backtest.py
    cli.py
    data.py
    deflation.py
    fast_search.py
    plotting.py
    reporting.py
    signals.py
```

### `xma_signal.py`

Thin command-line entry point. It imports `main` from `xma.cli`.

### `xma/cli.py`

Defines argument parsing, list/range parsing, input validation, price-series orchestration, common-window setup, and top-level search orchestration. It selects the exact block-vectorized path for large terse common-window searches and otherwise uses the scalar backtest path.

### `xma/data.py`

Handles CSV price files and adjusted-close downloads. The yfinance dependency is lazy so file-only usage does not require yfinance.

### `xma/signals.py`

Contains moving-average crossover-date logic and precomputes the requested moving-average lengths once per signal asset for reuse by both scalar and vectorized searches.

### `xma/backtest.py`

Builds strategy positions and returns, applies execution lag and transaction costs, computes benchmark and regime statistics, and builds the result records consumed by the reporting and deflation code. It also implements the equal-average system.

### `xma/deflation.py`

Implements the effective-trial calculations, Gaussian maximum simulation, probabilistic/deflated Sharpe calculations, correlation diagnostics, and empirical-Bayes Sharpe shrinkage.

### `xma/fast_search.py`

Implements exact block-vectorized backtesting for terse common-window searches. It consumes the precomputed moving averages, constructs many candidate signal states at once, evaluates returns and turnover in NumPy arrays, and emits the same summary-row format used by the scalar backtester.

### `xma/reporting.py`

Formats detailed strategy reports, summary tables, annual results, and the `--deflate` diagnostics.

### `xma/plotting.py`

Constructs cumulative-return plots with matplotlib.

## Reproducible research example

A useful workflow is to first save the relevant adjusted closes:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --write-prices-file prices.csv --terse
```

Then run the strategy search from the frozen data file:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --read-prices-file prices.csv --terse --best --deflate --time
```

For a more stable Monte Carlo max-test estimate:

```bash
python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --read-prices-file prices.csv --terse --best --deflate --deflate-sims 100000 --deflate-seed 12345 --time
```

Saving the price data and the random seed makes it much easier to reproduce a reported result later.

## When this tool is a good fit

`xma_signal` is especially well suited to questions such as:

- Which fast/slow moving-average pair had the highest in-sample Sharpe for SPY?
- Does an IEF or TLT moving-average signal improve a strategy that trades SPY?
- How stable is performance across neighboring moving-average lengths?
- How many effectively independent strategies were tested in a dense MA grid?
- How much of the winning Sharpe may be attributable to searching many correlated systems?
- How does an equal-weight ensemble of tested signals compare with choosing the single best one?

The narrow scope is intentional. A researcher can inspect the full path from prices to moving averages to positions to returns to selection-bias diagnostics without a large framework between the strategy definition and the reported result.

## Research cautions

This program is a research and backtesting tool. In particular:

- `--best` is explicitly an in-sample selection procedure;
- correlated strategies reduce but do not eliminate multiple-testing bias;
- `--deflate` only knows about candidates actually supplied to the run, not strategies tried in earlier research;
- transaction costs are represented by a simple proportional-turnover model;
- the backtest does not model taxes, market impact, bid/ask dynamics, borrow constraints, or other implementation details;
- historical relationships among SPY, IEF, TLT, or any other assets need not persist; and
- DSR, Max-p, Haircut, and EB SR are diagnostics, not guarantees of future performance.

Use genuinely untouched data, rolling validation, or other out-of-sample methods when evaluating whether a strategy has persistent predictive value.
