"""Print moving-average crossovers and optionally backtest them.

Examples:
  python xma_signal.py SPY 50 200
  python xma_signal.py IEF 20 200 --trade SPY
  python xma_signal.py "[IEF TLT]" 1 "[21 63 200]" --trade SPY
  python xma_signal.py "[SPY IEF TLT]" 1 10:200:10 --trade SPY --best --deflate
"""

from xma.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
