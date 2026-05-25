"""SPY intraday: real data + BarSchedule + backtest + Deflated Sharpe.

Pipeline:
    1. Download SPY 5-min bars from Yahoo (last 60 days, no API key needed).
    2. Build a simple mean-reversion intraday signal (5-bar z-score).
    3. Backtest with BarSchedule.spy_5m() so annualization is honest.
    4. Run the Deflated Sharpe gate assuming N=20 trials (be honest about it).
"""

import numpy as np
import pandas as pd

from ai_quant_lab.backtest import vectorized_backtest
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.data import load_yfinance
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe


def main() -> None:
    print("=== SPY 5-min intraday (yfinance, last 60d) ===\n")

    df = load_yfinance("SPY", interval="5m", period="60d", regular_hours_only=True)
    print(f"Bars: {len(df):,}  ({df.index[0]} → {df.index[-1]})")

    close = df["close"]
    returns = close.pct_change().fillna(0.0)

    # Simple intraday mean reversion: short when 5-bar z-score is high, long when low.
    window = 5
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std
    signal = -z.clip(-3, 3) / 3.0   # in [-1, 1]
    positions = signal.shift(1).fillna(0.0)   # explicit lag for clarity

    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=0)
    print(f"Annualization: {schedule.annualization:,} (5m bars × 252 days)")

    result = vectorized_backtest(positions, returns, config=cfg)
    m = result.metrics
    print(
        f"\nBacktest:"
        f"\n  Sharpe (annualized)  : {m['sharpe_ratio']:+.2f}"
        f"\n  Annualized return    : {m['annualized_return']:+.2%}"
        f"\n  Max drawdown         : {m['max_drawdown']:+.2%}"
        f"\n  Hit rate             : {m['hit_rate']:.2%}"
        f"\n  Annual turnover      : {m['annual_turnover']:,.0f}x"
    )

    n_trials = 20
    dsr = deflated_sharpe(result.returns, n_trials=n_trials, annualization=schedule.annualization)
    print(
        f"\nDeflated Sharpe gate (assuming you tested {n_trials} variants):"
        f"\n  Observed SR          : {dsr.sharpe_ratio:+.2f}"
        f"\n  Expected-max-under-null: {dsr.expected_max_sharpe:+.2f}"
        f"\n  Deflated SR          : {dsr.deflated_sharpe_ratio:+.2f}"
        f"\n  p-value              : {dsr.pvalue:.4f}"
        f"\n  Passes 5% gate?      : {'YES' if dsr.passes(0.05) else 'NO'}"
    )


if __name__ == "__main__":
    main()
