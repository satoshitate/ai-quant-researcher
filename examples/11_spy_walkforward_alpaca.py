"""SPY 5min walk-forward on 6 years of Alpaca data.

This is the first example with serious data:
    - 123k bars (2020-01 → 2026-05)
    - Walk-forward: 1 year train, 3 months test, rolling
    - Honest DSR with the trial count
"""

import pandas as pd

from ai_quant_lab.backtest import vectorized_backtest
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe
from ai_quant_lab.validation.walk_forward import walk_forward_evaluate


# Strategy: short-term mean reversion. 20-bar z-score, sized inversely.
def mean_reversion_strategy(prices: pd.Series, window: int = 20) -> pd.Series:
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std
    return (-z.clip(-2, 2) / 2.0).fillna(0.0)


def main() -> None:
    print("Loading SPY 5min...")
    df = pd.read_pickle("data/spy_5m_2020_2026.pkl")
    close = df["close"]
    print(f"  {len(close):,} bars, {close.index[0].date()} → {close.index[-1].date()}\n")

    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=1)

    # ---- Full sample backtest (in-sample, for comparison) ----
    positions = mean_reversion_strategy(close)
    returns = close.pct_change().fillna(0.0)
    is_result = vectorized_backtest(positions, returns, config=cfg)
    print(f"In-sample (full 6yr):  SR={is_result.metrics['sharpe_ratio']:+.2f}  "
          f"DD={is_result.metrics['max_drawdown']:+.2%}  "
          f"turnover={is_result.metrics['annual_turnover']:,.0f}x/yr")

    # ---- Walk-forward ----
    # 1yr train (≈ 19,656 bars), 3mo test (≈ 4,914 bars), rolling
    train_size = schedule.annualization                       # 1 year
    test_size = schedule.annualization // 4                   # 3 months
    print(f"\nWalk-forward: train={train_size:,} test={test_size:,}")

    wf = walk_forward_evaluate(
        close,
        strategy=mean_reversion_strategy,
        train_size=train_size,
        test_size=test_size,
        purge=20,
        mode="rolling",
        config=cfg,
    )
    n_folds = len(wf["folds"])
    m = wf["metrics"]
    print(f"  Folds: {n_folds}")
    print(f"  OOS Sharpe (concat): {m['sharpe_ratio']:+.2f}")
    print(f"  OOS max DD:          {m['max_drawdown']:+.2%}")
    print(f"  Fold SRs:            {[f'{s:+.2f}' for s in wf['fold_sharpes']]}")

    degradation = (1 - m['sharpe_ratio'] / is_result.metrics['sharpe_ratio']) \
        if is_result.metrics['sharpe_ratio'] != 0 else float('nan')
    print(f"  Degradation IS→OOS:  {degradation:+.0%}  "
          f"(healthy: 20-40%, >50% = overfit)")

    # ---- DSR on OOS, honestly counting parameter trials ----
    # We tried one window (20). In reality you'd sweep 5/10/20/50/100 → n_trials=5+.
    n_trials = 5
    dsr = deflated_sharpe(
        wf["concatenated_returns"],
        n_trials=n_trials,
        annualization=schedule.annualization,
    )
    print(f"\nDeflated Sharpe (n_trials={n_trials}):")
    print(f"  Observed SR:           {dsr.sharpe_ratio:+.2f}")
    print(f"  Expected max under H0: {dsr.expected_max_sharpe:+.2f}")
    print(f"  Deflated SR:           {dsr.deflated_sharpe_ratio:+.2f}")
    print(f"  p-value:               {dsr.pvalue:.4f}")
    print(f"  Passes 5% gate?        {'YES' if dsr.passes(0.05) else 'NO'}")


if __name__ == "__main__":
    main()
