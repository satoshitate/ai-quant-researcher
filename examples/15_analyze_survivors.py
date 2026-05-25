"""Post-loop analysis: deep-dive on each survivor.

For each accepted trial:
    1. Re-run the strategy on the full SPY 5min tape (sanity-check vs stored metrics).
    2. Walk-forward validation with 1yr train / 3mo test, rolling.
    3. Deflated Sharpe with the honest n_trials from the run.
    4. Show the LLM-generated source code.

Usage:
    python examples/15_analyze_survivors.py                       # uses memory_spy_serious.db
    python examples/15_analyze_survivors.py memory_spy.db         # custom db
"""

import sys
from pathlib import Path

import pandas as pd

from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest import vectorized_backtest
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.orchestrator.sandbox import run_strategy
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe
from ai_quant_lab.validation.walk_forward import walk_forward_evaluate


def main(db_path: str = "memory_spy_serious.db") -> None:
    print(f"Reading {db_path}...")
    with ResearchMemory(Path(db_path)) as memory:
        survivors = list(memory.survivors())
        n_trials = memory.n_trials()
    print(f"  {n_trials} total trials, {len(survivors)} survivors\n")

    if not survivors:
        print("No survivors. Try a longer loop or relax DSR (carefully).")
        return

    print("Loading SPY 5min...")
    df = pd.read_pickle("data/spy_5m_2020_2026.pkl")
    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=1)
    print(f"  {len(df):,} bars\n")

    for i, trial in enumerate(survivors):
        print("=" * 72)
        print(f"SURVIVOR {i+1}/{len(survivors)} — {trial.hypothesis_text}")
        print("=" * 72)
        print(f"\nRationale:\n  {trial.rationale}\n")

        # --- 1. Re-run on full tape ---
        sandbox = run_strategy(trial.code, df)
        returns = df["close"].pct_change().fillna(0.0)
        result = vectorized_backtest(sandbox.positions, returns, config=cfg)
        m = result.metrics
        print(f"Full-tape backtest:")
        print(f"  Sharpe:        {m['sharpe_ratio']:+.2f}")
        print(f"  Ann. return:   {m['annualized_return']:+.2%}")
        print(f"  Max DD:        {m['max_drawdown']:+.2%}")
        print(f"  Turnover/yr:   {m['annual_turnover']:,.0f}x")
        print(f"  Hit rate:      {m['hit_rate']:.2%}")

        # --- 2. Walk-forward ---
        try:
            wf = walk_forward_evaluate(
                df["close"],
                strategy=lambda close_series: _run_against_close(trial.code, df, close_series),
                train_size=schedule.annualization,        # 1yr
                test_size=schedule.annualization // 4,    # 3mo
                purge=20,
                mode="rolling",
                config=cfg,
            )
            wfm = wf["metrics"]
            print(f"\nWalk-forward (1yr train / 3mo test, {len(wf['folds'])} folds):")
            print(f"  OOS Sharpe:    {wfm['sharpe_ratio']:+.2f}")
            print(f"  OOS max DD:    {wfm['max_drawdown']:+.2%}")
            print(f"  Per-fold SRs:  {[f'{s:+.1f}' for s in wf['fold_sharpes']]}")
            degradation = 1 - wfm['sharpe_ratio'] / m['sharpe_ratio'] if m['sharpe_ratio'] != 0 else 0
            print(f"  Degradation:   {degradation:+.0%}  (healthy: 20-40%)")
        except Exception as e:
            print(f"\nWalk-forward failed: {e}")

        # --- 3. DSR honest ---
        dsr = deflated_sharpe(result.returns, n_trials=n_trials, annualization=schedule.annualization)
        print(f"\nDeflated Sharpe (n_trials={n_trials}):")
        print(f"  Observed SR:  {dsr.sharpe_ratio:+.2f}")
        print(f"  Expected max: {dsr.expected_max_sharpe:+.2f}")
        print(f"  Deflated SR:  {dsr.deflated_sharpe_ratio:+.2f}")
        print(f"  p-value:      {dsr.pvalue:.4f}  → {'PASS' if dsr.passes() else 'REJECT'}")

        # --- 4. Source code ---
        print(f"\nGenerated code:")
        print("```python")
        print(trial.code)
        print("```")
        print()


def _run_against_close(code: str, full_df: pd.DataFrame, close_series: pd.Series) -> pd.Series:
    """Helper: walk_forward passes only a close slice. Reconstruct the OHLCV slice."""
    sliced_df = full_df.loc[close_series.index]
    return run_strategy(code, sliced_df).positions


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "memory_spy_serious.db"
    main(db)
