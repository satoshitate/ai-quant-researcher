"""Serious research loop on SPY 5min: 30 iterations, OHLCV mode, Groq.

This is the "find something that works" run. Differences from example 13:
    - Passes full OHLCV DataFrame instead of just close.
    - Uses the new `intraday_ohlcv` CodeAgent mode with anti-pitfall prompt.
    - 30 iterations, target 3 survivors.
    - At the end, prints a survivor report with all metrics.
"""

import json
import time
from pathlib import Path

import pandas as pd

from ai_quant_lab.agents import CodeAgent, CriticAgent, HypothesisAgent, ResearchMemory
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop


def main() -> None:
    print("Loading SPY 5min (2020-2026)...")
    df = pd.read_pickle("data/spy_5m_2020_2026.pkl")
    print(f"  {len(df):,} bars\n")

    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=1)

    loop_cfg = LoopConfig(
        market_description=(
            "5-minute OHLCV bars on SPY (S&P 500 ETF). Regular hours only "
            "(9:30-16:00 ET). 6 years of data, 2020-01 to 2026-04, ~124k bars. "
            "Highly liquid, tight spreads (~1bp). Cost assumed 3 bps round-trip. "
            "You have access to open, high, low, close, volume. Looking for "
            "intraday alpha (mean reversion, momentum, vol carry, opening range, "
            "VWAP reversion) that survives realistic frictions and the deflated "
            "Sharpe gate. Strategies should have reasonable turnover (under 500x/yr)."
        ),
        market_type="equities",
        iterations=30,
        target_survivors=3,
        max_llm_calls=120,
        backtest_config=cfg,
        annualization=schedule.annualization,
    )

    # Model split for Groq free-tier budgeting:
    #   - hypothesis: 70B (needs creativity + grounded specs)
    #   - critic:     70B (needs nuanced judgment — 8B is too literal)
    #   - code:       8B Instant (mechanical — high TPM, fast)
    SMART = "llama-3.3-70b-versatile"
    FAST = "llama-3.1-8b-instant"
    hypothesis_agent = HypothesisAgent(model=SMART)
    critic_agent = CriticAgent(model=SMART, market_type="equities")
    code_agent = CodeAgent(model=FAST, mode="intraday_ohlcv")

    db_path = Path("memory_spy_serious.db")
    print(f"Memory db: {db_path.resolve()}")
    print(f"Annualization: {schedule.annualization:,}  Cost: {cfg.cost_bps} bps")
    print(f"Iterations: {loop_cfg.iterations}  Target survivors: {loop_cfg.target_survivors}\n")

    t0 = time.time()
    with ResearchMemory(db_path) as memory:
        artifacts, survivors = run_research_loop(
            df, loop_cfg, memory=memory,
            hypothesis_agent=hypothesis_agent,
            critic_agent=critic_agent,
            code_agent=code_agent,
        )

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print(f"Done in {elapsed:.1f}s ({elapsed/max(len(artifacts),1):.1f}s per iteration)")
    print(f"Iterations: {len(artifacts)}  Accepted: {sum(a.accepted for a in artifacts)}")
    print(f"Sandbox errors: {sum(a.rejection_reason == 'sandbox_error' for a in artifacts)}")
    print(f"Critic kills:   {sum(a.rejection_reason == 'critic' for a in artifacts)}")
    print(f"DSR kills:      {sum((a.rejection_reason or '').startswith('deflated_sharpe') for a in artifacts)}")
    print()
    print("All iterations:")
    for art in artifacts:
        status = "ACCEPT" if art.accepted else f"REJECT ({(art.rejection_reason or '')[:40]})"
        print(f"  [{art.iteration:02d}] SR={art.sharpe:+6.2f}  {status:<55}  {art.title[:50]}")

    print()
    print("=" * 72)
    print(f"SURVIVORS ({len(survivors)}):")
    print("=" * 72)
    for trial in survivors:
        m = trial.metrics
        print(f"\n  [{trial.hypothesis_id}] {trial.hypothesis_text}")
        print(f"    Sharpe:    {m.get('sharpe_ratio', 0):+.2f}")
        print(f"    Return:    {m.get('annualized_return', 0):+.2%}")
        print(f"    Max DD:    {m.get('max_drawdown', 0):+.2%}")
        print(f"    Turnover:  {m.get('annual_turnover', 0):,.0f}x/yr")
        print(f"    Hit rate:  {m.get('hit_rate', 0):.2%}")
        print(f"    Rationale: {trial.rationale[:150]}")


if __name__ == "__main__":
    main()
