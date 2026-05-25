"""Research loop on SPY 5min using Groq (Llama 3.3 70B), free tier.

Reads the .env (ALPACA + GROQ keys), loads 6 years of SPY 5-min closes,
and runs the full pipeline:
    HypothesisAgent → CriticAgent → CodeAgent → Sandbox → Backtest → Gates
on Groq instead of Anthropic. Each LLM call is ~0.5s on Groq vs ~2-3s on Claude
Sonnet, so a 10-iter loop wraps up in well under a minute.

Run with:
    set -a && source .env && set +a
    python examples/13_research_loop_spy_groq.py
"""

from pathlib import Path

import pandas as pd

from ai_quant_lab.agents.memory import ResearchMemory
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.orchestrator.loop import LoopConfig, run_research_loop


def main() -> None:
    print("Loading SPY 5min (2020-2026)...")
    df = pd.read_pickle("data/spy_5m_2020_2026.pkl")
    close = df["close"]
    print(f"  {len(close):,} bars, {close.index[0]} → {close.index[-1]}\n")

    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=1)

    loop_cfg = LoopConfig(
        market_description=(
            "5-minute bars on SPY (S&P 500 ETF). Regular hours only (9:30-16:00 ET). "
            "6 years of data, 2020-2026, ~124k bars. Highly liquid, tight spreads. "
            "Cost ~3 bps round-trip. Looking for intraday alpha that survives realistic "
            "frictions and the deflated Sharpe gate."
        ),
        market_type="equities",
        iterations=5,            # keep it short for first run
        target_survivors=2,
        max_llm_calls=30,
        backtest_config=cfg,
        annualization=schedule.annualization,
    )

    db_path = Path("memory_spy.db")
    print(f"Memory db: {db_path.resolve()}")
    print(f"Annualization: {schedule.annualization:,}  Cost: {cfg.cost_bps} bps")
    print(f"Iterations: {loop_cfg.iterations}  Target survivors: {loop_cfg.target_survivors}\n")

    with ResearchMemory(db_path) as memory:
        artifacts, survivors = run_research_loop(close, loop_cfg, memory=memory)

    print()
    print("=" * 70)
    print(f"Total iterations:  {len(artifacts)}")
    print(f"Accepted:          {sum(a.accepted for a in artifacts)}")
    print(f"Survivors in db:   {len(survivors)}")
    print()
    for art in artifacts:
        status = "ACCEPT" if art.accepted else f"REJECT ({art.rejection_reason})"
        print(f"  [{art.iteration:02d}] SR={art.sharpe:+.2f}  {status}  {art.title[:60]}")


if __name__ == "__main__":
    main()
