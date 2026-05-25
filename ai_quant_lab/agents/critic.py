"""CriticAgent: adversarial review of a hypothesis BEFORE the backtest.

Most LLM agent loops are credulous — propose, run, accept. The critic exists
to kill bad ideas before they get a backtest budget. It is prompted skeptically:
"assume the hypothesis is wrong; explain why."

Different markets have different failure modes:

    - equities:  factor crowding, post-2010 regime shift, decimalization, ETF effects
    - crypto:    24/7 funding flips, exchange-level liquidity gaps, leverage cascades
    - futures:   roll yield, contango/backwardation flips, exchange limit halts
    - options:   variance risk premium (short vol always looks great), pin risk
    - fx:        carry trade unwinds, central bank surprise, weekend gaps

The right template is picked from the `market_type` argument. Default falls
back to a generic equity prompt.

A pass from the critic doesn't mean the strategy works. It means the idea is
not laughable on its face. The validation gates do the real work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_quant_lab.agents.base import AgentMessage, call_claude, extract_first_json
from ai_quant_lab.agents.hypothesis import StrategyHypothesis


MarketType = Literal["equities", "crypto", "futures", "options", "fx", "generic"]


_GENERIC_FAILURE_MODES = """
1. Implicit lookahead / forward-looking data.
2. Survivorship or selection bias in the implied universe.
3. Already-arbitraged factor exposure with no plausible reason it persists.
4. EXPLICIT parameter optimization (the hypothesis itself states the parameter was picked by sweep, OR cites an unusually precise value like "lookback of 37" without justification). Standard round numbers (20, 21, 50, 100, 252) are NOT evidence of optimization.
5. Reliance on transaction costs that wouldn't survive realistic frictions.
6. Conflation of in-sample fit with out-of-sample edge.
""".strip()


_MARKET_SPECIFIC_FAILURES: dict[str, str] = {
    "equities": """
1. Implicit lookahead / forward-looking data.
2. Survivorship bias (universe defined by what exists today).
3. Factor that was arbitraged out post-2003 (decimalization), post-2010 (ETF era), or post-2020.
4. EXPLICIT parameter optimization (the hypothesis itself states the parameter was picked by sweep, OR cites an unusually precise value like "lookback of 37" without justification). Standard round numbers (20, 21, 50, 100, 252) are NOT evidence of optimization.
5. Costs that ignore short-borrow fees, hard-to-borrow lists, locate failure.
6. Performance driven by a handful of names (Tesla, NVDA, GME etc.).
7. Edge that lives entirely inside earnings windows or option-expiry weeks.
""".strip(),
    "crypto": """
1. Implicit lookahead, especially across exchanges with different timestamps.
2. Survivorship: every dead token, exchange, or chain that's gone.
3. Funding-rate flips that turn a perpetual carry trade into a melt.
4. Single-venue liquidity that can't be hit during fast moves (Binance, FTX-style).
5. Costs that ignore funding payments and gas/withdrawal fees.
6. Leverage cascades (LUNA, 3AC) that don't appear in clean tape data.
7. Stablecoin de-pegs treated as "tradeable" when execution was halted.
""".strip(),
    "futures": """
1. Implicit lookahead in front-month vs back-month signals.
2. Roll-yield contamination: a "trend" that's just the contract calendar.
3. Contango↔backwardation regime flips (e.g. crude, VIX futures, USO).
4. Limit-up/limit-down halts where the printed price isn't the executable price.
5. Costs that ignore exchange fees, NFA, clearing.
6. Strategies that depend on overnight ranges in markets that don't trade overnight.
""".strip(),
    "options": """
1. Implicit lookahead via IV surfaces computed from same-day prints.
2. Short-volatility strategies that ALWAYS look great in calm markets and explode in tail events.
3. Pin risk and assignment risk being ignored.
4. Costs that ignore bid-ask, exercise fees, the fact that quotes are far from mid.
5. Greek-based hedging that assumes continuous rebalancing.
6. Open interest assumed available for clean exit.
""".strip(),
    "fx": """
1. Implicit lookahead, especially around weekend gaps.
2. Carry-trade-style strategies that ignore tail risk (CHF 2015, TRY 2018).
3. Central bank intervention windows treated as continuous time.
4. Liquidity falling off a cliff during NY-Asia handoff.
5. Costs that ignore swap charges, especially for longer holdings.
6. Triangular arbitrage that ignores execution latency.
""".strip(),
}


SYSTEM_PROMPT_TEMPLATE = """You are a pre-backtest structural reviewer of trading hypotheses.

You are ONE of three gates. The other two (Deflated Sharpe, correlation) are
empirical and catch most bad ideas. YOUR job is ONLY to catch ideas that are
structurally broken before they waste backtest compute. Default to PASS.

Structural failure modes you should look for:

{failure_modes}

DEFAULT VERDICT IS "pass". You should ONLY kill if you can:
  (a) quote the exact clause from the hypothesis, AND
  (b) explain why it triggers a specific numbered failure mode.

What is NOT a valid kill reason:
  ✗ "This edge is probably arbitraged" — that's what DSR decides empirically.
  ✗ "Uses a specific number like 21 or 0.2" — round-ish numbers are fine.
  ✗ "The market is efficient" — not your concern.
  ✗ "Sharpe ratio might be low" — that's what backtest decides.
  ✗ "Strategy depends on parameters" — every strategy does.

What IS a valid kill reason:
  ✓ Signal uses `close.shift(-1)` or any forward-looking computation.
  ✓ Universe defined as "current S&P 500 members" with 10-year backtest.
  ✓ Hypothesis explicitly says "we tried 50 lookbacks and picked the best."
  ✓ Costs assumed zero or unrealistically low.
  ✓ Strategy assumes data the system doesn't have (fundamentals, news, alt).

If you are uncertain, PASS. The downstream gates are strict.

Output JSON ONLY:
{{
  "verdict": "pass" | "kill",
  "reasoning": "<2-3 sentences. If kill, quote the exact clause from the hypothesis>",
  "kill_reasons": [<list of failure mode numbers triggered. Empty if pass.>]
}}
"""


def build_system_prompt(market_type: MarketType = "generic") -> str:
    """Compose the critic system prompt for a given market type."""
    if market_type == "generic":
        failures = _GENERIC_FAILURE_MODES
    else:
        failures = _MARKET_SPECIFIC_FAILURES.get(market_type, _GENERIC_FAILURE_MODES)
    return SYSTEM_PROMPT_TEMPLATE.format(failure_modes=failures)


SYSTEM_PROMPT = build_system_prompt("generic")


@dataclass(frozen=True)
class CriticVerdict:
    passes: bool
    reasoning: str
    kill_reasons: list[str]


class CriticAgent:
    """Pre-backtest adversarial review."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        market_type: MarketType = "generic",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.market_type = market_type
        self._system_prompt = build_system_prompt(market_type)

    def review(self, hypothesis: StrategyHypothesis) -> CriticVerdict:
        user_content = f"""Hypothesis: {hypothesis.title}

Rationale: {hypothesis.rationale}

Spec: {hypothesis.spec}

Author claims: works in '{hypothesis.works_in_regime}', breaks in '{hypothesis.breaks_in_regime}',
expected Sharpe in {hypothesis.expected_sharpe_range}.

Argue against this hypothesis. Output JSON only."""
        response = call_claude(
            system=self._system_prompt,
            messages=[AgentMessage(role="user", content=user_content)],
            model=self.model,
            temperature=self.temperature,
            max_tokens=512,
            json_mode=True,
        )
        payload = extract_first_json(response.text)
        verdict = str(payload.get("verdict", "kill")).lower()
        return CriticVerdict(
            passes=(verdict == "pass"),
            reasoning=str(payload.get("reasoning", "")),
            kill_reasons=[str(r) for r in payload.get("kill_reasons", [])],
        )
