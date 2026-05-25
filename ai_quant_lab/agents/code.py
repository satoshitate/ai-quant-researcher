"""CodeAgent: turn a StrategyHypothesis into a runnable Python function.

The output is a single function `strategy(price_data: pd.Series) -> pd.Series`
that returns target positions. The function must:
    - use only `numpy`, `pandas`, and `ai_quant_lab.features.library`
    - never reference future bars (the leakage detector will catch this)
    - return a Series indexed like the input

The sandbox (orchestrator/sandbox.py) is what actually executes the code.
This agent only produces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ai_quant_lab.agents.base import AgentMessage, call_claude
from ai_quant_lab.agents.hypothesis import StrategyHypothesis


SYSTEM_PROMPT_SINGLE = """You are a Python developer translating quantitative hypotheses into code.

Constraints (non-negotiable):
1. Output ONE function named `strategy(price_data: pd.Series) -> pd.Series`.
2. Imports allowed: numpy as np, pandas as pd, and ai_quant_lab.features.library
   (momentum, rolling_zscore, realized_volatility, range_pct, ewma).
3. NEVER look at future bars. Use .shift(1) or .rolling(...).<aggregate>().
4. Return positions in [-1, 1]. Use .clip(-1, 1) at the end.
5. NaN positions are fine; the engine treats them as 0.
6. Output ONLY the code in a ```python block. No prose before or after.

If the hypothesis is ambiguous, make sensible defaults — do not ask questions.
"""


SYSTEM_PROMPT_INTRADAY_OHLCV = """You are a Python developer translating intraday hypotheses into code.

You receive a pandas DataFrame `df` with these columns: open, high, low, close, volume.
The index is a tz-aware DatetimeIndex (US/Eastern), 5-minute bars, regular hours
9:30-16:00 ET. Multiple days are concatenated end-to-end.

Constraints (non-negotiable):
1. Output ONE function: `strategy(price_data: pd.DataFrame) -> pd.Series`.
   Return a Series indexed by `price_data.index`, values in [-1, 1], representing
   the target position at the close of each bar (will be lagged by the engine).
2. Imports allowed: numpy as np, pandas as pd, and
   `from ai_quant_lab.features.library import ...`. EXACT SIGNATURES (do not
   invent kwargs — only these args exist):

      momentum(price_data: pd.Series, window: int = 21) -> pd.Series
      rolling_zscore(price_data: pd.Series, window: int = 21) -> pd.Series
      realized_volatility(price_data: pd.Series, window: int = 21) -> pd.Series
      range_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series
      parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series
      garman_klass_volatility(open_price, high, low, close, window: int = 21) -> pd.Series
      vwap_deviation(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series
      ewma(price_data: pd.Series, halflife: float) -> pd.Series

   NO other kwargs exist. Do NOT pass `freq=`, `method=`, `min_periods=` to
   these functions. If you need behavior outside this surface, use raw pandas.
3. NEVER look at future bars. Compute features via .shift(1), .rolling(...),
   or .ewm(...). NEVER use .shift(-1) or center=True.
4. ALWAYS clip the final signal to [-1, 1] with `.clip(-1, 1)`.
5. Output ONLY the code in a ```python block. No prose.

PITFALLS to avoid (these have crashed prior attempts):
- DO NOT do `df.index % N` — datetime index doesn't support modulo.
  For "every Nth bar", use `np.arange(len(df)) % N == 0`.
- DO NOT return `np.array(...)` — must be `pd.Series(values, index=df.index)`.
- DO NOT do `pd.Series(some_list).rolling(N)` if `some_list` was built from a
  groupby-apply that returned mixed types. Convert to a flat float Series first.
- DO NOT compute features per-day with `df.groupby(df.index.date).apply(...)` —
  that re-introduces tz issues. Use rolling windows on the whole series; if you
  need session resets, use `df.index.normalize()` or `df.index.time` masks.
- Position thresholds should produce reasonable turnover. A signal that flips
  every bar will be killed by costs. Add hysteresis or hold periods.

GOOD EXAMPLE:
```python
import pandas as pd
import numpy as np
from ai_quant_lab.features.library import rolling_zscore

def strategy(price_data: pd.DataFrame) -> pd.Series:
    close = price_data["close"]
    z = rolling_zscore(close, window=20)
    # Mean reversion: fade extremes, hold while |z| > 1
    raw = (-z / 2.0).clip(-1, 1)
    # Hysteresis: only enter when |z| > 1.5, exit when |z| < 0.5
    entry = (z.abs() > 1.5).astype(float)
    holding = entry.replace(0, np.nan).ffill().fillna(0)
    exit_mask = (z.abs() < 0.5).astype(float)
    active = holding * (1 - exit_mask.cummax().diff().fillna(0))
    signal = (raw * active).clip(-1, 1)
    return signal.fillna(0.0)
```

If the hypothesis is ambiguous, make sensible defaults — do not ask questions.
"""


SYSTEM_PROMPT_CROSS_SECTIONAL = """You are a Python developer translating quantitative hypotheses into code.

The strategy is CROSS-SECTIONAL: it ranks/scores assets at each bar and goes
long the best, short the worst (or whatever the hypothesis specifies).

Constraints (non-negotiable):
1. Output ONE function: `strategy(price_data: pd.DataFrame) -> pd.DataFrame`.
   `price_data` has time on the index, asset id on columns. Return a DataFrame
   of the SAME shape, where each cell is the target weight for that asset at
   that time.
2. Imports allowed: numpy as np, pandas as pd,
   ai_quant_lab.features.library, ai_quant_lab.features.cross_sectional
   (rank_within_universe, zscore_cross_section, neutralize_by_factor,
    industry_neutralize, cross_sectional_momentum).
3. NEVER look at future bars. Always .shift(1) before computing signals.
4. The portfolio should be roughly dollar-neutral: sum(weights per row) ≈ 0.
   Use long_short_quantile_portfolio shape: long top quantile, short bottom.
5. Weights are unbounded per asset but the engine clips to [-1, 1] per cell.
   Typical magnitudes are 1/N where N is universe size.
6. Output ONLY the code in a ```python block.
"""


# Default to single-asset for backward compatibility.
SYSTEM_PROMPT = SYSTEM_PROMPT_SINGLE


@dataclass(frozen=True)
class CodeArtifact:
    source: str  # full function source


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```")


class CodeAgent:
    """Renders a strategy hypothesis into a runnable function.

    Switches between single-asset and cross-sectional system prompts based on
    the `mode` argument. Cross-sectional mode is used when the universe is
    a basket (e.g. equities) and dollar-neutral long-short is the goal.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        mode: str = "single",  # 'single' | 'cross_sectional' | 'intraday_ohlcv'
    ) -> None:
        if mode not in {"single", "cross_sectional", "intraday_ohlcv"}:
            raise ValueError("mode must be 'single', 'cross_sectional', or 'intraday_ohlcv'")
        self.model = model
        self.temperature = temperature
        self.mode = mode
        self._system_prompt = {
            "single": SYSTEM_PROMPT_SINGLE,
            "cross_sectional": SYSTEM_PROMPT_CROSS_SECTIONAL,
            "intraday_ohlcv": SYSTEM_PROMPT_INTRADAY_OHLCV,
        }[mode]

    def render(self, hypothesis: StrategyHypothesis) -> CodeArtifact:
        user_content = f"""Hypothesis: {hypothesis.title}

Rationale: {hypothesis.rationale}

Spec:
{_format_spec(hypothesis.spec)}

Write the strategy function."""
        response = call_claude(
            system=self._system_prompt,
            messages=[AgentMessage(role="user", content=user_content)],
            model=self.model,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return CodeArtifact(source=_extract_code(response.text))


def _format_spec(spec: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in spec.items())


def _extract_code(text: str) -> str:
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    # Fallback: assume the whole response is code if no fence.
    return text.strip()
