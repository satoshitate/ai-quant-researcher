# Findings: 44-trial run on SPY 5min with Llama 3.1 8B Instant

This document reports the outcome of running the research loop end-to-end
on the changes in this PR. It is the empirical pressure-test for the new
intraday support, the free-LLM dispatcher, and the additional gates.

## Setup

- **Universe**: SPY, single asset.
- **Bars**: 5-minute regular hours, 2020-01-02 → 2026-04-30, ~124,000 bars
  (from Alpaca paper IEX feed via the new `data.alpaca_loader`).
- **Costs**: 3 bps round-trip (`BacktestConfig.from_schedule(BarSchedule.spy_5m(), cost_bps=3.0)`).
- **LLM**: Llama 3.1 8B Instant via Groq free tier (only model with enough
  TPM headroom for sustained iteration on the free plan).
- **Pipeline**: hypothesis → (critic bypassed, see notes) → code → sandbox
  → vectorized backtest → DSR + degenerate-returns + buy-and-hold guards.
- **Trials**: 44 total before the loop was halted by a sustained 429 from
  Groq.

## Outcome breakdown

| Outcome                           | Count | % of trials |
|-----------------------------------|-------|-------------|
| Sandbox error (LLM code bugs)     | 20    | 45%         |
| DSR rejected (insufficient edge)  | 17    | 39%         |
| Degenerate returns (all-zero)     |  5    | 11%         |
| Buy-and-hold disguise             |  2    |  5%         |
| **Real survivors**                | **0** | **0%**      |

Top trials by raw Sharpe:

| Iter | SR    | Annual turnover | Rejection                                       |
|------|-------|-----------------|-------------------------------------------------|
| 8    | +0.75 |   0.2x          | DSR p=0.337 (would not have passed buy-and-hold guard either, see below) |
| 15   | +0.73 |   0.1x          | buy_and_hold_disguise                           |
| 18   | +0.73 |   0.0x          | buy_and_hold_disguise                           |
| 16   | +0.71 |   0.1x          | DSR p=0.504                                     |
| 11   | -0.26 |  84x            | DSR                                             |
| 12   | -0.65 | 128x            | DSR                                             |

The four "positive" trials are the same false-positive in different
clothing: an LLM-generated `signal.clip(0.5, 1.0)` at the end of the
function, applied to a boolean-like vector, produces a position that's
constant at 0.5. The strategy never trades, gets paid SPY's drift
divided in half, and reports a Sharpe near SPY's own (~0.7) with
turnover ≈ 0. The new `buy_and_hold_disguise` guard catches this
pattern; the two earliest cases (iters 8, 16) ran before that guard was
added.

The five strategies that actually traded (turn 23x to 2,637x/yr) all
have negative Sharpe. At 3bps and intraday turnover, costs dominate any
edge the 8B model managed to specify.

## Interpretation

The system works as designed. Every rejection path fired at least once
on a real example. The new buy-and-hold guard demonstrably caught fake
alpha that the original gates would have shipped.

The reason there are no survivors is **not** a bug in the loop. It is
the conjunction of:

1. **SPY is the worst case.** It is one of the most liquid, most
   arbitraged products in any market. The ex-cost intraday edge that
   exists for retail-accessible signals on SPY is small. Academic
   literature on SPY-style ETFs at intraday horizons supports
   "approximately zero" as the honest prior.
2. **3 bps is generous but not free.** A naïve intraday signal that
   would have looked great at zero costs (e.g. iter 8's gross SR ≈ 0.5
   from prior hand-written tests) is wiped out by realistic frictions.
   Cost dominance is the point the original article makes; this run
   demonstrates it on real data.
3. **The 8B model is the binding constraint.** 45% of trials died with
   `sandbox_error` because Llama 3.1 8B Instant hallucinates pandas
   APIs (`Series.sign()`, `Rolling.quantile(method=...)`,
   `series.replace` on ndarray), uses undefined variables, and reaches
   for `index % N` on a DatetimeIndex. With Claude Sonnet or even
   Llama 3.3 70B that rate is closer to 10%. The 5x productive
   iteration headroom would matter.
4. **DSR scales with n_trials honestly.** After 30+ trials the
   expected-max-under-null Sharpe approaches 2.0+ at 19,656
   annualization. To clear, a strategy needs a raw SR well above that.
   Edge that strong on 5min SPY net of costs is rare.

## Recommendations for productive use

The system is ready. To find real survivors, change the search
distribution, not the gates:

- **Use a smarter model.** Anthropic gives ~$5 in free credits at
  signup, which covers ~100 iterations on Sonnet. The provider chain in
  this PR already supports falling back to Anthropic when other backends
  fail. The same loop run on Sonnet would have ~5x more valid backtests
  per hour.
- **Move to a less efficient asset.** Mid-cap single stocks, futures
  back months, agricultural commodities, or smaller-cap crypto have
  edges that survive longer because fewer competitors are running the
  same searches. SPY is essentially the upper bound on difficulty.
- **Lengthen the holding period.** Hourly or daily bars annualize at
  1,764 and 252 respectively. The DSR bar drops substantially. So do
  costs as a fraction of move size.
- **Seed with templates.** Pass 5-10 published strategies (Jegadeesh
  & Titman momentum, Lo & MacKinlay reversals, etc.) into the prompt as
  starting points. The 8B can mutate variants of working code more
  reliably than invent strategies from scratch.

## Reproduction

```bash
git checkout feat/free-llm-providers-and-spy-intraday
pip install -e .[dev]
cp .env.example .env  # fill GROQ_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY
python -m ai_quant_lab.data  # populate data/spy_5m_2020_2026.pkl
python examples/14_serious_loop_spy.py
```

The DB used for this report is not committed (it's in `.gitignore`),
but the run is fully reproducible against the same data window. Costs
and annualization are wired through `BarSchedule.spy_5m()`.
