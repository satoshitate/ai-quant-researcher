"""Three classic intraday strategies on SPY 5min (2020-2026 Alpaca data).

    1. Opening Range Breakout (ORB) — first 30min defines the range.
    2. Session VWAP reversion — fade large deviations from session VWAP.
    3. Trend continuation with vol filter — momentum only when vol is low.

Each one is fed to the same backtest engine and DSR gate so they're
directly comparable. The point isn't to find the winner — it's to show
how to plug a real strategy into the rig and read the verdict.
"""

import numpy as np
import pandas as pd

from ai_quant_lab.backtest import vectorized_backtest
from ai_quant_lab.backtest.bar_engine import BarSchedule
from ai_quant_lab.backtest.engine import BacktestConfig
from ai_quant_lab.features.library import momentum, realized_volatility
from ai_quant_lab.validation.deflated_sharpe import deflated_sharpe


# ----------------------------- session helpers -----------------------------

def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP within each trading day (resets at 9:30)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical * df["volume"]
    day = df.index.date
    cum_tpv = tp_vol.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (cum_tpv / cum_vol).rename("session_vwap")


def first_n_bars_mask(df: pd.DataFrame, n: int) -> pd.Series:
    """Boolean: True for the first n bars of each session."""
    day = df.index.date
    cumcount = pd.Series(1, index=df.index).groupby(day).cumsum()
    return cumcount <= n


# --------------------------- strategy 1: ORB ----------------------------

def opening_range_breakout(df: pd.DataFrame, opening_bars: int = 6) -> pd.Series:
    """First 30min (6 bars × 5min) defines high/low. Trade breakouts, flat at EOD.

    Long when price > opening_high, short when price < opening_low. Flat for the
    final 6 bars of the day so we don't carry overnight.
    """
    day = df.index.date
    in_opening = first_n_bars_mask(df, opening_bars)
    # Within each day, the running max of high (or min of low) during the opening,
    # then frozen after the opening. Use shift(1) so we trade at t+1's open.
    high_during_opening = df["high"].where(in_opening)
    low_during_opening = df["low"].where(in_opening)
    opening_high = high_during_opening.groupby(day).cummax().groupby(day).ffill()
    opening_low = low_during_opening.groupby(day).cummin().groupby(day).ffill()

    price = df["close"].shift(1)  # decide on last bar's close
    oh = opening_high.shift(1)
    ol = opening_low.shift(1)

    signal = pd.Series(0.0, index=df.index)
    signal[price > oh] = 1.0
    signal[price < ol] = -1.0

    # Flat at EOD: zero out last 6 bars (30 min) of each day.
    ones = pd.Series(1, index=df.index)
    bars_per_day = ones.groupby(day).transform("size")
    cumcount = ones.groupby(day).cumsum()
    near_close = cumcount > (bars_per_day - 6)
    signal[near_close] = 0.0
    # Also flat during opening (no position before range is defined).
    signal[in_opening] = 0.0
    return signal


# --------------------------- strategy 2: VWAP reversion ----------------------------

def vwap_reversion(df: pd.DataFrame, threshold_pct: float = 0.002) -> pd.Series:
    """Fade large deviations from session VWAP.

    When close is `threshold_pct` above VWAP → short. Below → long.
    Position scales linearly with deviation up to ±1.
    """
    vwap = session_vwap(df)
    dev = (df["close"] - vwap) / vwap            # signed deviation
    # Scale: at threshold, full position. Capped at ±1.
    signal = (-dev / threshold_pct).clip(-1, 1)
    return signal.shift(1).fillna(0.0)


# --------------------------- strategy 3: momentum w/ vol filter ----------------------------

def momentum_lowvol(df: pd.DataFrame, mom_window: int = 12, vol_window: int = 78) -> pd.Series:
    """20-bar momentum signal, but only trade when realized vol is below its 1-day median.

    The idea: momentum works in calm regimes, falls apart in chop. The filter
    zeros out the signal in high-vol regimes.
    """
    mom = momentum(df["close"], window=mom_window)
    vol = realized_volatility(df["close"], window=vol_window)
    vol_median = vol.rolling(78 * 5, min_periods=78).median()   # 5-day rolling median
    low_vol = (vol < vol_median).astype(float)
    raw = np.sign(mom) * low_vol
    return raw.fillna(0.0)


# ----------------------------- main driver -----------------------------

def evaluate(name: str, signal: pd.Series, df: pd.DataFrame, cfg: BacktestConfig,
             schedule: BarSchedule, n_trials: int) -> None:
    returns = df["close"].pct_change().fillna(0.0)
    # Gross (no costs) — does the signal even have raw edge?
    gross_cfg = BacktestConfig.from_schedule(schedule, cost_bps=0.0, execution_lag=cfg.execution_lag)
    gross = vectorized_backtest(signal, returns, config=gross_cfg)
    net = vectorized_backtest(signal, returns, config=cfg)
    dsr_gross = deflated_sharpe(gross.returns, n_trials=n_trials, annualization=schedule.annualization)
    dsr_net = deflated_sharpe(net.returns, n_trials=n_trials, annualization=schedule.annualization)
    print(f"\n{name}")
    print(f"  {'metric':<20} {'GROSS (0bps)':>18} {'NET (3bps)':>14}")
    print(f"  {'Sharpe (ann)':<20} {gross.metrics['sharpe_ratio']:>+18.2f} {net.metrics['sharpe_ratio']:>+14.2f}")
    print(f"  {'Ann. return':<20} {gross.metrics['annualized_return']:>+18.2%} {net.metrics['annualized_return']:>+14.2%}")
    print(f"  {'Max drawdown':<20} {gross.metrics['max_drawdown']:>+18.2%} {net.metrics['max_drawdown']:>+14.2%}")
    print(f"  {'Hit rate':<20} {gross.metrics['hit_rate']:>18.2%} {net.metrics['hit_rate']:>14.2%}")
    print(f"  {'Ann. turnover':<20} {gross.metrics['annual_turnover']:>18,.0f} {net.metrics['annual_turnover']:>14,.0f}")
    print(f"  {'DSR p-value':<20} {dsr_gross.pvalue:>18.4f} {dsr_net.pvalue:>14.4f}")
    print(f"  {'Verdict':<20} {('PASS' if dsr_gross.passes() else 'REJECT'):>18} "
          f"{('PASS' if dsr_net.passes() else 'REJECT'):>14}")


def main() -> None:
    print("Loading SPY 5min (2020-2026)...")
    df = pd.read_pickle("data/spy_5m_2020_2026.pkl")
    print(f"  {len(df):,} bars\n")

    schedule = BarSchedule.spy_5m()
    cfg = BacktestConfig.from_schedule(schedule, cost_bps=3.0, execution_lag=0)
    n_trials = 5  # honest count: we're testing 3 strategies + 2 parameter variants

    evaluate("1. Opening Range Breakout (30min)", opening_range_breakout(df), df, cfg, schedule, n_trials)
    evaluate("2. Session VWAP reversion (20bps)", vwap_reversion(df, 0.002), df, cfg, schedule, n_trials)
    evaluate("3. Momentum + low-vol filter",     momentum_lowvol(df),        df, cfg, schedule, n_trials)


if __name__ == "__main__":
    main()
