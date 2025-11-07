#!/usr/bin/env python3
"""
Generate synthetic SPY options minute "bars" from underlying minute prices.

- Loads SPY minute parquet (your existing underlying cache).
- Estimates per-minute annualized IV via EWMA of log returns.
- Prices calls & puts with Black-Scholes for a grid of DTEs & moneyness.
- Writes per-contract minute parquet in the same shape as Alpaca outputs:
  data/synth/SPY/SPY_<OCC_SYMBOL>_bars_<start>_<end>.parquet

This lets you train/validate your RL setup immediately, then swap to real options data later
without changing the training code.

Requires: pandas, numpy, pyarrow, scipy, pyyaml
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
import yaml  # ensure pyyaml is installed


# ------------------------------ Config ------------------------------

@dataclass
class SynthCfg:
    underlying: str
    underlying_parquet: str
    outdir: str
    dte_list: List[int]
    moneyness_pct: List[float]
    ewma_halflife_minutes: int
    min_annual_vol: float
    max_annual_vol: float
    risk_free_rate: float
    timezone: str
    rth_only: bool
    write_chain: bool
    symbol_prefix: str


# ------------------------------ Utils ------------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def find_underlying_file(root: str, sym: str) -> str:
    base = os.path.join(root, sym)
    if not os.path.isdir(base):
        return ""
    cands = [
        os.path.join(base, f)
        for f in os.listdir(base)
        if f.endswith(".parquet") and "_underlying_" in f
    ]
    return sorted(cands)[-1] if cands else ""


def occ_symbol(root: str, expiry: pd.Timestamp, right: str, strike: float) -> str:
    # OCC compact: <ROOT><YYMMDD><C|P><STRIKE*1000 (8 digits)>
    yy = expiry.year % 100
    mm = expiry.month
    dd = expiry.day
    k1000 = int(round(strike * 1000))
    return f"{root}{yy:02d}{mm:02d}{dd:02d}{right.upper()}{k1000:08d}"


def rth_filter(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    # FIX: use .dt.tz_convert for Series (not .tz_convert which is for DatetimeIndex)
    tzz = ZoneInfo(tz)
    ts = pd.to_datetime(df["timestamp"], utc=True)  # Series[datetime64[ns, UTC]]
    local = ts.dt.tz_convert(tzz)
    # NYSE RTH 09:30–16:00 (inclusive of 16:00)
    mask = (local.dt.hour > 9) | ((local.dt.hour == 9) & (local.dt.minute >= 30))
    mask &= (local.dt.hour < 16) | ((local.dt.hour == 16) & (local.dt.minute == 0))
    out = df.copy()
    out["timestamp"] = ts
    return out.loc[mask]


def ewma_annualized_vol(close: pd.Series, halflife_min: int) -> pd.Series:
    ret = np.log(close.astype(float)).diff().fillna(0.0)
    # EWMA alpha from half-life in minutes
    alpha = 1 - np.exp(np.log(0.5) / float(max(halflife_min, 1)))
    var = ret.ewm(alpha=alpha, adjust=False).var().fillna(ret.var())
    minutes_per_year = 252 * 390
    ann_var = var * minutes_per_year
    return np.sqrt(np.clip(ann_var, 1e-12, None))


def clamp(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


def black_scholes_call(S, K, T, r, sigma):
    from scipy.stats import norm
    eps = 1e-12
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), eps)
    sigma = np.maximum(np.asarray(sigma, float), eps)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    from scipy.stats import norm
    eps = 1e-12
    S = np.asarray(S, float)
    K = np.asarray(K, float)
    T = np.maximum(np.asarray(T, float), eps)
    sigma = np.maximum(np.asarray(sigma, float), eps)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def to_minutes_ttm(ts: pd.Series, expiry: pd.Timestamp, tz: str) -> np.ndarray:
    """
    Time to maturity in years from UTC timestamps to expiry end-of-day (16:00) in local tz.
    Returns an array same length as ts.
    """
    tzz = ZoneInfo(tz)
    eod_local = pd.Timestamp(expiry.date(), tz=tzz) + pd.Timedelta(hours=16)  # 16:00 local
    eod_utc = eod_local.tz_convert("UTC")
    eod_ns = eod_utc.value  # int nanoseconds
    # ts is Series[datetime64[ns, UTC]]
    ts_ns = ts.astype("int64")  # int nanoseconds
    T_min = (eod_ns - ts_ns) / 1e9 / 60.0  # minutes to expiry EOD
    minutes_per_year = 252 * 390
    return np.maximum(T_min, 1.0) / minutes_per_year  # ensure positive T


# ------------------------------ Main ------------------------------

def load_cfg(path: str) -> SynthCfg:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) if path.endswith((".yaml", ".yml")) else json.load(f)
    raw = raw.get("synth", raw)
    return SynthCfg(**raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="configs/synth_spy.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    ensure_dir(cfg.outdir)

    # Locate underlying parquet
    u_parq = cfg.underlying_parquet or find_underlying_file("data/alpaca", cfg.underlying)
    if not u_parq:
        raise RuntimeError(
            "No underlying_parquet provided and none auto-discovered under data/alpaca/."
        )

    df = pd.read_parquet(u_parq)

    # Normalize timestamp column
    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        elif df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "timestamp"})
        else:
            raise RuntimeError("Cannot find 'timestamp' in underlying parquet.")

    use_cols = [c for c in ["timestamp", "open", "high", "low", "close"] if c in df.columns]
    if "timestamp" not in use_cols or "close" not in use_cols:
        raise RuntimeError("Underlying parquet must contain at least 'timestamp' and 'close' columns.")
    df = df[use_cols].copy()

    # RTH filter (optional)
    if cfg.rth_only:
        df = rth_filter(df, cfg.timezone)

    if df.empty:
        raise RuntimeError("Underlying dataframe is empty after RTH filtering.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Annualized IV via EWMA of minute returns, clamped
    ann_vol = ewma_annualized_vol(df["close"], cfg.ewma_halflife_minutes)
    ann_vol = clamp(ann_vol, cfg.min_annual_vol, cfg.max_annual_vol)
    df["ann_vol"] = ann_vol.values

    # Convenience arrays
    mid = df["close"].astype(float).values
    ts = pd.to_datetime(df["timestamp"], utc=True)

    start_date = ts.iloc[0].date().isoformat()
    end_date = ts.iloc[-1].date().isoformat()

    chain_rows = []
    root = cfg.symbol_prefix

    # Generate synthetic per-contract bars
    for dte in cfg.dte_list:
        for m in cfg.moneyness_pct:
            for right in ("C", "P"):
                first_spot = float(mid[0])
                K = round(first_spot * m, 2)

                # Compute expiry as (first_local_day + dte)
                tzz = ZoneInfo(cfg.timezone)
                first_local_day = ts.iloc[0].tz_convert(tzz).date()
                expiry = pd.Timestamp(first_local_day, tz=tzz) + pd.Timedelta(days=int(dte))

                # OCC symbol
                sym = occ_symbol(root, expiry, right, K)

                # Time-to-maturity vector in YEARS to EOD of expiry
                T = to_minutes_ttm(ts, expiry, cfg.timezone)

                sigma = df["ann_vol"].astype(float).values
                r = float(cfg.risk_free_rate)
                S = mid

                # Price via BS
                if right == "C":
                    price = black_scholes_call(S, K, T, r, sigma)
                else:
                    price = black_scholes_put(S, K, T, r, sigma)

                # Simple OHLC around synthetic mid
                jitter = np.maximum(price * 0.001, 0.001)  # small relative jitter
                rng = np.random.default_rng()              # reproducible if you set global seed earlier
                open_ = price + rng.uniform(-1, 1, size=price.shape) * jitter
                close_ = price + rng.uniform(-1, 1, size=price.shape) * jitter
                high_ = np.maximum(open_, close_) + np.abs(rng.uniform(0, 1, size=price.shape)) * jitter
                low_ = np.minimum(open_, close_) - np.abs(rng.uniform(0, 1, size=price.shape)) * jitter

                # Numerically safe synthetic volume:
                # heuristic = 1000 * 1 / |m - 1| (more vol near ATM),
                # with epsilon, NaN/Inf handling, expiry cutoff, and clipping.
                eps = 1e-6
                inv_dist = 1.0 / np.maximum(abs(m - 1.0), eps)
                vol_ = 1000.0 * inv_dist * (T > 0).astype(float)
                vol_ = np.nan_to_num(vol_, nan=1.0, posinf=1e5, neginf=1.0)
                vol_ = np.clip(vol_, 1.0, 5e5).astype(np.int64)

                bars = pd.DataFrame(
                    {
                        "timestamp": ts,
                        "open": open_,
                        "high": high_,
                        "low": low_,
                        "close": close_,
                        "volume": vol_,
                        "symbol": sym,
                    }
                )

                ensure_dir(cfg.outdir)
                out_path = os.path.join(
                    cfg.outdir, f"{root}_{sym}_bars_{start_date}_{end_date}.parquet"
                )
                bars.to_parquet(out_path, index=False)

                chain_rows.append(
                    {
                        "symbol": sym,
                        "expiry": expiry,
                        "strike": K,
                        "right": "call" if right == "C" else "put",
                    }
                )

    # Write a synthetic chain file for inspection
    if cfg.write_chain and chain_rows:
        chain_df = pd.DataFrame(chain_rows).drop_duplicates(subset=["symbol"])
        chain_path = os.path.join(cfg.outdir, f"{root}_chain_{start_date}_{end_date}.parquet")
        chain_df.to_parquet(chain_path, index=False)

    print(f"[Synth] wrote synthetic options for {cfg.underlying} → {cfg.outdir}")
    print(f"[Synth] contracts: ~{len(chain_rows)} files, window: {start_date} → {end_date}")


if __name__ == "__main__":
    main()
