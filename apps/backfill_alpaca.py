#!/usr/bin/env python3
"""
Backfill Alpaca options + underlying data into parquet with low-empties.

What it does:
- Fetches option contracts for an underlying (Trading API), robust to SDK variants.
- Falls back to snapshot option chain if needed.
- Filters contracts by DTE and moneyness around first-day spot.
- Uses NYSE RTH sessions to fetch bars (no weekends/overnights).
- Uses OPRA feed for options bars.
- Writes parquet shards only when non-empty; persists a skip list.

Requires:
    pip install alpaca-py pandas pyarrow python-dotenv pandas-market-calendars

Reads API keys from environment:
    ALPACA_API_KEY, ALPACA_SECRET_KEY  (or APCA_API_KEY_ID, APCA_API_SECRET_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# --- Alpaca SDK (Data: stocks/options; Trading: contracts) ---
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    OptionBarsRequest,
    OptionChainRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

# --------------------------------------------------------------------------------------
# Config model
# --------------------------------------------------------------------------------------

@dataclass
class BackfillCfg:
    underlying: str               # e.g., "SPY"
    start_date: str               # "YYYY-MM-DD"
    end_date: str                 # "YYYY-MM-DD"
    outdir: str                   # e.g., "data/alpaca/SPY"
    timeframe: str = "1Min"       # "1Min" or "5Min"
    feed: str = "opra"            # OPRA feed for options
    atm_moneyness_pct: float = 0.15  # keep strikes within ±15% of spot
    min_dte: int = 3
    max_dte: int = 45
    rth_only: bool = True
    timezone: str = "America/New_York"
    chunk_symbols: int = 200
    write_chain: bool = True
    write_underlying: bool = True


# --------------------------------------------------------------------------------------
# Helpers: sessions, env, IO
# --------------------------------------------------------------------------------------

def trading_sessions(start_date: str, end_date: str, tz: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (RTH_open, RTH_close) timestamps for NYSE between dates (inclusive)."""
    cal = mcal.get_calendar("XNYS")
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    tzz = ZoneInfo(tz)
    sessions: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, row in sched.iterrows():
        o = pd.Timestamp(row["market_open"]).tz_convert(tzz)
        c = pd.Timestamp(row["market_close"]).tz_convert(tzz)
        sessions.append((o, c))
    return sessions


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: str) -> None:
    if len(df):
        df.to_parquet(path, index=False)


def load_env():
    load_dotenv(override=False)


# --------------------------------------------------------------------------------------
# Alpaca clients
# --------------------------------------------------------------------------------------

def make_clients() -> tuple[OptionHistoricalDataClient, StockHistoricalDataClient, TradingClient]:
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment.")
    opt_client = OptionHistoricalDataClient(api_key, secret_key)
    stk_client = StockHistoricalDataClient(api_key, secret_key)
    trading = TradingClient(api_key, secret_key, paper=True)  # works for paper or live
    return opt_client, stk_client, trading


# --------------------------------------------------------------------------------------
# Chain parsing & filtering
# --------------------------------------------------------------------------------------

def parse_occ_symbol(sym: str):
    """
    Parse OCC option symbol like 'SPY241018C00535000' → (underlying, expiry_ts, right, strike_float)

    OCC format (compact):
      <ROOT><YYMMDD><C|P><STRIKE * 1000, 8 digits>
    ROOT: letters until the first digit (variable length)
    """
    if not sym or not isinstance(sym, str):
        return None, None, None, None

    i = 0
    while i < len(sym) and not sym[i].isdigit():
        i += 1
    root = sym[:i]
    rest = sym[i:]

    if len(rest) < 6 + 1 + 8:
        return root or None, None, None, None

    yymmdd = rest[:6]
    right = rest[6]  # 'C' or 'P'
    strike_raw = rest[7:15]  # 8 digits

    try:
        yy = int(yymmdd[0:2])
        mm = int(yymmdd[2:4])
        dd = int(yymmdd[4:6])
        yyyy = 2000 + yy
        expiry = pd.Timestamp(year=yyyy, month=mm, day=dd, tz="America/New_York")
    except Exception:
        expiry = None

    try:
        strike = int(strike_raw) / 1000.0
    except Exception:
        strike = None

    return root or None, expiry, right, strike


def fetch_contracts_via_trading(trading: TradingClient, underlying: str,
                                start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull contracts from Trading API, constrained to expiration in [start_date, end_date].
    Handles pagination if the response is paged.
    """
    rows = []
    page_token = None
    while True:
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            # limit to your backfill window so DTE filter doesn't nuke everything
            expiration_date_gte=start_date,   # e.g., "2024-08-01"
            expiration_date_lte=end_date,     # e.g., "2024-11-01"
            status="active",                   # default is fine; explicit for clarity
            page_token=page_token,             # paginate if needed
        )
        resp = trading.get_option_contracts(req)

        # unwrap items + next token; support dict or object shapes
        items, next_token = [], None
        if hasattr(resp, "option_contracts"):
            items = resp.option_contracts
            next_token = getattr(resp, "next_page_token", None)
        elif isinstance(resp, dict):
            items = resp.get("option_contracts") or resp.get("contracts") or resp.get("data") or []
            next_token = resp.get("next_page_token") or resp.get("next")
        elif hasattr(resp, "__dict__"):
            d = resp.__dict__
            items = d.get("option_contracts") or []
            next_token = d.get("next_page_token")

        for c in items:
            if isinstance(c, dict):
                sym = c.get("symbol")
                expiry = c.get("expiration_date")
                right = c.get("type") or c.get("option_type")
                strike = c.get("strike_price")
            else:
                sym = getattr(c, "symbol", None)
                expiry = getattr(c, "expiration_date", None)
                right = getattr(c, "type", None) or getattr(c, "option_type", None)
                strike = getattr(c, "strike_price", None)

            if not sym:
                continue

            # fill from OCC symbol if missing
            _, exp2, right2, strike2 = parse_occ_symbol(sym)
            expiry = expiry or exp2
            right = right or right2
            strike = strike or strike2

            if expiry is not None:
                try:
                    expiry = pd.Timestamp(expiry)
                    if expiry.tz is None:
                        expiry = expiry.tz_localize("America/New_York")
                except Exception:
                    pass

            rows.append(dict(symbol=sym, expiry=expiry, right=right, strike=strike))

        if not next_token:
            break
        page_token = next_token

    return pd.DataFrame(rows)


def fetch_chain_snapshot(opt_client: OptionHistoricalDataClient, underlying: str) -> pd.DataFrame:
    """
    Fallback: snapshot option chain from Data API.
    """
    req = OptionChainRequest(underlying_symbol=underlying)
    resp = opt_client.get_option_chain(req)

    contracts = None
    if hasattr(resp, "options"):
        contracts = resp.options
    if contracts is None and isinstance(resp, dict):
        contracts = resp.get("options") or resp.get("data") or resp.get("contracts")
    if contracts is None and hasattr(resp, "__dict__"):
        contracts = resp.__dict__.get("options")
    if not contracts:
        return pd.DataFrame(columns=["symbol", "expiry", "strike", "right"])

    rows = []
    for c in contracts:
        if isinstance(c, dict):
            cd = c
        else:
            cd = getattr(c, "__dict__", {}) or {}
            if "symbol" not in cd and hasattr(c, "symbol"):
                cd["symbol"] = c.symbol
            if "expiration_date" not in cd and hasattr(c, "expiration_date"):
                cd["expiration_date"] = c.expiration_date
            if "type" not in cd and hasattr(c, "type"):
                cd["type"] = c.type
            if "strike_price" not in cd and hasattr(c, "strike_price"):
                cd["strike_price"] = c.strike_price

        sym = cd.get("symbol") or cd.get("option_symbol") or cd.get("id") or cd.get("symbol_code")
        expiry = cd.get("expiration_date") or cd.get("expiry") or cd.get("expiration")
        right = cd.get("type") or cd.get("right") or cd.get("option_type")
        strike = cd.get("strike_price") or cd.get("strike")

        if sym:
            _, exp2, right2, strike2 = parse_occ_symbol(sym)
            expiry = expiry or exp2
            right = right or right2
            strike = strike or strike2

        if expiry is not None:
            try:
                expiry = pd.Timestamp(expiry)
                if expiry.tz is None:
                    expiry = expiry.tz_localize("America/New_York")
            except Exception:
                pass

        rows.append(dict(symbol=sym, expiry=expiry, right=right, strike=strike))

    return pd.DataFrame(rows)


def filter_chain(
    chain_df: pd.DataFrame,
    first_session_start: pd.Timestamp,
    spot0: float,
    atm_pct: float,
    min_dte: int,
    max_dte: int,
    tz: str,
) -> pd.DataFrame:
    """Apply DTE and moneyness filters; keep unique symbols."""
    if chain_df.empty:
        return chain_df

    tzz = ZoneInfo(tz)
    first_day = first_session_start.tz_convert(tzz)
    chain = chain_df.copy()
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    if chain["expiry"].dt.tz is None:
        chain["expiry"] = chain["expiry"].dt.tz_localize(tzz)
    chain["dte"] = (chain["expiry"] - first_day).dt.days
    chain = chain[(chain["dte"] >= min_dte) & (chain["dte"] <= max_dte)]

    # moneyness: strike within ± atm_pct of spot
    m = pd.to_numeric(chain["strike"], errors="coerce").astype(float) / float(spot0)
    chain = chain[(m >= (1 - atm_pct)) & (m <= (1 + atm_pct))]

    chain = chain.dropna(subset=["symbol"])
    chain = chain.drop_duplicates(subset=["symbol"])
    return chain[["symbol", "expiry", "strike", "right", "dte"]]


# --------------------------------------------------------------------------------------
# Data fetchers
# --------------------------------------------------------------------------------------

def fetch_underlying_bars(
    stk_client: StockHistoricalDataClient,
    symbol: str,
    sessions: List[Tuple[pd.Timestamp, pd.Timestamp]],
    timeframe: str,
) -> pd.DataFrame:
    """Concatenate RTH bars across sessions for the underlying (e.g., SPY)."""
    frames = []
    tf = TimeFrame.Minute if timeframe.lower().startswith("1") else TimeFrame(5, "Min")
    for (start_ts, end_ts) in sessions:
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_ts.tz_convert("UTC"),
            end=end_ts.tz_convert("UTC"),
            limit=10_000,
        )
        bars = stk_client.get_stock_bars(req)
        if hasattr(bars, "df"):
            df = bars.df.copy()
        else:
            df = pd.DataFrame(bars)
        if len(df):
            df = df.reset_index()
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
        out["underlying"] = symbol
        return out
    return pd.DataFrame()


def fetch_option_bars_for_symbol(
    opt_client: OptionHistoricalDataClient,
    opt_symbol: str,
    sessions: List[Tuple[pd.Timestamp, pd.Timestamp]],
    timeframe: str,
    feed: str,
) -> pd.DataFrame:
    """Concatenate RTH bars across sessions for a single option contract."""
    frames = []
    tf = TimeFrame.Minute if timeframe.lower().startswith("1") else TimeFrame(5, "Min")
    for (start_ts, end_ts) in sessions:
        req = OptionBarsRequest(
            symbol_or_symbols=[opt_symbol],
            timeframe=tf,
            start=start_ts.tz_convert("UTC"),
            end=end_ts.tz_convert("UTC"),
            feed=feed,            # ensure OPRA for options
            limit=10_000,
        )
        bars = opt_client.get_option_bars(req)
        if hasattr(bars, "df"):
            df_all = bars.df.copy()
            if isinstance(df_all.index, pd.MultiIndex):
                try:
                    df = df_all.xs(opt_symbol, level=0).reset_index()
                except Exception:
                    df = df_all.reset_index()
            else:
                df = df_all.reset_index()
        else:
            df = pd.DataFrame(bars)
        if len(df):
            df["symbol"] = opt_symbol
            frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
        return out
    return pd.DataFrame()


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML or JSON config path")
    args = parser.parse_args()

    # Support YAML or JSON
    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
        import yaml
        with open(args.config, "r") as f:
            cfg_raw = yaml.safe_load(f)
    else:
        with open(args.config, "r") as f:
            cfg_raw = json.load(f)

    backfill_dict = cfg_raw["backfill"] if "backfill" in cfg_raw else cfg_raw
    cfg = BackfillCfg(**backfill_dict)
    tz = cfg.timezone

    load_env()
    opt_client, stk_client, trading = make_clients()

    # Sessions
    sessions = trading_sessions(cfg.start_date, cfg.end_date, tz)
    if not sessions:
        print("[Backfill] No RTH sessions in range; nothing to do.")
        return

    first_open, _ = sessions[0]
    print(f"[Backfill] underlying={cfg.underlying} {cfg.start_date} \u2192 {cfg.end_date}")

    ensure_dir(cfg.outdir)

    # --- Underlying bars (for spot0 + modeling) ---
    under_df = fetch_underlying_bars(stk_client, cfg.underlying, sessions, cfg.timeframe) if cfg.write_underlying else pd.DataFrame()
    if cfg.write_underlying and len(under_df):
        spot0 = float(under_df["close"].astype(float).median())
        print(f"[Backfill] first-day median close (spot0) ~ {spot0:.2f}")
        save_parquet(
            under_df,
            os.path.join(cfg.outdir, f"{cfg.underlying}_underlying_{cfg.start_date}_{cfg.end_date}.parquet"),
        )
    else:
        spot0 = np.nan
        print("[Backfill] WARNING: no underlying bars written; spot0 unavailable.")

    # --- Contracts (Trading API) ---
    chain_df = fetch_contracts_via_trading(trading, cfg.underlying, cfg.start_date, cfg.end_date)

    # Fallback: snapshot chain (Data API)
    if chain_df.empty:
        snap_df = fetch_chain_snapshot(opt_client, cfg.underlying)
        chain_df = snap_df

    if cfg.write_chain and len(chain_df):
        save_parquet(chain_df, os.path.join(cfg.outdir, f"{cfg.underlying}_chain_{cfg.start_date}_{cfg.end_date}.parquet"))

    if not np.isfinite(spot0):
        if len(chain_df):
            spot0 = float(pd.to_numeric(chain_df["strike"], errors="coerce").dropna().median())
            print(f"[Backfill] spot0 fallback from chain median strike ~ {spot0:.2f}")
        else:
            raise RuntimeError("No spot0 available and chain is empty; cannot filter.")

    filt = filter_chain(
        chain_df=chain_df,
        first_session_start=first_open,
        spot0=spot0,
        atm_pct=cfg.atm_moneyness_pct,
        min_dte=cfg.min_dte,
        max_dte=cfg.max_dte,
        tz=tz,
    )

    print(f"[Backfill] chain={len(chain_df)} filtered={len(filt)} (DTE in [{cfg.min_dte},{cfg.max_dte}], ATM ±{int(cfg.atm_moneyness_pct*100)}%)")

    # --- Fetch bars per option symbol over RTH sessions ---
    skipped: list[str] = []
    written = 0
    for i, row in filt.reset_index(drop=True).iterrows():
        sym = row["symbol"]
        df = fetch_option_bars_for_symbol(opt_client, sym, sessions, cfg.timeframe, cfg.feed)
        if len(df):
            df["expiry"] = row["expiry"]
            df["strike"] = row["strike"]
            df["right"] = row["right"]
            out_path = os.path.join(cfg.outdir, f"{cfg.underlying}_{sym}_bars_{cfg.start_date}_{cfg.end_date}.parquet")
            save_parquet(df, out_path)
            written += 1
        else:
            skipped.append(sym)

        if (i + 1) % 50 == 0:
            print(f"[Backfill] progress {i+1}/{len(filt)} written={written} skipped={len(skipped)}")

    if skipped:
        with open(os.path.join(cfg.outdir, "_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=2)

    print(f"[Backfill] ✅ done. written={written} skipped={len(skipped)} outdir={cfg.outdir}")


if __name__ == "__main__":
    main()
