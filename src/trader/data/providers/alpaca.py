# src/trader/data/providers/alpaca.py
from __future__ import annotations
import os
import time
import json
import gzip
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dateutil import parser as duparser

from trader.utils.logging import get_logger
from trader.utils.env import require_env

UTC = timezone.utc

@dataclass
class AlpacaConfig:
    symbols: List[str]
    start: str
    end: str
    timeframe: str
    dte_min: int
    dte_max: int
    strikes_around: int
    cache_dir: str
    backfill: bool
    fetch_iv: bool
    batch_size: int
    rate_limit_sleep: float

def _get_base_url() -> str:
    # Default to official data endpoint if not set
    return os.getenv("APCA_API_BASE_URL", "https://data.alpaca.markets")

def _get_headers() -> dict:
    envs = require_env(["APCA_API_KEY_ID", "APCA_API_SECRET_KEY"])
    return {
        "APCA-API-KEY-ID": envs["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": envs["APCA_API_SECRET_KEY"],
        "Accept-Encoding": "gzip",
    }

def _parse_ts(s: str) -> datetime:
    return duparser.isoparse(s).astimezone(UTC)

def _safe_json(resp: requests.Response) -> dict:
    """
    Robust JSON loader:
    - Prefer resp.json() (requests already handles gzip/deflate).
    - If that fails *and* header claims gzip, try manual gunzip.
    - If still failing, raise with the first ~200 chars of body.
    """
    try:
        return resp.json()
    except Exception:
        enc = resp.headers.get("Content-Encoding", "")
        if "gzip" in enc.lower():
            try:
                buf = io.BytesIO(resp.content)
                with gzip.GzipFile(fileobj=buf) as gz:
                    data = gz.read()
                return json.loads(data.decode("utf-8"))
            except Exception:
                pass
        # final fallback: try plain text->json
        try:
            return json.loads(resp.text)
        except Exception:
            snippet = resp.content[:200]
            raise requests.RequestException(
                f"Failed to parse JSON (status={resp.status_code}, encoding='{enc}'). "
                f"Body head: {snippet!r}"
            )

class AlpacaHTTP:
    def __init__(self, rate_limit_sleep: float = 0.5):
        self.base_url = _get_base_url()
        self.s = requests.Session()
        self.s.headers.update(_get_headers())
        self.sleep = rate_limit_sleep
        self.log = get_logger("alpaca.http")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((requests.RequestException,))
    )
    def get(self, path: str, params: dict) -> dict:
        url = f"{self.base_url}{path}"
        r = self.s.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(self.sleep * 2.0)
            raise requests.RequestException("429 Too Many Requests")
        if not r.ok:
            # Try to surface Alpaca's message if any
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise requests.RequestException(f"HTTP {r.status_code}: {err}")
        time.sleep(self.sleep)
        return _safe_json(r)

def _select_atm_strikes(chain_df: pd.DataFrame, spot: float, k: int) -> pd.DataFrame:
    chain_df = chain_df.copy()
    chain_df["atm_dist"] = (chain_df["strike"] - spot).abs()
    chain_df.sort_values("atm_dist", inplace=True)
    return chain_df.groupby(["expiration", "right"]).head(1 + 2 * k)

def _partition_path(cache_dir: str, underlying: str, date_str: str, right: str, expiration: str) -> str:
    p = os.path.join(cache_dir, underlying, date_str, right, expiration)
    os.makedirs(p, exist_ok=True)
    return p

def _write_parquet(df: pd.DataFrame, path: str):
    if df.empty:
        return
    df.to_parquet(path, index=False)

def _read_parquet_dir(dir_path: str) -> pd.DataFrame:
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".parquet")]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

class AlpacaBackfill:
    def __init__(self, cfg: AlpacaConfig):
        # ensure env is present now (after .env load)
        _ = _get_headers()
        self.cfg = cfg
        self.h = AlpacaHTTP(rate_limit_sleep=cfg.rate_limit_sleep)
        self.log = get_logger("alpaca.backfill")

    def run(self):
        for underlying in self.cfg.symbols:
            self.log.info(f"[Backfill] underlying={underlying} {self.cfg.start} → {self.cfg.end}")
            u_bars = self._fetch_underlying_bars(underlying)
            if u_bars.empty:
                raise RuntimeError("No underlying bars returned")
            first_day = u_bars["ts"].dt.date.min()
            spot0 = float(u_bars.loc[u_bars["ts"].dt.date == first_day, "close"].median())
            self.log.info(f"[Backfill] first-day median close (spot0) ~ {spot0:.2f}")

            chain = self._fetch_chain(underlying)
            if chain.empty:
                raise RuntimeError("Empty options chain")
            chain = self._filter_chain_by_dte(chain)
            chain = _select_atm_strikes(chain, spot0, self.cfg.strikes_around)
            self.log.info(f"[Backfill] selected contracts: {len(chain)}")

            self._write_underlying_parquet(underlying, u_bars)
            self._fetch_and_write_option_bars(underlying, chain)

    def _fetch_underlying_bars(self, symbol: str) -> pd.DataFrame:
        params = {
            "timeframe": self.cfg.timeframe,
            "start": self.cfg.start,
            "end": self.cfg.end,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "sip",
            "page_token": None,
        }
        path = f"/v2/stocks/{symbol}/bars"
        rows = []
        while True:
            data = self.h.get(path, params)
            bars = data.get("bars", [])
            for b in bars:
                rows.append({
                    "ts": _parse_ts(b["t"]),
                    "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"],
                    "volume": b.get("v", 0),
                })
            nxt = data.get("next_page_token")
            if nxt:
                params["page_token"] = nxt
            else:
                break
        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("ts", inplace=True)
        return df

    def _fetch_chain(self, underlying: str) -> pd.DataFrame:
        params = {"underlying": underlying, "status": "active", "limit": 1000, "page_token": None}
        path = "/v2/options/contracts"
        rows = []
        while True:
            data = self.h.get(path, params)
            cs = data.get("contracts", [])
            for c in cs:
                rows.append({
                    "symbol": c["symbol"],
                    "underlying": c["underlying_symbol"],
                    "right": c["type"].upper()[0],
                    "strike": float(c["strike"]),
                    "expiration": c["expiration_date"],
                })
            nxt = data.get("next_page_token")
            if nxt:
                params["page_token"] = nxt
            else:
                break
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df.drop_duplicates(subset=["symbol"], inplace=True)
        return df

    def _filter_chain_by_dte(self, chain: pd.DataFrame) -> pd.DataFrame:
        start_dt = duparser.isoparse(self.cfg.start).date()
        end_dt = duparser.isoparse(self.cfg.end).date()
        chain = chain.copy()
        chain["expiration"] = pd.to_datetime(chain["expiration"]).dt.date
        dte = (chain["expiration"] - start_dt).apply(lambda x: x.days)
        mask = (dte >= self.cfg.dte_min) & (dte <= self.cfg.dte_max) & (chain["expiration"] <= end_dt)
        return chain.loc[mask].reset_index(drop=True)

    def _write_underlying_parquet(self, underlying: str, u_bars: pd.DataFrame):
        u_bars = u_bars.copy()
        u_bars["date"] = u_bars["ts"].dt.date.astype(str)
        for d, sdf in u_bars.groupby("date"):
            outdir = os.path.join(self.cfg.cache_dir, underlying, d, "_underlying")
            os.makedirs(outdir, exist_ok=True)
            _write_parquet(sdf.drop(columns=["date"]), os.path.join(outdir, "bars.parquet"))

    def _fetch_and_write_option_bars(self, underlying: str, chain_df: pd.DataFrame):
        sym_list = chain_df["symbol"].tolist()
        B = self.cfg.batch_size
        for i in range(0, len(sym_list), B):
            batch = sym_list[i:i+B]
            self._fetch_write_bars_batch(underlying, chain_df, batch)

    def _fetch_write_bars_batch(self, underlying: str, chain_df: pd.DataFrame, batch_syms: List[str]):
        params = {
            "symbols": ",".join(batch_syms),
            "start": self.cfg.start,
            "end": self.cfg.end,
            "timeframe": self.cfg.timeframe,
            "limit": 10000,
            "adjustment": "raw",
            "feed": "opra",
            "page_token": None,
        }
        path = "/v2/options/bars"
        all_rows: List[dict] = []
        while True:
            data = self.h.get(path, params)
            bars = data.get("bars", {})
            for sym, recs in bars.items():
                row = chain_df.loc[chain_df["symbol"] == sym]
                if row.empty:
                    continue
                right = row["right"].values[0]
                strike = float(row["strike"].values[0])
                expiration = str(row["expiration"].values[0])
                for b in recs:
                    all_rows.append({
                        "ts": _parse_ts(b["t"]),
                        "symbol": sym,
                        "underlying": underlying,
                        "right": right,
                        "strike": strike,
                        "expiration": expiration,
                        "open": b.get("o", np.nan),
                        "high": b.get("h", np.nan),
                        "low": b.get("l", np.nan),
                        "close": b.get("c", np.nan),
                        "volume": b.get("v", 0),
                        "trade_count": b.get("n", 0),
                        "vw": b.get("vw", np.nan),
                    })
            nxt = data.get("next_page_token")
            if nxt:
                params["page_token"] = nxt
            else:
                break

        if not all_rows:
            return
        df = pd.DataFrame(all_rows)
        df.sort_values(["symbol", "ts"], inplace=True)
        df["date"] = df["ts"].dt.date.astype(str)

        for (d, r, exp), sdf in df.groupby(["date", "right", "expiration"]):
            outdir = _partition_path(self.cfg.cache_dir, underlying, d, r, exp)
            fname = f"part_{hash(tuple(sdf['symbol'].unique())) & 0xfffffff:x}.parquet"
            _write_parquet(sdf.drop(columns=["date"]), os.path.join(outdir, fname))

class AlpacaProvider:
    def __init__(self, cfg_dict: Dict):
        self.log = get_logger("alpaca.provider")
        self.cfg = AlpacaConfig(
            symbols=cfg_dict["symbols"],
            start=cfg_dict["start"],
            end=cfg_dict["end"],
            timeframe=cfg_dict.get("timeframe", "1Min"),
            dte_min=int(cfg_dict.get("dte_min", 7)),
            dte_max=int(cfg_dict.get("dte_max", 30)),
            strikes_around=int(cfg_dict.get("strikes_around", 5)),
            cache_dir=cfg_dict.get("cache_dir", "data/alpaca"),
            backfill=bool(cfg_dict.get("backfill", False)),
            fetch_iv=bool(cfg_dict.get("fetch_iv", False)),
            batch_size=int(cfg_dict.get("batch_size", 100)),
            rate_limit_sleep=float(cfg_dict.get("rate_limit_sleep", 0.5)),
        )

        if self.cfg.backfill:
            self.log.info("Backfill requested; fetching from Alpaca into Parquet cache ...")
            AlpacaBackfill(self.cfg).run()
            self.log.info("Backfill complete. Switching to cache-only mode.")

        self._index_cache()
        self._feature_dim = 8  # simple starter spec; extend as you add features

    def observation_spec(self) -> Dict[str, int]:
        return {"feature_dim": self._feature_dim}

    def reset(self) -> None:
        pass

    def _index_cache(self):
        self.by_und_dates: Dict[str, List[str]] = {}
        for und in self.cfg.symbols:
            und_root = os.path.join(self.cfg.cache_dir, und)
            if not os.path.isdir(und_root):
                self.by_und_dates[und] = []
                continue
            dates = sorted([d for d in os.listdir(und_root) if d[:4].isdigit()])
            self.by_und_dates[und] = dates
        self.log.info(f"Indexed cache: { {k: len(v) for k,v in self.by_und_dates.items()} }")

    def stream(self) -> Iterable[tuple[dict, list[dict]]]:
        for und in self.cfg.symbols:
            for d in self.by_und_dates.get(und, []):
                u_path = os.path.join(self.cfg.cache_dir, und, d, "_underlying", "bars.parquet")
                if not os.path.exists(u_path):
                    continue
                u_df = pd.read_parquet(u_path)

                parts = []
                day_dir = os.path.join(self.cfg.cache_dir, und, d)
                for right in ("C", "P"):
                    right_dir = os.path.join(day_dir, right)
                    if not os.path.isdir(right_dir):
                        continue
                    for exp in os.listdir(right_dir):
                        pdir = os.path.join(right_dir, exp)
                        if os.path.isdir(pdir):
                            parts.append(_read_parquet_dir(pdir))
                if not parts:
                    continue
                o_df = pd.concat(parts, ignore_index=True)
                o_df.sort_values(["symbol", "ts"], inplace=True)
                o_df["close_ffill"] = o_df.groupby("symbol")["close"].ffill()

                for _, row in u_df.iterrows():
                    ts = row["ts"]
                    sdf = o_df[o_df["ts"] == ts]
                    if sdf.empty:
                        sdf = o_df[o_df["ts"] < ts].groupby("symbol").tail(1)
                        if sdf.empty:
                            continue
                    und_bar = {
                        "ts": ts,
                        "open": float(row["open"]), "high": float(row["high"]),
                        "low": float(row["low"]), "close": float(row["close"]),
                        "volume": float(row.get("volume", 0)),
                    }
                    slice_list: list[dict] = []
                    for _, r in sdf.iterrows():
                        dte = (pd.to_datetime(r["expiration"]).tz_localize(UTC) - ts).days
                        mny = float(row["close"]) / float(r["strike"])
                        slice_list.append({
                            "ts": ts,
                            "symbol": r["symbol"],
                            "right": r["right"],
                            "strike": float(r["strike"]),
                            "expiration": str(r["expiration"]),
                            "dte": int(max(dte, 0)),
                            "close": float(r.get("close_ffill", r.get("close", np.nan))),
                            "volume": float(r.get("volume", 0)),
                            "moneyness": mny,
                        })
                    yield und_bar, slice_list
