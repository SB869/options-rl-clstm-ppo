# src/trader/data/providers/alpaca.py
from __future__ import annotations
import os
import io
import gzip
import json
import time
import pathlib
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable, Tuple

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from trader.utils.logging import get_logger
from trader.utils.env import load_env, expand_env_in_obj


@dataclass
class BackfillConfig:
    underlying: str
    start: str
    end: str
    cache_dir: str = "data/options/alpaca"
    timeframe: str = "1Day"     # 1Min / 1Hour / 1Day (v1beta1 options bars)
    feed: str = "indicative"    # kept for future use; NOT sent to /v1beta1/options/bars
    page_limit_bars: int = 50000
    page_limit_chain: int = 1000
    # endpoints / credentials (can be filled from YAML or env)
    data_base: str = "https://data.alpaca.markets"
    key_id: Optional[str] = None
    secret_key: Optional[str] = None


class AlpacaHTTP:
    def __init__(self, data_base: str, key_id: str, secret_key: str):
        self.data_base = data_base.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json",
        })
        self.log = get_logger("alpaca.http")

    def _maybe_gzip_to_json(self, r: requests.Response) -> Any:
        """
        Safely parse JSON whether or not the server uses gzip encoding.
        requests/urllib3 already auto-decompresses gzip/deflate.
        """
        try:
            return r.json()
        except ValueError:
            raw = r.content
            if raw.startswith(b"\x1f\x8b"):  # gzip magic
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                    return json.loads(gz.read().decode("utf-8"))
            return json.loads(raw.decode("utf-8"))

    def _check(self, r: requests.Response) -> Any:
        if 200 <= r.status_code < 300:
            return self._maybe_gzip_to_json(r)
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        raise requests.RequestException(f"HTTP {r.status_code}: {body}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def get_data(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.data_base}/{path.lstrip('/')}"
        r = self.s.get(url, params=params, timeout=60)
        # handle basic rate limiting gently
        if r.status_code == 429:
            self.log.warning("429 Too Many Requests; backing off…")
            time.sleep(1.5)
            raise requests.RequestException("rate limited")
        return self._check(r)


class AlpacaBackfill:
    """
    Backfills:
      1) underlying daily bars from /v2/stocks/bars  (data.alpaca.markets)
      2) option 'chain' via /v1beta1/options/snapshots/{UNDERLYING}
      3) option bars via /v1beta1/options/bars
    Saves Parquet into cache_dir.
    """
    def __init__(self, cfg: BackfillConfig | Dict[str, Any]):
        load_env()  # ensure .env loaded

        if isinstance(cfg, dict):
            cfg = BackfillConfig(**expand_env_in_obj(cfg))
        else:
            # in case any fields in dataclass carry ${VAR}
            cfg = BackfillConfig(**expand_env_in_obj(cfg.__dict__))

        # allow env fallback for creds
        key = cfg.key_id or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID")
        sec = cfg.secret_key or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
        if not key or not sec:
            raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in env or config.")

        self.cfg = cfg
        self.http = AlpacaHTTP(cfg.data_base, key, sec)
        self.log = get_logger("alpaca.backfill")

        self.base_dir = pathlib.Path(cfg.cache_dir) / cfg.underlying.upper()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        u = self.cfg.underlying.upper()
        self.log.info(f"[Backfill] underlying={u} {self.cfg.start} → {self.cfg.end}")

        # 1) underlying bars (daily)
        u_df = self._fetch_underlying_bars(u)
        if not u_df.empty:
            first_close = u_df.iloc[:10]["close"].median()
            self.log.info(f"[Backfill] first-day median close (spot0) ~ {first_close:.2f}")
        self._save_parquet(u_df, self.base_dir / f"{u}_underlying_{self.cfg.start}_{self.cfg.end}.parquet")

        # 2) chain via snapshots (data domain)
        chain = self._fetch_chain_snapshots(u)
        if chain.empty:
            self.log.warning("Snapshots returned empty chain; stopping.")
            return
        self._save_parquet(chain, self.base_dir / f"{u}_chain_{self.cfg.start}_{self.cfg.end}.parquet")

        # 3) bars for symbols (chunked)
        syms = chain["symbol"].unique().tolist()
        if not syms:
            self.log.warning("No symbols from chain; stopping.")
            return

        CHUNK = 150
        for i in range(0, len(syms), CHUNK):
            batch = syms[i:i+CHUNK]
            bars = self._fetch_option_bars(batch)
            out = self.base_dir / f"{u}_bars_{i:05d}_{i+len(batch)-1:05d}.parquet"
            self._save_parquet(bars, out)

        self.log.info("[Backfill] ✅ done.")

    # ---------- helpers ----------
    def _fetch_underlying_bars(self, symbol: str) -> pd.DataFrame:
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": self.cfg.start,
            "end": self.cfg.end,
            "limit": 5000,
            "adjustment": "raw",
        }
        out: list[dict] = []
        next_token: Optional[str] = None
        while True:
            if next_token:
                params["page_token"] = next_token
            data = self.http.get_data("/v2/stocks/bars", params)
            bars = (data or {}).get("bars", {}).get(symbol, [])
            for b in bars:
                out.append({
                    "symbol": symbol,
                    "t": b.get("t"),
                    "open": b.get("o"),
                    "high": b.get("h"),
                    "low":  b.get("l"),
                    "close": b.get("c"),
                    "volume": b.get("v"),
                    "n": b.get("n"),
                    "vw": b.get("vw"),
                })
            next_token = (data or {}).get("next_page_token")
            if not next_token:
                break
            time.sleep(0.05)
        return pd.DataFrame(out)

    def _iter_snapshots(self, snaps: Any) -> Iterable[Tuple[Optional[str], Dict[str, Any]]]:
        """
        Normalize 'snapshots' to (symbol_key, snapshot_dict) pairs.
        - If snaps is a dict: { "SYM": {...}, ... }
        - If snaps is a list: [ {...}, ... ]
        """
        if isinstance(snaps, dict):
            for sym_key, s in snaps.items():
                if isinstance(s, dict):
                    yield sym_key, s
                else:
                    # Unexpected, but preserve symbol if possible
                    yield sym_key, {}
        elif isinstance(snaps, list):
            for s in snaps:
                if isinstance(s, dict):
                    yield None, s
                elif isinstance(s, str):
                    # Some APIs may return a plain symbol string
                    yield s, {}
        else:
            return  # empty/unknown

    def _fetch_chain_snapshots(self, underlying: str) -> pd.DataFrame:
        """
        DATA API (not trading): /v1beta1/options/snapshots/{UNDERLYING}
        Returns snapshots per contract; we extract contract symbol/metadata.
        Handles both list and dict response shapes.
        """
        params = {
            "limit": self.cfg.page_limit_chain,
        }
        out: list[dict] = []
        page_token: Optional[str] = None
        path = f"/v1beta1/options/snapshots/{underlying}"
        while True:
            if page_token:
                params["page_token"] = page_token
            data = self.http.get_data(path, params)

            snaps_raw = (data or {}).get("snapshots", [])
            for sym_key, s in self._iter_snapshots(snaps_raw):
                # prefer contract/option blocks; fall back to top-level and/or key
                meta = {}
                if isinstance(s, dict):
                    meta = s.get("contract") or s.get("option") or {}
                sym = None
                if isinstance(meta, dict):
                    sym = meta.get("symbol")
                if not sym and isinstance(s, dict):
                    sym = s.get("symbol")
                if not sym:
                    sym = sym_key  # final fallback from dict key or string entry
                if not sym:
                    continue

                # Extract metadata defensively
                right = ""
                strike = 0.0
                expiration = None
                if isinstance(meta, dict):
                    right = (meta.get("type") or meta.get("right") or "")
                    strike = meta.get("strike_price") or meta.get("strike") or 0.0
                    expiration = meta.get("expiration_date") or meta.get("expiration")

                out.append({
                    "symbol": sym,
                    "right": str(right).upper()[:1] if right else "",
                    "strike": float(strike) if strike is not None else 0.0,
                    "expiration": expiration,
                })

            page_token = (data or {}).get("next_page_token")
            if not page_token:
                break
            time.sleep(0.1)

        df = pd.DataFrame(out)
        if not df.empty:
            df.drop_duplicates(subset=["symbol"], inplace=True)
        return df

    def _fetch_option_bars(self, symbols: List[str]) -> pd.DataFrame:
        """
        DATA API v1beta1: /v1beta1/options/bars
        Note: do NOT send 'feed' – endpoint may reject it; Alpaca selects feed based on entitlements.
        Limit is capped at 10,000 per request; use next_page_token for the rest.
        """
        if not symbols:
            return pd.DataFrame()

        params = {
            "symbols": ",".join(symbols),
            "timeframe": self.cfg.timeframe,
            "start": self.cfg.start,
            "end": self.cfg.end,
            "limit": min(self.cfg.page_limit_bars, 10000),  # API max
        }
        out: list[dict] = []
        next_token: Optional[str] = None
        while True:
            if next_token:
                params["page_token"] = next_token
            data = self.http.get_data("/v1beta1/options/bars", params)
            bars_by_sym = (data or {}).get("bars", {})
            for sym, rows in bars_by_sym.items():
                for b in rows:
                    out.append({
                        "symbol": sym,
                        "t": b.get("t"),
                        "open": b.get("o"),
                        "high": b.get("h"),
                        "low":  b.get("l"),
                        "close": b.get("c"),
                        "volume": b.get("v"),
                        "n": b.get("n"),
                        "vw": b.get("vw"),
                    })
            next_token = (data or {}).get("next_page_token")
            if not next_token:
                break
            time.sleep(0.1)

        return pd.DataFrame(out)

    def _save_parquet(self, df: pd.DataFrame, path: pathlib.Path) -> None:
        if df is None or df.empty:
            self.log.warning(f"[Backfill] empty frame, skip save: {path.name}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        self.log.info(f"[Backfill] wrote {path} rows={len(df):,}")
