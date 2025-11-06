# src/trader/data/providers/alpaca.py
from __future__ import annotations
import os, io, gzip, json, time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterable
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from trader.utils.logging import get_logger
from trader.utils.env import load_dotenv_if_present

log = get_logger("alpaca")

API_BASE = "https://data.alpaca.markets"
# Endpoints:
#  - underlying bars (stocks):   /v2/stocks/bars
#  - option chain:               /v1beta3/options/snapshots/chains
#  - option bars (historical):   /v1beta3/options/bars

# -----------------------
# Safe JSON decode helper
# -----------------------
def _safe_json(r: requests.Response) -> Any:
    enc = (r.headers.get("Content-Encoding") or "").lower()
    ct  = (r.headers.get("Content-Type") or "").lower()

    # happy path: plain JSON
    if enc != "gzip":
        try:
            return r.json()
        except Exception:
            # diagnostic breadcrumbs
            snippet = r.text[:400].replace("\n", "\\n")
            raise RuntimeError(
                f"Expected JSON but got parse error. "
                f"status={r.status_code} ct={ct} enc={enc} body_snippet={snippet}"
            )

    # gzip path
    raw = r.content or b""
    if not raw:
        # empty body (some 204/304 etc.)
        return None

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            data = gz.read()
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            snippet = (data[:400] if isinstance(data, (bytes, bytearray)) else b"")\
                .decode("utf-8", errors="replace").replace("\n", "\\n")
            raise RuntimeError(
                f"Gzipped body wasn’t valid JSON. "
                f"status={r.status_code} ct={ct} enc={enc} body_snippet={snippet}"
            )
    except OSError as e:
        # This is the BadGzipFile path you saw
        snippet = (raw[:400]).decode("utf-8", errors="replace").replace("\n", "\\n")
        raise RuntimeError(
            f"Content-Encoding advertised gzip, but payload not gzipped or corrupt. "
            f"status={r.status_code} ct={ct} enc={enc} err={e} body_snippet={snippet}"
        )

# -----------------------
# HTTP wrapper with retry
# -----------------------
class AlpacaHTTP:
    def __init__(self, key: str, secret: str):
        self.s = requests.Session()
        # ask for gzip but don’t assume it will be returned
        self.s.headers.update({
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "options-rl-clstm-ppo/0.1",
        })

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((requests.RequestException,))
    )
    def get(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{API_BASE}{path}"
        r = self.s.get(url, params=params, timeout=30)
        # Handle rate limits explicitly
        if r.status_code == 429:
            reset = r.headers.get("x-ratelimit-reset") or r.headers.get("Retry-After")
            log.warning(f"429 Too Many Requests; reset={reset}, backing off…")
            time.sleep(2)
            raise requests.RequestException("rate limited")
        if r.status_code >= 500:
            raise requests.RequestException(f"server error {r.status_code}")
        if r.status_code >= 400:
            # try to extract error JSON/text
            try:
                body = r.json()
            except Exception:
                body = r.text[:400]
            raise RuntimeError(f"HTTP {r.status_code} {url} params={params} body={body}")
        return _safe_json(r)

# -----------------------
# Backfill driver
# -----------------------
@dataclass
class BackfillConfig:
    underlying: str
    start: str          # YYYY-MM-DD
    end: str            # YYYY-MM-DD
    output_dir: str     # base cache dir
    strikes_around: int = 10
    dte_min: int = 3
    dte_max: int = 45
    bar_timeframe: str = "1Day"   # Alpaca: 1Min, 1Hour, 1Day

class AlpacaBackfill:
    def __init__(self, cfg: BackfillConfig):
        load_dotenv_if_present()  # loads .env if found
        key = os.getenv("ALPACA_KEY_ID")
        secret = os.getenv("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_KEY_ID / ALPACA_SECRET_KEY in environment (.env).")
        self.h = AlpacaHTTP(key, secret)
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

    def run(self) -> None:
        u = self.cfg.underlying
        log.info(f"[Backfill] underlying={u} {self.cfg.start} → {self.cfg.end}")

        u_bars = self._fetch_underlying_bars(u)
        self._write_parquet(u_bars, f"{u}_underlying_bars.parquet")

        chain = self._fetch_chain(u)
        symbols = self._slice_chain(chain)
        log.info(f"[Backfill] {len(symbols)} option symbols to fetch")

        for i in range(0, len(symbols), 100):
            batch = symbols[i:i+100]
            log.info(f"[Backfill] fetching option bars batch {i//100+1}/{(len(symbols)+99)//100} (n={len(batch)})")
            bars = self._fetch_option_bars(batch)
            self._write_parquet(bars, f"{u}_options_bars_{i//100:03d}.parquet")

    # ---------- HTTP helpers per endpoint ----------
    def _fetch_underlying_bars(self, symbol: str) -> List[Dict[str, Any]]:
        # /v2/stocks/bars
        params = dict(
            symbols=symbol,
            timeframe=self.cfg.bar_timeframe,
            start=self.cfg.start,
            end=self.cfg.end,
            limit=10_000,
            adjustment="raw",
        )
        data = self.h.get("/v2/stocks/bars", params)
        # Alpaca response shape:
        # { "bars": { "SPY": [ {t, o, h, l, c, v, ...}, ... ] } }
        bars = (data or {}).get("bars", {}).get(symbol, [])
        if not isinstance(bars, list):
            raise RuntimeError(f"Unexpected underlying bars format: {type(bars)}")
        return bars

    def _fetch_chain(self, underlying: str) -> Dict[str, Any]:
        # /v1beta3/options/snapshots/chains
        params = dict(underlying_symbol=underlying)
        data = self.h.get("/v1beta3/options/snapshots/chains", params)
        # Shape:
        # { "symbol": "SPY", "options": [ { "symbol": "...", "expiration_date": "...", "strike_price": ..., "type": "call"/"put" }, ... ] }
        return data

    def _slice_chain(self, chain: Dict[str, Any]) -> List[str]:
        opts = (chain or {}).get("options", [])
        if not isinstance(opts, list):
            raise RuntimeError("Unexpected chain format")
        # Filter by DTE & strikes around ATM
        # NOTE: At backfill time we don’t have spot here; keep broad slice.
        out: List[str] = []
        for o in opts:
            sym = o.get("symbol")
            if sym:
                out.append(sym)
        # Thin aggressively to cap dataset size
        k = max(1, self.cfg.strikes_around)
        return out[: k * 200]  # very rough throttle

    def _fetch_option_bars(self, option_symbols: Iterable[str]) -> List[Dict[str, Any]]:
        # /v1beta3/options/bars
        params = dict(
            symbols=",".join(option_symbols),
            timeframe=self.cfg.bar_timeframe,
            start=self.cfg.start,
            end=self.cfg.end,
            limit=10_000,
        )
        data = self.h.get("/v1beta3/options/bars", params)
        # Shape:
        # { "bars": { "O:SPY241122C00450000": [ {...}, ... ], "O:SPY..." : [ ... ] } }
        bars_by_sym = (data or {}).get("bars", {})
        rows: List[Dict[str, Any]] = []
        for sym, rows_list in bars_by_sym.items():
            if not isinstance(rows_list, list):
                continue
            for r in rows_list:
                r2 = dict(r)
                r2["symbol"] = sym
                rows.append(r2)
        return rows

    # ---------- Write helpers ----------
    def _write_parquet(self, rows: List[Dict[str, Any]], filename: str) -> None:
        import pandas as pd
        if not rows:
            log.warning(f"[Backfill] no rows to write for {filename}")
            return
        df = pd.DataFrame(rows)
        out = os.path.join(self.cfg.output_dir, filename)
        df.to_parquet(out, index=False)
        log.info(f"[Backfill] wrote {len(df):,} rows -> {out}")
