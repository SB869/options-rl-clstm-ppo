# src/trader/data/providers/synth.py
from __future__ import annotations
import glob
import os
from typing import Iterator, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


class SynthProvider:
    """
    Parquet-backed provider for synthetic options data written by
    apps/simulate_options_from_spy.py.

    Directory layout expected:
      <cache_dir>/<SYMBOL>/
        <SYMBOL>_chain_*.parquet                (optional but used for selection)
        <SYMBOL>_*_bars_<start>_<end>.parquet   (one per contract)
    Underlying (spot) minute bars are looked up from either:
      <cache_dir>/<SYMBOL>/<SYMBOL>_underlying_*.parquet
      or fallback:
      data/alpaca/<SYMBOL>/<SYMBOL>_underlying_*.parquet
    """

    def __init__(self, cfg_data: dict):
        self.cache_dir: str = cfg_data.get("cache_dir", "data/synth")
        self.symbols: List[str] = [str(s) for s in cfg_data.get("symbols", [])]
        if not self.symbols:
            raise ValueError("SynthProvider requires data.symbols (e.g., ['SPY']).")

        # internal streaming state
        self._sym: str = self.symbols[0]
        self._df: Optional[pd.DataFrame] = None
        self._i: int = 0

    # ---------- Discovery helpers ----------

    def _sym_dir(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, symbol)

    def _glob_latest(self, pattern: str) -> Optional[str]:
        hits = sorted(glob.glob(pattern))
        return hits[-1] if hits else None

    def chain_path(self, symbol: str) -> Optional[str]:
        return self._glob_latest(os.path.join(self._sym_dir(symbol), f"{symbol}_chain_*.parquet"))

    def latest_underlying_path(self, symbol: str) -> Optional[str]:
        # first try inside cache_dir
        p = self._glob_latest(os.path.join(self._sym_dir(symbol), f"{symbol}_underlying_*.parquet"))
        if p:
            return p
        # fallback to alpaca cache
        return self._glob_latest(os.path.join("data", "alpaca", symbol, f"{symbol}_underlying_*.parquet"))

    def iter_contract_bars(self, symbol: str) -> Iterator[str]:
        pat = os.path.join(self._sym_dir(symbol), f"{symbol}_*_bars_*.parquet")
        yield from glob.iglob(pat)

    # ---------- IO ----------

    def read_parquet(self, path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    # ---------- Selection & featurization ----------

    def _pick_contract_file(self, symbol: str, spot0: float) -> str:
        """
        Pick a contract parquet to stream.
        Preference: CALL, DTE ~14 (closest), strike closest to ATM.
        Falls back to the first bars file if chain is missing.
        """
        chain_p = self.chain_path(symbol)
        if chain_p:
            ch = pd.read_parquet(chain_p).copy()

            # Normalize columns
            if "right" in ch.columns:
                ch["right"] = ch["right"].astype(str).str.lower()
            else:
                ch["right"] = "call"  # default bias

            if "strike" not in ch.columns:
                raise RuntimeError("Chain parquet missing 'strike' column.")

            # Ensure expiry exists and is datetime64[ns, tz] in UTC
            if "expiry" in ch.columns:
                if not is_datetime64_any_dtype(ch["expiry"]):
                    ch["expiry"] = pd.to_datetime(ch["expiry"], utc=True, errors="coerce")
                # If tz-aware but not UTC, convert; if tz-naive, localize to UTC
                ch["expiry"] = ch["expiry"].dt.tz_convert("UTC")
            else:
                # If no expiry, make a dummy far date to neutralize DTE scoring
                ch["expiry"] = pd.Timestamp("2100-01-01", tz="UTC")

            # Scoring: CALL first, then DTE closeness to 14 days, then ATM-ness
            now_utc = pd.Timestamp.now(tz="UTC").normalize()
            dte_days = (ch["expiry"].dt.normalize() - now_utc).dt.days.abs()
            atm = (ch["strike"].astype(float) - float(spot0)).abs()

            # CALLs get lower base score (preferred)
            call_penalty = (ch["right"] != "call").astype(int) * 1_000_000
            score = call_penalty + (dte_days - 14).abs() * 1_000 + atm

            pick = ch.loc[score.idxmin()]
            occ = str(pick["symbol"])
            pat = os.path.join(self._sym_dir(symbol), f"{symbol}_{occ}_bars_*.parquet")
            f = self._glob_latest(pat)
            if f:
                return f

        # fallback: first bars file
        any_bars = sorted(self.iter_contract_bars(symbol))
        if not any_bars:
            raise RuntimeError(f"No synthetic bars found in {self._sym_dir(symbol)}")
        return any_bars[0]

    def _make_frame(self, symbol: str) -> pd.DataFrame:
        """
        Load underlying + chosen contract bars and align on timestamp.
        Build features to a fixed 8-D vector per row.
        """
        # Load underlying (spot)
        u_p = self.latest_underlying_path(symbol)
        if not u_p:
            raise RuntimeError(
                f"Underlying parquet not found for {symbol} (looked in {self._sym_dir(symbol)} and data/alpaca/{symbol})."
            )
        u = self.read_parquet(u_p)
        ts_col = "timestamp" if "timestamp" in u.columns else ("time" if "time" in u.columns else None)
        if ts_col is None:
            u = u.reset_index().rename(columns={u.index.name: "timestamp"})
            ts_col = "timestamp"
        u = u.rename(columns={ts_col: "timestamp"})
        u["timestamp"] = pd.to_datetime(u["timestamp"], utc=True)
        u = u[["timestamp", "close"]].rename(columns={"close": "spot_close"}).dropna().sort_values("timestamp")

        spot0 = float(u["spot_close"].iloc[0])

        # Pick contract bars file
        bar_file = self._pick_contract_file(symbol, spot0)
        b = self.read_parquet(bar_file)
        if "timestamp" not in b.columns:
            if "time" in b.columns:
                b = b.rename(columns={"time": "timestamp"})
            else:
                b = b.reset_index().rename(columns={b.index.name: "timestamp"})
        b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
        b = b[["timestamp", "open", "high", "low", "close", "volume"]].dropna().sort_values("timestamp")

        # Align on minute timestamps
        df = pd.merge_asof(
            u.sort_values("timestamp"), b.sort_values("timestamp"),
            on="timestamp", direction="nearest", tolerance=pd.Timedelta("1min")
        ).dropna()

        # Compute auxiliary columns
        df["spot_ret_1m"] = df["spot_close"].pct_change().fillna(0.0)
        df["opt_ret_1m"] = df["close"].pct_change().fillna(0.0)
        df["spot_vol_30m"] = df["spot_ret_1m"].rolling(30, min_periods=5).std().fillna(0.0)

        # Extract strike from OCC part in filename (…_<OCC>_bars_…)
        base = os.path.basename(bar_file)
        try:
            occ_part = base.split("_bars_")[0].split("_")[-1]  # e.g., SPY240808C00469930
            K1000 = int(occ_part[-8:])
            K = K1000 / 1000.0
        except Exception:
            K = float(spot0)
        df["moneyness"] = (K / df["spot_close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # DTE proxy from total span of file (kept simple for synth)
        total_minutes = max(len(df), 1)
        approx_days = max(total_minutes / (252 * 390), 1e-6)
        df["dte_norm"] = np.clip(approx_days / 30.0, 0.0, 1.0)

        # normalized option price relative to spot for env._option_price_from_obs
        df["opt_over_spot"] = (df["close"] / df["spot_close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # log-volume normalization
        df["vol_norm"] = np.log1p(df["volume"]).astype(float) / 10.0

        # Final 8-D observation
        obs_cols = [
            "spot_ret_1m",
            "spot_vol_30m",
            "opt_ret_1m",
            "moneyness",
            "dte_norm",
            "spot_close",     # raw spot; env uses it for price denorm
            "opt_over_spot",  # env uses x[6]*S
            "vol_norm",
        ]
        df = df[["timestamp"] + obs_cols].copy().dropna().reset_index(drop=True)
        df["x_vec"] = df[obs_cols].astype(np.float32).values.tolist()
        return df

    # ---------- Gym-facing lifecycle ----------

    def reset(self) -> Dict[str, Any]:
        """
        Prepare a streaming dataframe and return the initial observation dict:
          {"x": np.ndarray(8,), "S": float, "done": False}
        """
        self._sym = self.symbols[0]
        self._df = self._make_frame(self._sym)
        self._i = 0

        if self._df.empty:
            raise RuntimeError("SynthProvider: aligned dataframe is empty after merge.")

        row = self._df.iloc[self._i]
        x = np.asarray(row["x_vec"], dtype=np.float32)
        S = float(row["spot_close"])
        return {"x": x, "S": S, "done": False}

    def step(self) -> Dict[str, Any]:
        """
        Advance one minute and return:
          {"x": np.ndarray(8,), "S": float, "done": bool}
        """
        if self._df is None:
            raise RuntimeError("SynthProvider.step() called before reset().")
        self._i += 1
        if self._i >= len(self._df):
            self._i = len(self._df) - 1
            row = self._df.iloc[self._i]
            return {
                "x": np.asarray(row["x_vec"], dtype=np.float32),
                "S": float(row["spot_close"]),
                "done": True,
            }

        row = self._df.iloc[self._i]
        x = np.asarray(row["x_vec"], dtype=np.float32)
        S = float(row["spot_close"])
        return {"x": x, "S": S, "done": False}
