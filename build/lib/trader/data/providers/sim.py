from __future__ import annotations
import math
from typing import Dict, Any, List

import numpy as np
from scipy.stats import norm

# --- tiny helpers -------------------------------------------------------------

def _rsi(series: np.ndarray, window: int = 14) -> np.ndarray:
    """Simple RSI for flavor (not strictly needed but useful signal)."""
    series = np.asarray(series, dtype=np.float32)
    delta = np.diff(series, prepend=series[0])
    up = np.clip(delta, 0, None)
    down = -np.clip(delta, None, 0)
    roll_up = np.convolve(up, np.ones(window) / window, mode="same")
    roll_down = np.convolve(down, np.ones(window) / window, mode="same") + 1e-8
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # normalize to [-1, 1]
    return (rsi - 50.0) / 50.0


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(series, dtype=np.float32)
    out[0] = series[0]
    for t in range(1, len(series)):
        out[t] = alpha * series[t] + (1 - alpha) * out[t - 1]
    return out


def _macd(series: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    e_fast = _ema(series, fast)
    e_slow = _ema(series, slow)
    macd = e_fast - e_slow
    # scale to roughly [-1,1]
    denom = np.maximum(1e-6, np.std(macd) * 6.0)
    return (macd - np.mean(macd)) / denom


# --- Black–Scholes pricing / greeks ------------------------------------------

def _bs_d1(S: float, K: float, T: float, sigma: float, r: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _bs_price(S: float, K: float, T: float, sigma: float, r: float, kind: str) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return max(0.0, (S - K) if kind == "call" else (K - S))
    d1 = _bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S: float, K: float, T: float, sigma: float, r: float, kind: str) -> float:
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 1.0 if (kind == "call" and S > K) else (-1.0 if (kind == "put" and S < K) else 0.0)
    d1 = _bs_d1(S, K, T, sigma, r)
    return norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0


def _bs_theta_per_day(S: float, K: float, T: float, sigma: float, r: float, kind: str) -> float:
    """Return *per calendar day* theta (approx)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    d1 = _bs_d1(S, K, T, sigma, r)
    d2 = d1 - sigma * math.sqrt(T)
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * d1 * d1)
    # classic theta (per year); convert to per day by /365
    first = - (S * pdf * sigma) / (2.0 * math.sqrt(T))
    if kind == "call":
        second = - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        second = + r * K * math.exp(-r * T) * norm.cdf(-d2)
    theta_annual = first + second
    return theta_annual / 365.0


# --- Provider -----------------------------------------------------------------

class SimProvider:
    """
    Fast, deterministic option simulator:

    - Underlying follows GBM with annual mu/vol.
    - IV follows AR(1) with mean reversion and noise.
    - Single option (call/put) with decaying DTE, strike near ATM.
    - Features (D=8):
        [ret, iv_chg, rsi_norm, macd_norm, delta, theta_norm, opt_price_norm, dte_norm]

    Returned dicts:
        reset() -> {"x": feat_t, "S": float, "K": float, "iv": float, "dte": float, "opt_price": float}
        step()  -> {"x": feat_t+1, "done": bool, ... same meta keys ...}
    """

    def __init__(
        self,
        symbols: List[str],
        days: int = 40,
        option_kind: str = "call",
        dte_start: int = 30,
        seed: int | None = 123,
        r_annual: float = 0.01,
        mu_annual: float = 0.06,
        vol_annual: float = 0.20,
    ):
        self.symbols = symbols
        self.days = int(days)
        self.kind = option_kind.lower()
        assert self.kind in {"call", "put"}, "option_kind must be 'call' or 'put'"
        self.dte0 = int(dte_start)
        self.rng = np.random.default_rng(seed)

        self.r_annual = float(r_annual)
        self.mu_annual = float(mu_annual)
        self.vol_annual = float(vol_annual)

        # internal state
        self.t = 0
        self._S = None
        self._IV = None
        self._DTE = None
        self._K = None
        self._opt_price = None
        self._features = None

    # public API ---------------------------------------------------------------

    def observation_spec(self) -> Dict[str, Any]:
        return {"feature_dim": 8}

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        self._simulate_paths()
        return self._pack_obs(self.t)

    def step(self) -> Dict[str, Any]:
        self.t += 1
        done = self.t >= (self.days - 1)
        out = self._pack_obs(min(self.t, self.days - 1))
        out["done"] = done
        return out

    # internals ----------------------------------------------------------------

    def _simulate_paths(self):
        # ---- underlying GBM path ----
        dt = 1.0 / 252.0
        S = np.empty(self.days, dtype=np.float32)
        S[0] = 100.0
        for i in range(1, self.days):
            z = self.rng.normal()
            drift = (self.mu_annual - 0.5 * self.vol_annual ** 2) * dt
            shock = self.vol_annual * math.sqrt(dt) * z
            S[i] = S[i - 1] * math.exp(drift + shock)
        self._S = S

        # ---- implied vol AR(1) ----
        IV = np.empty(self.days, dtype=np.float32)
        IV[0] = 0.25
        for i in range(1, self.days):
            IV[i] = 0.90 * IV[i - 1] + 0.10 * self.rng.normal(0.25, 0.05)
            IV[i] = float(np.clip(IV[i], 0.05, 1.50))
        self._IV = IV

        # ---- DTE decays linearly to floor 5 days ----
        DTE = np.linspace(self.dte0, max(5, self.dte0 // 6), self.days, dtype=np.float32)
        self._DTE = np.maximum(1.0, DTE)

        # ---- strike ~ ATM at start (can be extended to skew) ----
        self._K = np.full(self.days, S[0], dtype=np.float32)

        # ---- BS price + greeks per step ----
        opt = np.empty(self.days, dtype=np.float32)
        delta = np.empty(self.days, dtype=np.float32)
        theta_day = np.empty(self.days, dtype=np.float32)
        for i in range(self.days):
            T = self._DTE[i] / 252.0
            opt[i] = _bs_price(float(S[i]), float(self._K[i]), float(T), float(IV[i]), self.r_annual, self.kind)
            delta[i] = _bs_delta(float(S[i]), float(self._K[i]), float(T), float(IV[i]), self.r_annual, self.kind)
            theta_day[i] = _bs_theta_per_day(float(S[i]), float(self._K[i]), float(T), float(IV[i]), self.r_annual, self.kind)

        self._opt_price = opt

        # ---- features ----
        # returns & indicators from S
        ret = np.diff(S, prepend=S[0]) / np.maximum(1e-6, S)
        rsi = _rsi(S, window=14)
        macd = _macd(S, fast=12, slow=26)

        iv_chg = np.diff(IV, prepend=IV[0])

        # normalize some quantities
        theta_norm = theta_day / (np.std(theta_day) + 1e-6)      # ~[-1,1]
        opt_norm = opt / np.maximum(1e-6, S)                     # option price relative to S
        dte_norm = self._DTE / (self._DTE[0] + 1e-6)

        self._features = np.stack(
            [ret, iv_chg, rsi, macd, delta, theta_norm, opt_norm, dte_norm],
            axis=1,
        ).astype(np.float32)

    def _pack_obs(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self._features[idx],
            "S": float(self._S[idx]),
            "K": float(self._K[idx]),
            "iv": float(self._IV[idx]),
            "dte": float(self._DTE[idx]),
            "opt_price": float(self._opt_price[idx]),
        }
