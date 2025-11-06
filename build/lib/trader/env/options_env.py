from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trader.constants import CONTRACT_MULTIPLIER
from trader.env.risk import TurbulenceGate


class OptionsTradingEnv(gym.Env):
    """
    Single-instrument options-like environment.

    - Observation: provider returns an 8-D vector (see SimProvider) + metadata.
    - Action: continuous a ∈ [-1, 1], mapped to target contracts ∈ [-max_positions, +max_positions].
    - Costs: commission per contract + spread slippage (bps of notional).
    - Reward: (ΔPortfolio − costs) * scale  -> implemented as (ΔNAV) * scale since costs debit cash.
    - Turbulence: Mahalanobis@pct → sets target to 0 (risk-off) for that step.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        provider,
        costs: dict,
        turbulence_cfg: dict,
        max_positions: int = 10,
    ):
        super().__init__()
        self.provider = provider

        # paper reward scaling
        self.scale = float(costs.get("scale", 1e-4))
        self.commission = float(costs.get("commission_per_contract", 0.65))
        self.spread_bps = float(costs.get("spread_bps", 15.0))

        self.max_positions = int(max_positions)
        self.turbulence = TurbulenceGate(**turbulence_cfg)

        # Action/obs spaces (matches SimProvider feature_dim=8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

        # state
        self.position = 0.0   # contracts
        self.cash = 1_000_000.0
        self.nav = self.cash
        self.prev_opt_price = 5.0

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _option_price_from_obs(x: np.ndarray, S: float) -> float:
        """Denormalize option price: x[6] holds (option_price / S)."""
        return float(max(0.0, x[6] * max(1e-6, S)))

    # ---- gym API -------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        data = self.provider.reset()
        obs = np.asarray(data["x"], dtype=np.float32)
        S = float(data.get("S", 100.0))

        self.position = 0.0
        self.cash = 1_000_000.0
        self.nav = self.cash
        self.prev_opt_price = self._option_price_from_obs(obs, S)
        self.turbulence.reset()

        return obs, {}

    def step(self, action):
        # sanitize action
        a = np.asarray(action, dtype=np.float32).ravel()
        a = float(np.clip(a[0], -1.0, 1.0))

        data = self.provider.step()
        obs = np.asarray(data["x"], dtype=np.float32)
        terminated = bool(data.get("done", False))
        truncated = False
        S = float(data.get("S", 100.0))

        # current synthetic option price from obs
        opt_price = self._option_price_from_obs(obs, S)

        # one-dimensional return vector for turbulence gate
        opt_ret = (opt_price - self.prev_opt_price) / max(1e-6, self.prev_opt_price)
        risk_off = self.turbulence.update(np.array([opt_ret], dtype=np.float32))

        # target position from action (risk-off forces flat)
        target_pos = 0.0 if risk_off else (a * self.max_positions)
        target_pos = float(np.clip(target_pos, -self.max_positions, self.max_positions))
        trade = target_pos - self.position

        # trade costs
        notional = abs(trade) * opt_price * CONTRACT_MULTIPLIER
        commission = abs(trade) * self.commission
        slippage = (self.spread_bps * 1e-4) * notional
        costs = commission + slippage

        # execute: update cash/position
        self.cash -= trade * opt_price * CONTRACT_MULTIPLIER
        self.cash -= costs
        self.position += trade

        # mark-to-market; reward is ΔNAV * scale (costs already debited)
        pv = self.cash + self.position * opt_price * CONTRACT_MULTIPLIER
        d_nav = pv - self.nav
        reward = float(d_nav * self.scale)

        # update state
        self.nav = pv
        self.prev_opt_price = opt_price

        info = {
            "risk_off": risk_off,
            "opt_price": opt_price,
            "position": self.position,
            "nav": self.nav,
            "trade": trade,
            "costs": costs,
        }
        return obs, reward, terminated, truncated, info
