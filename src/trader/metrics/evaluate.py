from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class EvalMetrics:
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float
    avg_profit_per_trade: float  # APPT
    max_earning_rate: float      # MER (max single-step return)
    turnover: float              # sum |trades| / sum |position|+1e-8
    total_return: float
    equity_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "avg_profit_per_trade": self.avg_profit_per_trade,
            "max_earning_rate": self.max_earning_rate,
            "turnover": self.turnover,
            "total_return": self.total_return,
        }


def _max_drawdown(series: np.ndarray) -> float:
    peak = -np.inf
    mdd = 0.0
    for x in series:
        peak = max(peak, x)
        mdd = min(mdd, (x - peak) / (peak + 1e-8))
    return float(abs(mdd))


def compute_metrics(
    step_rewards: List[float],
    equity_curve: List[float],
    trade_sizes: List[float] | None = None,
) -> EvalMetrics:
    r = np.array(step_rewards, dtype=np.float64)
    eq = np.array(equity_curve, dtype=np.float64)

    # Per-step return proxy from equity curve
    eq_ret = np.diff(eq, prepend=eq[0]) / np.maximum(1e-8, eq)

    sharpe = 0.0
    if r.std() > 0:
        sharpe = float(np.sqrt(252) * r.mean() / (r.std() + 1e-8))

    downside = r.copy()
    downside[downside > 0] = 0.0
    denom = np.sqrt((downside ** 2).mean()) + 1e-8
    sortino = float(np.sqrt(252) * max(0.0, r.mean()) / denom) if denom > 0 else 0.0

    maxdd = _max_drawdown(eq)

    hits = (r > 0).sum()
    hit_rate = float(hits / max(1, len(r)))

    appt = float(r.mean()) if len(r) else 0.0
    mer = float(np.max(r)) if len(r) else 0.0

    if trade_sizes is not None and len(trade_sizes) == len(r):
        turnover = float(np.sum(np.abs(trade_sizes)) / (np.sum(np.abs(eq_ret)) + 1e-8))
    else:
        turnover = float(np.sum(np.abs(eq_ret)))

    total_ret = float((eq[-1] - eq[0]) / (eq[0] + 1e-8)) if len(eq) > 1 else 0.0

    return EvalMetrics(
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=maxdd,
        hit_rate=hit_rate,
        avg_profit_per_trade=appt,
        max_earning_rate=mer,
        turnover=turnover,
        total_return=total_ret,
        equity_curve=list(map(float, eq)),
    )
