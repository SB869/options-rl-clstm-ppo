from __future__ import annotations
import numpy as np
from scipy.spatial.distance import mahalanobis

class TurbulenceGate:
    """
    Rolling Mahalanobis-distance turbulence gate.
    When current return vector's distance > rolling pct-threshold, returns True (risk-off).
    """
    def __init__(self, window: int = 60, pct: float = 0.9, enabled: bool = True):
        self.window = int(window)
        self.pct = float(pct)
        self.enabled = bool(enabled)
        self._hist: list[np.ndarray] = []
        self._thresh: float | None = None

    def reset(self):
        self._hist.clear()
        self._thresh = None

    def update(self, ret_vec: np.ndarray) -> bool:
        if not self.enabled:
            return False
        ret_vec = np.asarray(ret_vec, dtype=np.float32).ravel()
        self._hist.append(ret_vec)
        if len(self._hist) < self.window:
            return False

        X = np.asarray(self._hist[-self.window:], dtype=np.float32)  # (W, D)
        mu = X.mean(axis=0)
        cov = np.cov(X.T) if X.shape[1] > 1 else np.array([[X.var() + 1e-6]], dtype=np.float32)
        cov = cov + 1e-6 * np.eye(cov.shape[0], dtype=np.float32)
        inv = np.linalg.pinv(cov)

        d = float(mahalanobis(ret_vec, mu, inv))

        # (Re)estimate rolling threshold lazily
        if self._thresh is None:
            ds = [float(mahalanobis(x, mu, inv)) for x in X]
            self._thresh = float(np.quantile(ds, self.pct))

        return d > self._thresh
