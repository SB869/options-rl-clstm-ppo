from trader.data.providers.sim import SimProvider
from trader.env.options_env import OptionsTradingEnv

def test_env_step_reward_scaled_and_shapes():
    p = SimProvider(symbols=["SPY"], days=12, option_kind="call", dte_start=30, seed=123)
    env = OptionsTradingEnv(
        provider=p,
        costs={"commission_per_contract": 0.65, "spread_bps": 15, "scale": 1e-4},
        turbulence_cfg={"window": 10, "pct": 0.9, "enabled": True},
        max_positions=5,
    )
    obs, info = env.reset()
    assert obs.shape == (8,)
    for _ in range(10):
        obs, r, term, trunc, info = env.step([0.5])  # mid-long action
        # reward should be nicely scaled around 0 (rarely exceeding 1 in magnitude)
        assert abs(r) < 1.0
        assert obs.shape == (8,)
        assert isinstance(term, bool) and isinstance(trunc, bool)
