from trader.data.providers.sim import SimProvider
from trader.env.options_env import OptionsTradingEnv
from trader.models.feature_lstm import FeatureLSTM
from trader.models.policy_lstm import RecurrentActorCritic
from trader.training.ppo import PPOTrainer

def test_train_smoke_runs():
    p = SimProvider(symbols=["SPY"], days=20, option_kind="call", dte_start=30, seed=123)
    env = OptionsTradingEnv(
        provider=p,
        costs={"commission_per_contract": 0.65, "spread_bps": 15, "scale": 1e-4},
        turbulence_cfg={"window": 10, "pct": 0.9, "enabled": False},
        max_positions=5,
    )
    feature = FeatureLSTM(input_dim=8, hidden_dim=16)
    policy = RecurrentActorCritic(obs_embed_dim=16, action_dim=env.action_space.shape[0], hidden=16)
    trainer = PPOTrainer(
        env=env,
        feature=feature,
        policy=policy,
        cfg={
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "lr": 3e-4,
            "batch_steps": 128,   # short rollout
        },
    )
    # Should complete without exceptions
    trainer.train(episodes=2)
