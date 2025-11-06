from __future__ import annotations
import argparse, os, json
from datetime import datetime

import torch
import yaml
import numpy as np

from trader.utils.logging import get_logger
from trader.utils.seed import set_global_seed
from trader.data.providers.sim import SimProvider
from trader.env.options_env import OptionsTradingEnv
from trader.models.feature_lstm import FeatureLSTM
from trader.models.policy_lstm import RecurrentActorCritic
from trader.metrics.evaluate import compute_metrics


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main(config_path: str, checkpoint: str, episodes: int, mode: str):
    log = get_logger("eval")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_global_seed(cfg.get("seed", 42))

    # Build env like training
    provider = SimProvider(
        symbols=cfg["data"]["symbols"],
        days=cfg["data"]["days"],
        option_kind=cfg["env"]["option"]["kind"],
        dte_start=cfg["env"]["option"]["dte_start"],
        seed=cfg.get("seed", 42),
    )
    env = OptionsTradingEnv(
        provider=provider,
        costs=cfg["env"]["costs"],
        turbulence_cfg=cfg["env"]["turbulence"],
        max_positions=cfg["env"]["max_positions"],
    )
    obs_dim = provider.observation_spec()["feature_dim"]

    # Small models (match your train_cli defaults)
    feature = FeatureLSTM(input_dim=obs_dim, hidden_dim=16)
    policy = RecurrentActorCritic(obs_embed_dim=16, action_dim=env.action_space.shape[0], hidden=16)

    # Load checkpoint
    state = torch.load(checkpoint, map_location="cpu")
    feature.load_state_dict(state["feature"])
    policy.load_state_dict(state["policy"])
    log.info(f"Loaded checkpoint: {checkpoint}")

    # Deterministic eval (use mu, not sampling)
    def eval_episode():
        obs, _ = env.reset()
        equity = [env.nav]
        rewards = []
        trades = []
        h_feat = None
        h_pol = None
        done = False
        steps = 0
        while not done and steps < 10_000:
            x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
            z, h_feat = feature(x, h_feat)
            mu, std, v, h_pol = policy(z, h_pol)

            a = torch.tanh(mu).detach().view(-1).numpy()  # deterministic: action = tanh(mu)
            obs, r, done, _, info = env.step(a)
            rewards.append(float(r))
            equity.append(float(info["nav"]))
            trades.append(float(info["trade"]))
            steps += 1

        return rewards, equity, trades

    all_rewards, all_equity, all_trades = [], [], []
    for ep in range(episodes):
        r, eq, tr = eval_episode()
        all_rewards.extend(r)
        all_equity.extend(eq)
        all_trades.extend(tr)

    metrics = compute_metrics(all_rewards, all_equity, trade_sizes=all_trades)

    # Output dirs
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("logs", mode, f"eval_{stamp}")
    _ensure_dir(out_dir)

    # Save JSON
    out_json = os.path.join(out_dir, "eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    log.info(f"✅ Saved eval metrics → {out_json}")

    # Plot equity curve to PNG (optional dependency)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        eq = np.array(metrics.equity_curve, dtype=float)
        plt.figure(figsize=(10, 5))
        plt.plot(eq / (eq[0] + 1e-8) - 1.0)
        plt.title("Equity Curve (normalized)")
        plt.xlabel("Step")
        plt.ylabel("Return")
        plt.grid(True, linestyle="--", alpha=0.5)
        out_png = os.path.join(out_dir, "eval_plot.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        log.info(f"✅ Saved eval plot → {out_png}")
    except Exception as e:
        log.info(f"[INFO] Skipping eval plot (matplotlib/pandas not available): {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/sim_spy_debug.yaml")
    ap.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint .pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--mode", choices=["local","live"], default="local")
    args = ap.parse_args()
    main(args.config, args.checkpoint, args.episodes, args.mode)
