# apps/eval_latest.py
from __future__ import annotations
import os, json, glob, argparse
import torch
import numpy as np

from trader.data.providers.sim import SimProvider
from trader.env.options_env import OptionsTradingEnv
from trader.models.feature_lstm import FeatureLSTM
from trader.models.policy_lstm import RecurrentActorCritic
from trader.metrics.evaluate import compute_metrics

def infer_hidden_from_state_dict(sd: dict) -> int:
    keys = [k for k in sd.keys() if k.endswith("lstm.weight_hh_l0")]
    for k in keys:
        _, H = sd[k].shape
        if H > 0:
            return int(H)
    for k, v in sd.items():
        if "lstm.weight_hh_l0" in k:
            return int(v.shape[1])
    raise RuntimeError("Could not infer hidden size from checkpoint")

def main(mode: str = "live", episodes: int = 3):
    latest_run_ptr = os.path.join("logs", mode, "latest.txt")
    if not os.path.exists(latest_run_ptr):
        print("No latest run pointer found.")
        return 1
    run_dir = open(latest_run_ptr, "r", encoding="utf-8").read().strip()

    latest_ckpt_dir_ptr = os.path.join("checkpoints", mode, "latest.txt")
    if not os.path.exists(latest_ckpt_dir_ptr):
        print("No latest checkpoints pointer found.")
        return 1
    ckpt_dir = open(latest_ckpt_dir_ptr, "r", encoding="utf-8").read().strip()
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpts:
        print("No checkpoints found in", ckpt_dir)
        return 1
    ckpt_path = ckpts[-1]
    print("Using checkpoint:", ckpt_path)

    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = meta.get("cfg", {})
    model_cfg = meta.get("model", {})

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

    state = torch.load(ckpt_path, map_location="cpu")
    hid_feat = infer_hidden_from_state_dict(state["feature"])
    hid_pol  = infer_hidden_from_state_dict(state["policy"])
    hidden = hid_pol if hid_pol != hid_feat else hid_feat

    obs_dim = int(model_cfg.get("obs_dim", provider.observation_spec()["feature_dim"]))
    act_dim = int(model_cfg.get("action_dim", env.action_space.shape[0]))

    feature = FeatureLSTM(input_dim=obs_dim, hidden_dim=hidden)
    policy  = RecurrentActorCritic(obs_embed_dim=hidden, action_dim=act_dim, hidden=hidden)
    feature.load_state_dict(state["feature"])
    policy.load_state_dict(state["policy"])

    def eval_episode():
        obs, _ = env.reset()
        equity = [env.nav]
        rewards, trades = [], []
        h1 = h2 = None
        done = False
        steps = 0
        while not done and steps < 10000:
            x = torch.tensor(obs, dtype=torch.float32).view(1,1,-1)
            z, h1 = feature(x, h1)
            mu, std, v, h2 = policy(z, h2)
            a = torch.tanh(mu).detach().view(-1).numpy()
            obs, r, done, _, info = env.step(a)
            rewards.append(float(r))
            equity.append(float(info["nav"]))
            trades.append(float(info["trade"]))
            steps += 1
        return rewards, equity, trades

    R, EQ, TR = [], [], []
    for _ in range(max(1, episodes)):
        r, eq, tr = eval_episode()
        R.extend(r); EQ.extend(eq); TR.extend(tr)

    metrics = compute_metrics(R, EQ, trade_sizes=TR)

    eval_json_path = os.path.join(run_dir, "eval.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print("Wrote:", eval_json_path)

    try:
        import matplotlib.pyplot as plt
        eq = np.array(metrics.equity_curve, dtype=float)
        if len(eq) > 1:
            eval_plot_path = os.path.join(run_dir, "eval_plot.png")
            plt.figure(figsize=(10,5))
            base = eq[0] + 1e-8
            plt.plot(eq / base - 1.0)
            plt.title(f"Equity Curve (normalized) — {os.path.basename(run_dir)}")
            plt.xlabel("Step"); plt.ylabel("Return")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout(); plt.savefig(eval_plot_path, dpi=150); plt.close()
            print("Wrote:", eval_plot_path)
    except Exception as e:
        print("[WARN] Could not render eval_plot.png:", e)

    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local","live"], default="live")
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    raise SystemExit(main(args.mode, args.episodes))
