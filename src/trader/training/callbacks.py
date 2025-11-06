from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from trader.utils.runs import append_run_index, write_latest_pointer

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


@dataclass
class CheckpointCfg:
    base_dir: str = "checkpoints"
    name: str = "sim"
    keep: int = 3
    run_mode: str = "local"


class Callbacks:
    """
    Handles:
      - Run directories
      - CSV logging
      - TensorBoard logging (optional)
      - Checkpoints (rotated)
      - Post-train evaluation (eval.json + eval_plot.png in SAME run dir)
      - Run registry (logs/index.csv)
      - 'latest.txt' pointers for logs/ and checkpoints/
    """

    def __init__(
        self,
        ckpt: Optional[CheckpointCfg] = None,
        base_log_dir: str = "logs",
        run_mode: str = "local",
        auto_eval_episodes: int = 3,
        config_path: Optional[str] = None,
        enable_tb: bool = True,
    ):
        self.ckpt_cfg = ckpt or CheckpointCfg(run_mode=run_mode)
        self.base_log_dir = base_log_dir
        self.run_mode = run_mode
        self.auto_eval_episodes = int(auto_eval_episodes)
        self.config_path = config_path
        self.enable_tb = enable_tb and SummaryWriter is not None

        # Will be set in on_train_start()
        self.run_id: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.plot_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.ckpt_dir: Optional[str] = None
        self._csv_init_done = False
        self._start_ts: Optional[str] = None
        self._last_ckpt_path: Optional[str] = None

        # TensorBoard
        self.tb_writer: Optional[SummaryWriter] = None
        self.tb_dir: Optional[str] = None

    # ---------- lifecycle ----------

    def on_train_start(self, full_cfg: Dict[str, Any]):
        """Pass the FULL YAML cfg (with data/env/trainer) so auto-eval can rebuild the env."""
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Run + files
        self.run_dir = os.path.join(self.base_log_dir, self.run_mode, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "train.csv")
        self.plot_path = os.path.join(self.run_dir, "train_plot.png")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        # Checkpoints
        self.ckpt_dir = os.path.join(
            self.ckpt_cfg.base_dir, self.ckpt_cfg.run_mode, f"{self.ckpt_cfg.name}_{self.run_id}"
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # TensorBoard
        if self.enable_tb:
            self.tb_dir = os.path.join(self.run_dir, "tb")
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)
            try:
                self.tb_writer.add_text("config", f"```\n{json.dumps(full_cfg, indent=2)}\n```", 0)
            except Exception:
                pass

        # Meta snapshot
        self._start_ts = datetime.now().isoformat()
        meta = {
            "run_id": self.run_id,
            "mode": self.run_mode,
            "start_time": self._start_ts,
            "cfg": full_cfg,               # store full config snapshot
            "config_path": self.config_path,
            "ckpt_dir": self.ckpt_dir,
            "tb_dir": self.tb_dir,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Latest pointers
        write_latest_pointer("logs", self.run_mode, self.run_dir)
        write_latest_pointer("checkpoints", self.run_mode, self.ckpt_dir)

        self._csv_init_done = False

    def on_episode_end(self, ep: int, logs: Dict[str, Any], trainer) -> None:
        # CSV
        row = {"episode": int(ep)}
        row.update({k: float(v) for k, v in logs.items()})
        self._csv_write(row)

        # TensorBoard scalars
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/total_loss", logs.get("loss", float("nan")), ep)
            self.tb_writer.add_scalar("train/policy_loss", logs.get("policy_loss", float("nan")), ep)
            self.tb_writer.add_scalar("train/value_loss", logs.get("value_loss", float("nan")), ep)
            self.tb_writer.add_scalar("train/entropy", logs.get("entropy", float("nan")), ep)
            if "approx_kl" in logs:
                self.tb_writer.add_scalar("train/kl", logs["approx_kl"], ep)
            try:
                lr = trainer.opt.param_groups[0]["lr"]
                self.tb_writer.add_scalar("opt/lr", lr, ep)
            except Exception:
                pass

        # Checkpoint save (rotate)
        path = os.path.join(self.ckpt_dir, f"{self.ckpt_cfg.name}_ep{ep:05d}.pt")
        state = {
            "episode": ep,
            "feature": trainer.feature.state_dict(),
            "policy": trainer.policy.state_dict(),
            "opt": trainer.opt.state_dict(),
            "cfg_trainer": trainer.cfg,
            "run_id": self.run_id,
            "mode": self.run_mode,
        }
        tmp = path + ".tmp"
        trainer._torch_save(state, tmp)
        os.replace(tmp, path)
        self._last_ckpt_path = path

        # Prune old checkpoints
        files = sorted(
            f for f in os.listdir(self.ckpt_dir)
            if f.startswith(self.ckpt_cfg.name) and f.endswith(".pt")
        )
        if len(files) > self.ckpt_cfg.keep:
            for f in files[:-self.ckpt_cfg.keep]:
                try:
                    os.remove(os.path.join(self.ckpt_dir, f))
                except OSError:
                    pass

    def on_train_end(self, total_episodes: int):
        # Update meta
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {"run_id": self.run_id, "mode": self.run_mode}

        meta["end_time"] = datetime.now().isoformat()
        meta["episodes"] = int(total_episodes)
        meta["last_checkpoint"] = self._last_ckpt_path
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Auto-eval into SAME run dir
        eval_json, eval_plot = self._auto_eval_safe()

        # Log eval metrics to TensorBoard
        if self.tb_writer is not None and eval_json and os.path.exists(eval_json):
            try:
                with open(eval_json, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                for k, v in ev.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f"eval/{k}", v, total_episodes)
            except Exception:
                pass

        # Close TB
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

        # Global index row
        append_run_index({
            "run_id": self.run_id or "",
            "mode": self.run_mode,
            "config": self.config_path or "",
            "episodes": str(total_episodes),
            "ckpt_dir": self.ckpt_dir or "",
            "last_ckpt": self._last_ckpt_path or "",
            "run_dir": self.run_dir or "",
            "eval_json": eval_json or "",
            "eval_plot": eval_plot or "",
            "start_time": meta.get("start_time", ""),
            "end_time": meta.get("end_time", ""),
        })

    # ---------- helpers ----------

    def get_run_paths(self) -> Dict[str, Optional[str]]:
        return {
            "run_dir": self.run_dir,
            "csv": self.csv_path,
            "plot": self.plot_path,
            "meta": self.meta_path,
            "ckpt_dir": self.ckpt_dir,
            "run_id": self.run_id,
        }

    def _csv_write(self, row: Dict[str, Any]):
        """Append a row to train.csv (write header once)."""
        if not self.csv_path:
            return
        header = list(row.keys())
        mode = "a" if self._csv_init_done else "w"
        with open(self.csv_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not self._csv_init_done:
                w.writeheader()
                self._csv_init_done = True
            w.writerow(row)

    def _auto_eval_safe(self):
        """
        Programmatic evaluation; saves eval.json + eval_plot.png into run dir.
        Works even if matplotlib/pandas are missing (JSON still saved).
        """
        if not self._last_ckpt_path:
            return None, None
        try:
            # Lazy imports
            import torch, numpy as np, yaml
            from trader.metrics.evaluate import compute_metrics
            from trader.data.providers.sim import SimProvider
            from trader.env.options_env import OptionsTradingEnv
            from trader.models.feature_lstm import FeatureLSTM
            from trader.models.policy_lstm import RecurrentActorCritic

            # Load full cfg (prefer meta snapshot; fallback to config_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = meta.get("cfg", {})
            if "data" not in cfg and self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, "r") as cf:
                    cfg = yaml.safe_load(cf)

            # Rebuild env like training
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

            # Models (small by default for eval CPU safety)
            feature = FeatureLSTM(input_dim=obs_dim, hidden_dim=16)
            policy = RecurrentActorCritic(obs_embed_dim=16, action_dim=env.action_space.shape[0], hidden=16)

            # Load checkpoint
            state = torch.load(self._last_ckpt_path, map_location="cpu")
            feature.load_state_dict(state["feature"])
            policy.load_state_dict(state["policy"])

            # Deterministic eval
            def eval_episode():
                obs, _ = env.reset()
                equity = [env.nav]
                rewards, trades = [], []
                h1 = h2 = None
                done = False
                steps = 0
                while not done and steps < 10_000:
                    x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
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
            for _ in range(max(1, self.auto_eval_episodes)):
                r, eq, tr = eval_episode()
                R.extend(r); EQ.extend(eq); TR.extend(tr)

            metrics = compute_metrics(R, EQ, trade_sizes=TR)

            # Save JSON
            eval_json_path = os.path.join(self.run_dir, "eval.json")
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics.to_dict(), f, indent=2)

            # Try to save equity plot
            eval_plot_path = os.path.join(self.run_dir, "eval_plot.png")
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                eq = np.array(metrics.equity_curve, dtype=float)
                if len(eq) > 1:
                    plt.figure(figsize=(10, 5))
                    plt.plot(eq / (eq[0] + 1e-8) - 1.0)
                    plt.title(f"Equity Curve (normalized) — {self.run_id}")
                    plt.xlabel("Step"); plt.ylabel("Return")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout(); plt.savefig(eval_plot_path, dpi=150)
                    plt.close()
            except Exception:
                eval_plot_path = None

            # Update meta with eval pointers/metrics
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta2 = json.load(f)
                meta2["eval"] = metrics.to_dict()
                meta2["eval_json"] = eval_json_path
                meta2["eval_plot"] = eval_plot_path
                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta2, f, indent=2)
            except Exception:
                pass

            return eval_json_path, eval_plot_path
        except Exception as e:
            print(f"[WARN] Auto-eval failed: {e}")
            return None, None
