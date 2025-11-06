from __future__ import annotations
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from trader.utils.runs import append_run_index, write_latest_pointer

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # optional


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

        self.run_id: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.plot_path: Optional[str] = None
        self.meta_path: Optional[str] = None
        self.ckpt_dir: Optional[str] = None
        self._csv_init_done = False
        self._start_ts: Optional[str] = None
        self._last_ckpt_path: Optional[str] = None

        self.tb_writer: Optional[SummaryWriter] = None
        self.tb_dir: Optional[str] = None

    # ---------- lifecycle ----------

    def on_train_start(self, full_cfg: Dict[str, Any]):
        """full_cfg must contain data/env/trainer keys (the whole YAML)."""
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_log_dir, self.run_mode, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "train.csv")
        self.plot_path = os.path.join(self.run_dir, "train_plot.png")
        self.meta_path = os.path.join(self.run_dir, "meta.json")

        self.ckpt_dir = os.path.join(self.ckpt_cfg.base_dir, self.ckpt_cfg.run_mode, f"{self.ckpt_cfg.name}_{self.run_id}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # TensorBoard
        if self.enable_tb:
            self.tb_dir = os.path.join(self.run_dir, "tb")
            os.makedirs(self.tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=self.tb_dir)
            # store a text snapshot of config for convenience
            self.tb_writer.add_text("config", f"```\n{json.dumps(full_cfg, indent=2)}\n```", global_step=0)

        self._start_ts = datetime.now().isoformat()
        meta = {
            "run_id": self.run_id,
            "mode": self.run_mode,
            "start_time": self._start_ts,
            "cfg": full_cfg,               # full config snapshot
            "config_path": self.config_path,
            "ckpt_dir": self.ckpt_dir,
            "tb_dir": self.tb_dir,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # latest pointers
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
            # learning rate (first param group)
            try:
                lr = trainer.opt.param_groups[0]["lr"]
                self.tb_writer.add_scalar("opt/lr", lr, ep)
            except Exception:
                pass

        # checkpoint save
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

        # prune
        files = sorted(f for f in os.listdir(self.ckpt_dir) if f.startswith(self.ckpt_cfg.name) and f.endswith(".pt"))
        if len(files) > self.ckpt_cfg.keep:
            for f in files[:-self.ckpt_cfg.keep]:
                try:
                    os.remove(os.path.join(self.ckpt_dir, f))
                except OSError:
                    pass

    def on_train_end(self, total_episodes: int):
        # update meta
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

        # auto-eval (writes eval.json/eval_plot.png into the run dir)
        eval_json, eval_plot = self._auto_eval_safe()

        # log eval metrics to TB
        if self.tb_writer is not None and eval_json and os.path.exists(eval_json):
            try:
                with open(eval_json, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                for k, v in ev.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f"eval/{k}", v, total_episodes)
            except Exception:
                pass

        # close TB
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

        # global index
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
            "start_time": meta.get("start_time",""),
            "end_time": meta.get("end_time",""),
        })

    # --- helpers (unchanged except omitted here for brevity) ---
    # keep your existing get_run_paths, _csv_write, _auto_eval_safe methods from earlier version
    # (they already live in your file; we didn't change their internals aside from TB lines)
