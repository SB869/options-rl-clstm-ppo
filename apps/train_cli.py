# apps/train_cli.py
from __future__ import annotations

import argparse
import json
import yaml
import torch

from trader.utils.env import load_env
from trader.utils.logging import get_logger
from trader.utils.seed import set_global_seed
from trader.env.options_env import OptionsTradingEnv
from trader.models.feature_lstm import FeatureLSTM
from trader.models.policy_lstm import RecurrentActorCritic
from trader.training.ppo import PPOTrainer
from trader.training.callbacks import Callbacks, CheckpointCfg

# Load .env before anything that might read credentials/paths
load_env()


def _make_provider(cfg: dict):
    """
    Create a data provider based on cfg['data']['provider'].
    Lazily imports provider modules so we don't import Alpaca on non-Alpaca runs.
    """
    data_cfg = cfg.get("data") or {}
    env_cfg = cfg.get("env") or {}
    prov_name = str(data_cfg.get("provider", "sim")).lower()

    if prov_name == "alpaca":
        from trader.data.providers.alpaca import AlpacaProvider  # lazy import
        return AlpacaProvider(data_cfg)

    if prov_name in ("synth", "parquet"):
        from trader.data.providers.synth import SynthProvider  # lazy import
        return SynthProvider(data_cfg)

    # default: sim (synthetic-in-memory market)
    if prov_name == "sim":
        from trader.data.providers.sim import SimProvider  # lazy import
        return SimProvider(
            symbols=data_cfg["symbols"],
            days=data_cfg["days"],
            option_kind=env_cfg["option"]["kind"],
            dte_start=env_cfg["option"]["dte_start"],
            seed=cfg.get("seed", 42),
        )

    raise ValueError(f"Unknown data provider: {prov_name}")


def main(config_path: str, mode: str, auto_eval_episodes: int = 3, enable_tb: bool = True):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    log = get_logger()
    set_global_seed(cfg.get("seed", 42))
    log.info(f"Loaded config: {config_path} | mode={mode}")

    # Select device
    dev_cfg = (cfg.get("trainer") or {}).get("device", "cuda")
    use_cuda = torch.cuda.is_available() and dev_cfg == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    log.info(f"Using device: {device}")

    # --- Provider selection (lazy imports inside) ---
    provider = _make_provider(cfg)
    prov_name = str((cfg.get("data") or {}).get("provider", "sim")).lower()

    # --- Environment ---
    env = OptionsTradingEnv(
        provider=provider,
        costs=cfg["env"]["costs"],
        turbulence_cfg=cfg["env"]["turbulence"],
        max_positions=cfg["env"]["max_positions"],
    )

    # NOTE: Instead of provider.observation_spec(), ask the env for the true obs dim.
    obs_dim = int(env.observation_space.shape[-1])

    # --- Model sizing ---
    hid = 128 if device.type == "cuda" else 16
    feature = FeatureLSTM(input_dim=obs_dim, hidden_dim=hid).to(device)
    policy = RecurrentActorCritic(
        obs_embed_dim=hid,
        action_dim=env.action_space.shape[0],
        hidden=hid,
    ).to(device)

    # --- Callbacks (TB, auto-eval, registry) ---
    run_tag = prov_name  # tag checkpoints/logs by provider
    callbacks = Callbacks(
        ckpt=CheckpointCfg(base_dir="checkpoints", name=run_tag, keep=3, run_mode=mode),
        base_log_dir="logs",
        run_mode=mode,
        auto_eval_episodes=auto_eval_episodes,
        config_path=config_path,
        enable_tb=enable_tb,
    )
    callbacks.on_train_start(cfg)

    # Save model sizes for eval
    paths = callbacks.get_run_paths()
    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["model"] = {
            "hidden": hid,
            "obs_dim": obs_dim,
            "action_dim": int(env.action_space.shape[0]),
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    amp_dtype = (cfg.get("trainer", {}).get("amp_dtype", "bfloat16"))
    trainer = PPOTrainer(
        env=env,
        feature=feature,
        policy=policy,
        cfg=cfg["trainer"],
        callbacks=callbacks,
        device=device,
        amp_dtype=amp_dtype,
    )
    trainer.train(episodes=cfg["trainer"]["episodes"])

    paths = callbacks.get_run_paths()
    log.info(
        "✅ Training complete.\n"
        f"   Run dir:   {paths['run_dir']}\n"
        f"   CSV:       {paths['csv']}\n"
        f"   Train PNG: {paths['plot']}\n"
        f"   Meta JSON: {paths['meta']}\n"
        f"   Ckpts dir: {paths['ckpt_dir']}\n"
        "   (Auto-eval saved eval.json / eval_plot.png alongside the run)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/sim_spy_debug.yaml")
    ap.add_argument("--mode", choices=["local", "live"], default="local")
    ap.add_argument("--eval-episodes", type=int, default=3, help="Auto-eval episodes after training")
    ap.add_argument("--no-tb", action="store_true", help="Disable TensorBoard logging")
    args = ap.parse_args()
    main(args.config, args.mode, args.eval_episodes, enable_tb=not args.no_tb)
