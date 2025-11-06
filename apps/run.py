# apps/run.py
from __future__ import annotations
import argparse
import json
import os
# Robust import so this works as a script or module
if __package__ is None or __package__ == "":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root
    from apps.train_cli import main as train_main
else:
    from .train_cli import main as train_main

from trader.utils.logging import get_logger
sys.path.insert(0, os.path.abspath("src"))

def main(config: str, mode: str):
    log = get_logger("run")
    # Train (auto-eval runs inside callbacks)
    train_main(config, mode, auto_eval_episodes=3)

    # Find latest run dir and print summary
    latest_ptr = os.path.join("logs", mode, "latest.txt")
    if not os.path.exists(latest_ptr):
        log.info("No latest run pointer found.")
        return

    with open(latest_ptr, "r", encoding="utf-8") as f:
        run_dir = f.read().strip()

    meta_path = os.path.join(run_dir, "meta.json")
    eval_json = os.path.join(run_dir, "eval.json")

    log.info(f"Latest run: {run_dir}")
    if os.path.exists(eval_json):
        with open(eval_json, "r", encoding="utf-8") as f:
            ev = json.load(f)
        log.info("Eval summary:\n" + json.dumps(ev, indent=2))
    else:
        log.info("No eval.json found (auto-eval may have been skipped).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/sim_spy_debug.yaml")
    ap.add_argument("--mode", choices=["local", "live"], default="local")
    args = ap.parse_args()
    main(args.config, args.mode)
