# apps/backfill_alpaca.py
from __future__ import annotations
import argparse
import yaml

from trader.utils.env import load_env
load_env()  # load .env early

from trader.data.providers.alpaca import AlpacaBackfill, AlpacaConfig

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    ac = AlpacaConfig(
        symbols=d["symbols"],
        start=d["start"],
        end=d["end"],
        timeframe=d.get("timeframe", "1Min"),
        dte_min=int(d.get("dte_min", 7)),
        dte_max=int(d.get("dte_max", 30)),
        strikes_around=int(d.get("strikes_around", 5)),
        cache_dir=d.get("cache_dir", "data/alpaca"),
        backfill=True,
        fetch_iv=bool(d.get("fetch_iv", False)),
        batch_size=int(d.get("batch_size", 100)),
        rate_limit_sleep=float(d.get("rate_limit_sleep", 0.5)),
    )
    AlpacaBackfill(ac).run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="configs/alpaca_*.yaml")
    args = ap.parse_args()
    main(args.config)
