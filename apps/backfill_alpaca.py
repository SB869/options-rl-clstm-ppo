# apps/backfill_alpaca.py
from __future__ import annotations
import argparse
import yaml

from trader.utils.logging import get_logger
from trader.utils.env import load_env, expand_env_in_obj
from trader.data.providers.alpaca import AlpacaBackfill, BackfillConfig

def main(config_path: str):
    log = get_logger("backfill")
    load_env()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = expand_env_in_obj(raw)

    data = cfg.get("data", {})
    alp = cfg.get("alpaca", {})  # optional block for endpoints/creds

    backfill_cfg = BackfillConfig(
        underlying=data["symbols"][0] if isinstance(data.get("symbols"), list) else data.get("underlying", "SPY"),
        start=data["start"],
        end=data["end"],
        cache_dir=data.get("cache_dir", "data/options/alpaca"),
        timeframe=data.get("timeframe", "1Day"),
        feed=data.get("feed", "indicative"),
        page_limit_bars=int(data.get("page_limit_bars", 50000)),
        page_limit_chain=int(data.get("page_limit_chain", 1000)),
        data_base=alp.get("data_base", "https://data.alpaca.markets"),
        key_id=alp.get("key_id"),        # if None, falls back to env APCA_API_KEY_ID
        secret_key=alp.get("secret_key") # if None, falls back to env APCA_API_SECRET_KEY
    )

    AlpacaBackfill(backfill_cfg).run()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to configs/alpaca_*.yaml")
    args = ap.parse_args()
    main(args.config)
