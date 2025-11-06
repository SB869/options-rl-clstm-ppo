# apps/backfill_alpaca.py
from __future__ import annotations
import argparse, yaml, traceback
from trader.utils.logging import get_logger
from trader.data.providers.alpaca import BackfillConfig, AlpacaBackfill

def main(config_path: str):
    log = get_logger("backfill")
    with open(config_path, "r") as f:
        c = yaml.safe_load(f)
    ac = BackfillConfig(
        underlying=c["underlying"],
        start=c["start"],
        end=c["end"],
        output_dir=c["output_dir"],
        strikes_around=c.get("strikes_around", 10),
        dte_min=c.get("dte_min", 3),
        dte_max=c.get("dte_max", 45),
        bar_timeframe=c.get("bar_timeframe", "1Day"),
    )
    try:
        AlpacaBackfill(ac).run()
    except Exception as e:
        log.error(f"Backfill failed: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    main(args.config)
