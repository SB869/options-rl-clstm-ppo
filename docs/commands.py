python apps/backfill_alpaca.py configs/alpaca_spy.yaml


Make sure you have at least one underlying parquet: data/alpaca/SPY/SPY_underlying_2024-08-01_2024-11-01.parquet
Generate synthetic options: python apps/simulate_options_from_spy.py configs/synth_spy.yaml
Train against the synthetic cache: python apps/train_cli.py --config configs/train_synth.yaml