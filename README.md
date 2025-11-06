# trading-rl (v0.1)

A clean, testable scaffold for a cascaded LSTM PPO options trader based on
the CLSTM-PPO paper (cascaded LSTM feature extractor + recurrent LSTM policy) with paper-style reward scaling and turbulence.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python apps/train_cli.py --config configs/sim_spy_debug.yaml
pytest -q
