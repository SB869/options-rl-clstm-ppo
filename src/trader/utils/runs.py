from __future__ import annotations
import csv, os
from typing import Dict, Optional

INDEX_PATH = os.path.join("logs", "index.csv")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_run_index(row: Dict[str, str]):
    """Append (or create) a row to logs/index.csv with standard columns."""
    _ensure_dir("logs")
    header = ["run_id","mode","config","episodes","ckpt_dir","last_ckpt","run_dir","eval_json","eval_plot","start_time","end_time"]
    write_header = not os.path.exists(INDEX_PATH)
    with open(INDEX_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})

def write_latest_pointer(kind: str, mode: str, path: str):
    """
    kind: 'logs' or 'checkpoints'
    writes logs/<mode>/latest.txt or checkpoints/<mode>/latest.txt with absolute/relative path
    """
    base = os.path.join(kind, mode)
    _ensure_dir(base)
    with open(os.path.join(base, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(path.strip() + "\n")
