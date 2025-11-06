# src/trader/utils/env.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

def load_env(dotenv_path: str | None = None) -> None:
    """
    Load .env once at process start. If dotenv_path is None,
    we search project root (directory containing pyproject.toml)
    and current working directory.
    """
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return

    # Try current working directory
    load_dotenv(override=False)

    # Try project root (walk up to find pyproject.toml)
    here = Path.cwd()
    for p in [here] + list(here.parents):
        cand = p / ".env"
        if cand.exists():
            load_dotenv(dotenv_path=str(cand), override=False)
            break

def require_env(keys: list[str]) -> dict[str, str]:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}. "
                           f"Create a .env file in project root or export them.")
    return {k: os.getenv(k, "") for k in keys}
