# src/trader/utils/env.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Mapping
from dotenv import load_dotenv

def load_env(dotenv_path: str | None = None) -> None:
    """
    Load .env once at process start. If dotenv_path is None, tries CWD and then the
    nearest parent containing pyproject.toml. Does NOT override already-set env vars.
    """
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
        return

    # Try current working directory first
    load_dotenv(override=False)

    # Try project root (walk up until pyproject.toml found)
    here = Path.cwd()
    for p in [here] + list(here.parents):
        if (p / "pyproject.toml").exists() and (p / ".env").exists():
            load_dotenv(p / ".env", override=False)
            break

def require_env(keys: list[str]) -> dict[str, str]:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
            + ". Create a .env file in project root and set them."
        )
    return {k: os.getenv(k, "") for k in keys}

def expand_env_in_obj(obj: Any) -> Any:
    """
    Recursively expand ${VAR} in strings found within dicts/lists.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_in_obj(v) for v in obj]
    return obj

# Backwards-compat alias for older imports
load_dotenv_if_present = load_env
