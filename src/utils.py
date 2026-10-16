"""Shared helpers."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
