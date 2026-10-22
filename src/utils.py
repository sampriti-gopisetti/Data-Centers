"""Shared helpers."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (keys already in the environment take precedence)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
