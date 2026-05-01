"""Shared Layer 1 paths and defaults."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCORED_DIR = DATA_DIR / "scored"
MUTATED_DIR = DATA_DIR / "mutated"
REPORTS_DIR = DATA_DIR / "reports"
DEPRECATED_DIR = DATA_DIR / "deprecated"

CRYPTO10K_FILENAME = "crypto_10k_tweets_(2021_2022Nov).csv"
CRYPTO10K_RAW_PATH = RAW_DIR / CRYPTO10K_FILENAME
CRYPTO10K_LEGACY_PATH = DATA_DIR / CRYPTO10K_FILENAME

DEFAULT_WINDOW_FREQ = "5min"
DEFAULT_MIN_TWEETS_PER_WINDOW = 20


def resolve_crypto10k_path(path: str | Path | None = None) -> Path:
    """Return an existing crypto10k path, preferring the new Layer 1 layout."""
    if path is not None:
        return Path(path)
    if CRYPTO10K_RAW_PATH.exists():
        return CRYPTO10K_RAW_PATH
    return CRYPTO10K_LEGACY_PATH


def ensure_layer1_dirs(base_dir: Path = DATA_DIR) -> None:
    """Create the local Layer 1 data directories if they do not exist."""
    for directory in (
        base_dir / "raw",
        base_dir / "processed",
        base_dir / "scored",
        base_dir / "mutated" / "sentiment",
        base_dir / "mutated" / "temporal",
        base_dir / "mutated" / "adversarial",
        base_dir / "reports",
        base_dir / "deprecated",
    ):
        directory.mkdir(parents=True, exist_ok=True)
