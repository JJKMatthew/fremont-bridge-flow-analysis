"""Shared numerical and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Return median/MAD z-scores, falling back to standard deviation."""

    array = np.asarray(values, dtype=float)
    median = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - median))
    if np.isfinite(mad) and mad > 0:
        return 0.67448975 * (array - median) / mad
    standard_deviation = np.nanstd(array)
    if np.isfinite(standard_deviation) and standard_deviation > 0:
        return (array - np.nanmean(array)) / standard_deviation
    return np.zeros_like(array)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
