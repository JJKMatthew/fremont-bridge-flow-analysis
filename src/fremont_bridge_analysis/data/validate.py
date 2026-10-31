"""Schema and coverage checks implemented after the first raw snapshot is inspected."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationSummary:
    """Stable interface for the upcoming P1.2 data-quality module."""

    row_count: int
    column_count: int
    duplicate_timestamps: int
    missing_values: int

