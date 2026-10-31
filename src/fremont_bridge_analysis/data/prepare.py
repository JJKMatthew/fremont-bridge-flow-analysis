"""Prepare the immutable raw CSV for all three analysis themes."""

from __future__ import annotations

from calendar import isleap
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fremont_bridge_analysis.analysis_utils import write_json
from fremont_bridge_analysis.config import ProjectConfig


STANDARD_COLUMNS = {
    "Date": "timestamp_text",
    "Fremont Bridge Sidewalks, south of N 34th St Total": "total",
    "Fremont Bridge Sidewalks, south of N 34th St Cyclist West Sidewalk": "west_sidewalk",
    "Fremont Bridge Sidewalks, south of N 34th St Cyclist East Sidewalk": "east_sidewalk",
}


@dataclass(frozen=True)
class PreparationResult:
    processed_path: Path
    quality_path: Path
    year_coverage_path: Path


def _expected_hours(year: int) -> int:
    return 8784 if isleap(year) else 8760


def prepare_data(config: ProjectConfig) -> PreparationResult:
    """Standardize schema, preserve missingness and create quality artifacts."""

    raw_path = config.path("raw") / str(config.dataset["raw_filename"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}. Run `fremont download` first.")

    frame = pl.read_csv(raw_path, infer_schema_length=10_000)
    missing_columns = set(STANDARD_COLUMNS).difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw dataset is missing expected columns: {names}")

    frame = (
        frame.rename(STANDARD_COLUMNS)
        .with_columns(
            pl.col("timestamp_text")
            .str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=True)
            .alias("timestamp_local")
        )
        .sort("timestamp_local")
        .with_columns(
            pl.col("timestamp_local").dt.date().alias("date"),
            pl.col("timestamp_local").dt.year().alias("year"),
            pl.col("timestamp_local").dt.month().alias("month"),
            pl.col("timestamp_local").dt.weekday().alias("weekday_iso"),
            pl.col("timestamp_local").dt.hour().alias("hour"),
            pl.any_horizontal(
                pl.col("total").is_null(),
                pl.col("west_sidewalk").is_null(),
                pl.col("east_sidewalk").is_null(),
            ).alias("is_missing"),
        )
        .with_columns((pl.col("weekday_iso") >= 6).alias("is_weekend"))
    )

    duplicate_timestamps = int(frame.select(pl.col("timestamp_local").is_duplicated().sum()).item())
    negative_counts = int(
        frame.select(
            pl.sum_horizontal(
                pl.col("total").lt(0).sum(),
                pl.col("west_sidewalk").lt(0).sum(),
                pl.col("east_sidewalk").lt(0).sum(),
            )
        ).item()
    )
    nonnull = frame.filter(
        pl.all_horizontal(
            pl.col("total").is_not_null(),
            pl.col("west_sidewalk").is_not_null(),
            pl.col("east_sidewalk").is_not_null(),
        )
    )
    total_mismatches = int(
        nonnull.select(
            (pl.col("total") != pl.col("west_sidewalk") + pl.col("east_sidewalk")).sum()
        ).item()
    )

    minimum, maximum = frame.select(
        pl.min("timestamp_local").alias("minimum"),
        pl.max("timestamp_local").alias("maximum"),
    ).row(0)
    expected_timestamps = pl.DataFrame(
        {
            "timestamp_local": pl.datetime_range(
                minimum,
                maximum,
                interval="1h",
                closed="both",
                eager=True,
            )
        }
    )
    absent_timestamps = expected_timestamps.join(
        frame.select("timestamp_local"), on="timestamp_local", how="anti"
    )

    processed_dir = config.path("processed")
    tables_dir = config.path("tables")
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "fremont_hourly.parquet"
    frame.write_parquet(processed_path, compression="zstd")

    missing_rows_path = tables_dir / "data_missing_value_rows.csv"
    frame.filter(pl.col("is_missing")).write_csv(missing_rows_path)
    absent_timestamps_path = tables_dir / "data_absent_timestamps.csv"
    absent_timestamps.write_csv(absent_timestamps_path)

    year_coverage = (
        frame.group_by("year")
        .agg(
            pl.len().alias("row_count"),
            pl.col("total").count().alias("valid_total_hours"),
            pl.col("total").null_count().alias("null_total_hours"),
            pl.min("timestamp_local").alias("first_timestamp"),
            pl.max("timestamp_local").alias("last_timestamp"),
        )
        .sort("year")
        .with_columns(
            pl.col("year")
            .map_elements(_expected_hours, return_dtype=pl.Int64)
            .alias("expected_full_year_hours")
        )
        .with_columns(
            (pl.col("row_count") / pl.col("expected_full_year_hours")).alias(
                "full_year_row_coverage"
            ),
            (pl.col("valid_total_hours") / pl.col("expected_full_year_hours")).alias(
                "full_year_value_coverage"
            ),
        )
    )
    year_coverage_path = tables_dir / "data_year_coverage.csv"
    year_coverage.write_csv(year_coverage_path)

    quality_path = tables_dir / "data_quality_summary.json"
    write_json(
        quality_path,
        {
            "row_count": frame.height,
            "column_count": frame.width,
            "minimum_timestamp": minimum,
            "maximum_timestamp": maximum,
            "timezone_interpretation": config.timezone,
            "duplicate_timestamps": duplicate_timestamps,
            "absent_hourly_timestamps": absent_timestamps.height,
            "rows_with_missing_counts": int(frame.select(pl.col("is_missing").sum()).item()),
            "negative_count_values": negative_counts,
            "total_not_equal_to_sidewalk_sum": total_mismatches,
            "processed_path": processed_path,
            "missing_rows_path": missing_rows_path,
            "absent_timestamps_path": absent_timestamps_path,
        },
    )
    return PreparationResult(
        processed_path=processed_path,
        quality_path=quality_path,
        year_coverage_path=year_coverage_path,
    )
