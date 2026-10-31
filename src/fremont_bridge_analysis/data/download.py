"""Download a reproducible raw-data snapshot from Seattle Open Data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from fremont_bridge_analysis.config import ProjectConfig


@dataclass(frozen=True)
class DownloadResult:
    data_path: Path
    metadata_path: Path


def download_dataset(config: ProjectConfig, *, timeout_seconds: int = 120) -> DownloadResult:
    """Download the configured CSV and write provenance metadata beside it."""

    raw_dir = config.path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_path = raw_dir / str(config.dataset["raw_filename"])
    temporary_path = data_path.with_suffix(data_path.suffix + ".part")

    sha256 = hashlib.sha256()
    total_bytes = 0
    response = requests.get(
        str(config.dataset["download_url"]),
        timeout=timeout_seconds,
        stream=True,
    )
    response.raise_for_status()

    with temporary_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            sha256.update(chunk)
            total_bytes += len(chunk)
    temporary_path.replace(data_path)

    metadata_path = data_path.with_suffix(".metadata.json")
    metadata = {
        "dataset_id": config.dataset["id"],
        "dataset_name": config.dataset["name"],
        "landing_page": config.dataset["landing_page"],
        "download_url": config.dataset["download_url"],
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "bytes": total_bytes,
        "sha256": sha256.hexdigest(),
        "content_type": response.headers.get("content-type"),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return DownloadResult(data_path=data_path, metadata_path=metadata_path)

