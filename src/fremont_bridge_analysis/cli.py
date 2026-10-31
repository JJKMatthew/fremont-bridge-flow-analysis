"""Command-line entry points for repeatable project operations."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence

from fremont_bridge_analysis.config import load_project_config
from fremont_bridge_analysis.data.download import download_dataset
from fremont_bridge_analysis.data.prepare import prepare_data
from fremont_bridge_analysis.theme1.analysis import run_theme1
from fremont_bridge_analysis.theme2.analysis import run_theme2

CORE_MODULES = (
    "altair",
    "holidays",
    "marimo",
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "requests",
    "sklearn",
    "statsmodels",
)


def doctor() -> int:
    """Check configuration, directories, Python and core imports."""

    config = load_project_config()
    print(f"Project: {config.values['project']['title_zh']}")
    print(f"Root: {config.root}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Timezone: {config.timezone}")

    failures: list[str] = []
    for module_name in CORE_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {module_name}: {version}")
        except Exception as exc:  # pragma: no cover - diagnostic boundary
            failures.append(module_name)
            print(f"[FAIL] {module_name}: {exc}")

    for key in config.values["paths"]:
        path = config.path(key)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] path {key}: {path}")

    if failures:
        print(f"Environment check failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("Environment check passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fremont")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="check the local development environment")
    subparsers.add_parser("download", help="download and fingerprint the raw dataset")
    subparsers.add_parser("prepare", help="validate and prepare the raw dataset")
    subparsers.add_parser("theme1", help="run annual and monthly analysis")
    subparsers.add_parser("theme2", help="run weekly/daily PCA and clustering analysis")
    subparsers.add_parser("through-theme2", help="prepare data and run themes 1 and 2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "download":
        result = download_dataset(load_project_config())
        print(f"Downloaded: {result.data_path}")
        print(f"Metadata: {result.metadata_path}")
        return 0
    if args.command == "prepare":
        result = prepare_data(load_project_config())
        print(f"Processed data: {result.processed_path}")
        print(f"Quality summary: {result.quality_path}")
        return 0
    if args.command == "theme1":
        result = run_theme1(load_project_config())
        print(f"Theme 1 summary: {result.summary_path}")
        return 0
    if args.command == "theme2":
        result = run_theme2(load_project_config())
        print(f"Theme 2 summary: {result.summary_path}")
        return 0
    if args.command == "through-theme2":
        config = load_project_config()
        prepare_data(config)
        theme1 = run_theme1(config)
        theme2 = run_theme2(config)
        print(f"Theme 1 summary: {theme1.summary_path}")
        print(f"Theme 2 summary: {theme2.summary_path}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
