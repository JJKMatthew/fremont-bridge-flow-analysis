from pathlib import Path

from fremont_bridge_analysis.config import load_project_config


def test_project_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_project_config(root)

    assert config.dataset["id"] == "65db-xm6k"
    assert config.timezone == "America/Los_Angeles"
    assert config.split["training_end"] == "2025-12-31"
    assert config.split["test_start"] == "2026-01-01"


def test_configured_paths_stay_inside_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_project_config(root)

    for key in config.values["paths"]:
        assert config.path(key).is_relative_to(root)
