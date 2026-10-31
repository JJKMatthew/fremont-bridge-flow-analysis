"""Project configuration and path resolution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Validated access to the central TOML configuration."""

    root: Path
    values: dict[str, Any]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.values["dataset"]

    @property
    def split(self) -> dict[str, Any]:
        return self.values["split"]

    @property
    def timezone(self) -> str:
        return str(self.values["project"]["timezone"])

    def path(self, key: str) -> Path:
        return self.root / str(self.values["paths"][key])


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by locating pyproject.toml."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml from the current directory")


def load_project_config(root: Path | None = None) -> ProjectConfig:
    """Load config/project.toml from the repository root."""

    project_root = (root or find_project_root()).resolve()
    config_path = project_root / "config" / "project.toml"
    with config_path.open("rb") as handle:
        values = tomllib.load(handle)

    required_sections = {"project", "dataset", "split", "paths"}
    missing = required_sections.difference(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing configuration sections: {names}")

    return ProjectConfig(root=project_root, values=values)

