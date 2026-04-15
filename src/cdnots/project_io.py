from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical filesystem layout for project data and outputs."""

    root: Path

    @property
    def raw_data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def results_tables_dir(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def results_figures_dir(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def discovery_output_dir(self) -> Path:
        return self.root / "discovery2" / "outputs"

    def ensure_standard_dirs(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_tables_dir.mkdir(parents=True, exist_ok=True)
        self.results_figures_dir.mkdir(parents=True, exist_ok=True)
        self.discovery_output_dir.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Create a stream logger with consistent script formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
