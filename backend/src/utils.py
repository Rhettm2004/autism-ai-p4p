"""Shared utilities: logger, CSV I/O."""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger with timestamp, level, and module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_benchmark(csv_path: str) -> list[dict]:
    """Load a benchmark CSV and return its rows as a list of dicts."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {csv_path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_results(
    results: list[dict],
    model_id: str,
    condition: str,
    output_dir: str,
    tag: str = "",
) -> str:
    """
    Save results to a timestamped CSV in output_dir.

    Filename format: phase1_{model_id}_{condition}[_{tag}]_{YYYYMMDD_HHMMSS}.csv
    The tag separates run types on disk, so a baseline and a RAG run of the same
    model and condition cannot be confused with one another.
    Raises FileExistsError if that filename already exists.
    Returns the path of the written file.
    """
    if not results:
        raise ValueError("results list is empty; nothing to save")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"phase1_{model_id}_{condition}{suffix}_{timestamp}.csv"
    out_path = Path(output_dir) / filename

    if out_path.exists():
        raise FileExistsError(f"Output file already exists: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return str(out_path)
