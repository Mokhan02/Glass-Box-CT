"""Every results table carries `dataset`, `in_distribution` and `n_scans`. The flag is derived from
the registry, never passed in by hand, so a row cannot claim to be held-out by mistake."""
from __future__ import annotations

import csv
from pathlib import Path

from glassbox.data.registry import is_in_distribution

REQUIRED = ("dataset", "in_distribution", "n_scans")


class ResultsTable:
    def __init__(self, columns: list[str]):
        extra = [c for c in columns if c not in REQUIRED]
        self.columns = list(REQUIRED) + extra
        self.rows: list[dict] = []

    def add(self, dataset: str, n_scans: int, **values) -> None:
        reserved = set(values) & set(REQUIRED)
        if reserved:
            raise KeyError(f"reserved columns are set by the table: {sorted(reserved)}")
        unknown = set(values) - set(self.columns)
        if unknown:
            raise KeyError(f"unknown columns: {sorted(unknown)}")
        if n_scans <= 0:
            raise ValueError("n_scans must be positive")
        self.rows.append(
            {"dataset": dataset, "in_distribution": is_in_distribution(dataset), "n_scans": n_scans, **values}
        )

    def held_out(self) -> list[dict]:
        return [r for r in self.rows if not r["in_distribution"]]

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            w.writerows(self.rows)
