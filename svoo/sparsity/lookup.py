#!/usr/bin/env python3
"""Lookup utilities for Step,Layer,Head,Sparsity CSV profiles."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SparsityLookup:
    """In-memory lookup table for per-step, per-layer, per-head sparsity."""

    def __init__(self, csv_path: str):
        """Load a Step,Layer,Head,Sparsity CSV profile."""
        self.lookup_dict: Dict[Tuple[int, int, int], float] = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        """Load CSV rows into the lookup dictionary."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Sparsity CSV file not found: {csv_path}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["Step"])
                layer = int(row["Layer"])
                head = int(row["Head"])
                sparsity = float(row["Sparsity"])
                self.lookup_dict[(step, layer, head)] = sparsity

        print(f"Loaded {len(self.lookup_dict)} sparsity entries from {csv_path}")

    def get_sparsity(self, step: int, layer: int, head: int) -> Optional[float]:
        """Return one sparsity value, or None if it is missing."""
        key = (step, layer, head)
        return self.lookup_dict.get(key)

    def get_sparsity_batch(self, step: int, layer: int, num_heads: int) -> List[float]:
        """Return sparsity values for all heads in one step/layer."""
        return [self.get_sparsity(step, layer, head) or 0.0 for head in range(num_heads)]

    def has_data_for_step_layer(self, step: int, layer: int, num_heads: int = 12) -> bool:
        """Return True only when every expected head exists for this step/layer."""
        return all(self.get_sparsity(step, layer, head) is not None for head in range(num_heads))
