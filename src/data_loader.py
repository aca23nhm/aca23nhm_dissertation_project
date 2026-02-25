from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Dict


@dataclass(frozen=True)
class Item:
    """One sentence-level evaluation item."""
    sentence_id: str
    source: str
    reference: str


def load_dataset(path: str | Path) -> List[Item]:
    """
    Load a sentence-level dataset from CSV.

    Expected columns:
      - sentence_id
      - source
      - reference
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    items: List[Item] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"sentence_id", "source", "reference"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns {sorted(required_cols)}; "
                f"found {reader.fieldnames}"
            )

        for row in reader:
            sid = (row.get("sentence_id") or "").strip()
            src = (row.get("source") or "").strip()
            ref = (row.get("reference") or "").strip()

            # Basic sanity checks; you can relax these if needed.
            if not sid or not src:
                continue

            items.append(Item(sentence_id=sid, source=src, reference=ref))

    if not items:
        raise ValueError(f"No valid rows loaded from: {path}")

    return items


def sample_subset(items: Sequence[Item], n: int, seed: int = 42) -> List[Item]:
    """
    Return a fixed-size random subset (without replacement) in a reproducible way.

    If n >= len(items), returns all items (shuffled deterministically).
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    rng = random.Random(seed)
    items_list = list(items)

    if n >= len(items_list):
        rng.shuffle(items_list)
        return items_list

    # random.sample is deterministic given a deterministic RNG
    return rng.sample(items_list, n)


def get_by_ids(items: Sequence[Item], ids: Iterable[str]) -> List[Item]:
    """
    Return items matching the given sentence_ids, preserving the order of ids.
    Raises if an id is missing (helps catch dataset/version mistakes).
    """
    index: Dict[str, Item] = {it.sentence_id: it for it in items}
    selected: List[Item] = []

    missing = []
    for sid in ids:
        sid = sid.strip()
        if sid in index:
            selected.append(index[sid])
        else:
            missing.append(sid)

    if missing:
        raise KeyError(f"Missing sentence_ids in dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return selected