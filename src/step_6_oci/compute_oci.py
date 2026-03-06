from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IN_CSV = ROOT / "outputs" / "style_eval" / "per_sentence_style_metrics.csv"
OUT_DIR = ROOT / "outputs" / "oci_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SENTENCE_OUT = OUT_DIR / "per_sentence_oci.csv"
AGG_OUT = OUT_DIR / "aggregate_oci.csv"

# Freeze weights before experiment
ALPHA = 0.5   # edit distance
BETA = 0.25   # delta TTR
GAMMA = 0.25  # delta readability


def safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def min_max_normalise(value: float, min_val: float, max_val: float) -> float:
    """
    Min-max normalisation to [0, 1].
    If max == min, return 0.0 to avoid division by zero.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(
            f"Missing input file: {IN_CSV}\n"
            f"Run Step 5 first: python src/step_5_style_evaluation/evaluate_style.py"
        )

    with IN_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {IN_CSV}")

    # Extract the three OCI components from Step 5
    edit_vals = [safe_float(r["word_levenshtein"]) for r in rows]
    ttr_vals = [safe_float(r["delta_ttr"]) for r in rows]
    read_vals = [safe_float(r["delta_readability"]) for r in rows]

    edit_min, edit_max = min(edit_vals), max(edit_vals)
    ttr_min, ttr_max = min(ttr_vals), max(ttr_vals)
    read_min, read_max = min(read_vals), max(read_vals)

    print("Global min-max values used for normalisation:")
    print(f"  word_levenshtein: min={edit_min}, max={edit_max}")
    print(f"  delta_ttr:        min={ttr_min}, max={ttr_max}")
    print(f"  delta_readability:min={read_min}, max={read_max}")

    per_sentence_rows = []
    grouped_oci = defaultdict(list)

    for r in rows:
        edit_distance = safe_float(r["word_levenshtein"])
        delta_ttr = safe_float(r["delta_ttr"])
        delta_r = safe_float(r["delta_readability"])

        norm_edit = min_max_normalise(edit_distance, edit_min, edit_max)
        norm_ttr = min_max_normalise(delta_ttr, ttr_min, ttr_max)
        norm_r = min_max_normalise(delta_r, read_min, read_max)

        oci = (ALPHA * norm_edit) + (BETA * norm_ttr) + (GAMMA * norm_r)

        out_row = dict(r)
        out_row["norm_edit_distance"] = norm_edit
        out_row["norm_delta_ttr"] = norm_ttr
        out_row["norm_delta_readability"] = norm_r
        out_row["oci"] = oci

        per_sentence_rows.append(out_row)

        condition = r.get("condition", "unknown")
        grouped_oci[condition].append(oci)

    # Save per-sentence OCI
    fieldnames = list(per_sentence_rows[0].keys())
    with PER_SENTENCE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sentence_rows)

    # Save aggregate OCI per condition
    agg_rows = []
    for condition, ocis in sorted(grouped_oci.items()):
        agg_rows.append({
            "condition": condition,
            "n_sentences": len(ocis),
            "mean_oci": mean(ocis),
            "median_oci": median(ocis),
            "min_oci": min(ocis) if ocis else 0.0,
            "max_oci": max(ocis) if ocis else 0.0,
        })

    with AGG_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "n_sentences", "mean_oci", "median_oci", "min_oci", "max_oci"],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Saved per-sentence OCI: {PER_SENTENCE_OUT}")
    print(f"Saved aggregate OCI:    {AGG_OUT}")


if __name__ == "__main__":
    main()