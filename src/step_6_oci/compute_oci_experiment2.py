from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IN_CSV = ROOT / "outputs" / "experiment_2" / "style_eval" / "per_sentence_style_metrics.csv"
OUT_DIR = ROOT / "outputs" / "experiment_2" / "oci_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SENTENCE_OUT = OUT_DIR / "per_sentence_oci.csv"
AGG_OUT = OUT_DIR / "aggregate_oci.csv"

W_EDIT = 0.35
W_DENSITY = 0.15
W_TTR = 0.20
W_READABILITY = 0.15
W_COSINE = 0.15


def safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def min_max_normalise(value: float, min_val: float, max_val: float) -> float:
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
            f"Missing input file: {IN_CSV}\nRun style evaluation first: python src/step_5_style_evaluation/evaluate_style_experiment2.py"
        )

    with IN_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {IN_CSV}")

    edit_vals = [safe_float(r["word_levenshtein"]) for r in rows]
    density_vals = [safe_float(r["edit_density"]) for r in rows]
    ttr_vals = [safe_float(r["delta_ttr"]) for r in rows]
    read_vals = [safe_float(r["delta_readability"]) for r in rows]
    cosine_distance_vals = [max(0.0, min(1.0, 1.0 - safe_float(r["stylometric_cosine"]))) for r in rows]

    edit_min, edit_max = min(edit_vals), max(edit_vals)
    density_min, density_max = min(density_vals), max(density_vals)
    ttr_min, ttr_max = min(ttr_vals), max(ttr_vals)
    read_min, read_max = min(read_vals), max(read_vals)
    cosine_min, cosine_max = min(cosine_distance_vals), max(cosine_distance_vals)

    print("Global min-max values used for normalisation:")
    print(f"  word_levenshtein: min={edit_min}, max={edit_max}")
    print(f"  edit_density:     min={density_min}, max={density_max}")
    print(f"  delta_ttr:        min={ttr_min}, max={ttr_max}")
    print(f"  delta_readability:min={read_min}, max={read_max}")
    print(f"  1-cosine:         min={cosine_min}, max={cosine_max}")

    per_sentence_rows = []
    grouped_oci = defaultdict(list)

    for r in rows:
        edit_distance = safe_float(r["word_levenshtein"])
        edit_density = safe_float(r["edit_density"])
        delta_ttr = safe_float(r["delta_ttr"])
        delta_r = safe_float(r["delta_readability"])
        cos_sim = safe_float(r["stylometric_cosine"])
        cosine_distance = max(0.0, min(1.0, 1.0 - cos_sim))

        norm_edit = min_max_normalise(edit_distance, edit_min, edit_max)
        norm_density = min_max_normalise(edit_density, density_min, density_max)
        norm_ttr = min_max_normalise(delta_ttr, ttr_min, ttr_max)
        norm_r = min_max_normalise(delta_r, read_min, read_max)
        norm_cosine_distance = cosine_distance

        oci = (
            W_EDIT * norm_edit
            + W_DENSITY * norm_density
            + W_TTR * norm_ttr
            + W_READABILITY * norm_r
            + W_COSINE * norm_cosine_distance
        )
        oci_percent = oci * 100

        out_row = dict(r)
        out_row["norm_edit_distance"] = norm_edit
        out_row["norm_edit_density"] = norm_density
        out_row["norm_delta_ttr"] = norm_ttr
        out_row["norm_delta_readability"] = norm_r
        out_row["norm_1_minus_cosine"] = norm_cosine_distance
        out_row["oci"] = oci
        out_row["oci_percent"] = oci_percent

        per_sentence_rows.append(out_row)
        grouped_oci[r.get("condition", "unknown")].append(oci)

    with PER_SENTENCE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sentence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sentence_rows)

    agg_rows = []
    for condition, ocis in sorted(grouped_oci.items()):
        mean_oci_val = mean(ocis)
        median_oci_val = median(ocis)
        agg_rows.append({
            "condition": condition,
            "n_sentences": len(ocis),
            "mean_oci": mean_oci_val,
            "mean_oci_percent": mean_oci_val * 100,
            "median_oci": median_oci_val,
            "median_oci_percent": median_oci_val * 100,
            "min_oci": min(ocis) if ocis else 0.0,
            "max_oci": max(ocis) if ocis else 0.0,
        })

    with AGG_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "condition", "n_sentences",
            "mean_oci", "mean_oci_percent",
            "median_oci", "median_oci_percent",
            "min_oci", "max_oci",
        ])
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Saved per-sentence OCI: {PER_SENTENCE_OUT}")
    print(f"Saved aggregate OCI:    {AGG_OUT}")


if __name__ == "__main__":
    main()