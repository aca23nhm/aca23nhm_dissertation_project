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
W_EDIT = 0.35        # Edit Distance (primary signal)
W_DENSITY = 0.15     # Edit Density (length-normalised)
W_TTR = 0.20         # TTR Difference (lexical change)
W_READABILITY = 0.15 # Readability difference (structure)
W_COSINE = 0.15      # Cosine distance (overall stylometric similarity)
W_FLUENCY_UTILITY = 0.50
W_MEANING_UTILITY = 0.50


def safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def min_max_normalise(value: float, min_val: float, max_val: float) -> float:
    """Min-max normalisation to [0, 1]."""
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def fluency_score(text: str) -> float:
    stripped = text.strip()
    words = [token.strip(".,!?;:\"'()[]{}").lower() for token in stripped.split()]
    words = [word for word in words if word]
    if not words:
        return 0.0
    repeated = sum(1 for left, right in zip(words, words[1:]) if left == right)
    repeated_penalty = repeated / max(1, len(words) - 1)
    punctuation_density = sum(1 for ch in stripped if ch in ".,!?;:\"'()[]{}") / max(1, len(words))
    punctuation_penalty = max(0.0, punctuation_density - 0.35)
    avg_word_len = sum(len(word) for word in words) / len(words)
    word_length_penalty = max(0.0, abs(avg_word_len - 5.0) / 20.0)
    sentence_len_penalty = min(0.25, max(0, len(words) - 40) / 120.0)
    terminal_penalty = 0.0 if stripped[-1:] in {".", "!", "?", '"', "'"} else 0.05
    return clip01(
        1.0
        - 0.35 * repeated_penalty
        - 0.20 * punctuation_penalty
        - 0.10 * word_length_penalty
        - 0.10 * sentence_len_penalty
        - terminal_penalty
    )


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

    # Extract the OCI components from Step 5
    edit_vals = [safe_float(r["word_levenshtein"]) for r in rows]
    density_vals = [safe_float(r["edit_density"]) for r in rows]
    ttr_vals = [safe_float(r["delta_ttr"]) for r in rows]
    read_vals = [safe_float(r["delta_readability"]) for r in rows]
    fluency_vals = [
        safe_float(r["delta_fluency"]) if "delta_fluency" in r and r["delta_fluency"] != ""
        else fluency_score(r.get("output", "")) - fluency_score(r.get("source", ""))
        for r in rows
    ]
    cosine_distance_vals = [clip01(1.0 - safe_float(r["stylometric_cosine"])) for r in rows]

    edit_min, edit_max = min(edit_vals), max(edit_vals)
    density_min, density_max = min(density_vals), max(density_vals)
    ttr_min, ttr_max = min(ttr_vals), max(ttr_vals)
    read_min, read_max = min(read_vals), max(read_vals)
    fluency_min, fluency_max = min(fluency_vals), max(fluency_vals)
    cosine_min, cosine_max = min(cosine_distance_vals), max(cosine_distance_vals)

    print("Global min-max values used for normalisation:")
    print(f"  word_levenshtein: min={edit_min}, max={edit_max}")
    print(f"  edit_density:     min={density_min}, max={density_max}")
    print(f"  delta_ttr:        min={ttr_min}, max={ttr_max}")
    print(f"  delta_readability:min={read_min}, max={read_max}")
    print(f"  delta_fluency:    min={fluency_min}, max={fluency_max}")
    print(f"  1-cosine:         min={cosine_min}, max={cosine_max}")

    per_sentence_rows = []
    grouped_oci = defaultdict(list)
    grouped_divergence = defaultdict(list)

    for r in rows:
        edit_distance = safe_float(r["word_levenshtein"])
        edit_density = safe_float(r["edit_density"])
        delta_ttr = safe_float(r["delta_ttr"])
        delta_r = safe_float(r["delta_readability"])
        source_fluency = safe_float(r["source_fluency"]) if "source_fluency" in r and r["source_fluency"] != "" else fluency_score(r.get("source", ""))
        output_fluency = safe_float(r["output_fluency"]) if "output_fluency" in r and r["output_fluency"] != "" else fluency_score(r.get("output", ""))
        delta_fluency = safe_float(r["delta_fluency"]) if "delta_fluency" in r and r["delta_fluency"] != "" else output_fluency - source_fluency
        cos_sim = clip01(safe_float(r["stylometric_cosine"]))
        cosine_distance = clip01(1.0 - cos_sim)

        # Normalise each component
        norm_edit = min_max_normalise(edit_distance, edit_min, edit_max)
        norm_density = min_max_normalise(edit_density, density_min, density_max)
        norm_ttr = min_max_normalise(delta_ttr, ttr_min, ttr_max)
        norm_r = min_max_normalise(delta_r, read_min, read_max)
        norm_fluency = min_max_normalise(delta_fluency, fluency_min, fluency_max)
        norm_cosine_distance = cosine_distance

        # Compute OCI using the new weighted formulation
        oci_divergence = clip01(
            W_EDIT * norm_edit
            + W_DENSITY * norm_density
            + W_TTR * norm_ttr
            + W_READABILITY * norm_r
            + W_COSINE * norm_cosine_distance
        )
        utility = clip01(W_FLUENCY_UTILITY * norm_fluency + W_MEANING_UTILITY * cos_sim)
        oci = clip01(oci_divergence * (1.0 - utility))
        oci_percent = oci * 100  # Convert to percentage

        out_row = dict(r)
        out_row["source_fluency"] = source_fluency
        out_row["output_fluency"] = output_fluency
        out_row["delta_fluency"] = delta_fluency
        out_row["norm_edit_distance"] = norm_edit
        out_row["norm_edit_density"] = norm_density
        out_row["norm_delta_ttr"] = norm_ttr
        out_row["norm_delta_readability"] = norm_r
        out_row["norm_delta_fluency"] = norm_fluency
        out_row["norm_1_minus_cosine"] = norm_cosine_distance
        out_row["utility"] = utility
        out_row["oci_divergence"] = oci_divergence
        out_row["oci_divergence_percent"] = oci_divergence * 100
        out_row["oci_utility"] = oci
        out_row["oci_utility_percent"] = oci_percent
        out_row["oci"] = oci
        out_row["oci_percent"] = oci_percent

        per_sentence_rows.append(out_row)

        condition = r.get("condition", "unknown")
        grouped_oci[condition].append(oci)
        grouped_divergence[condition].append(oci_divergence)

    # Save per-sentence OCI
    fieldnames = list(per_sentence_rows[0].keys())
    with PER_SENTENCE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sentence_rows)

    # Save aggregate OCI per condition
    agg_rows = []
    for condition, ocis in sorted(grouped_oci.items()):
        divergences = grouped_divergence[condition]
        mean_oci_val = mean(ocis)
        median_oci_val = median(ocis)
        mean_divergence = mean(divergences)
        median_divergence = median(divergences)
        agg_rows.append({
            "condition": condition,
            "n_sentences": len(ocis),
            "mean_oci_divergence": mean_divergence,
            "mean_oci_divergence_percent": mean_divergence * 100,
            "median_oci_divergence": median_divergence,
            "median_oci_divergence_percent": median_divergence * 100,
            "min_oci_divergence": min(divergences) if divergences else 0.0,
            "max_oci_divergence": max(divergences) if divergences else 0.0,
            "mean_oci_utility": mean_oci_val,
            "mean_oci_utility_percent": mean_oci_val * 100,
            "median_oci_utility": median_oci_val,
            "median_oci_utility_percent": median_oci_val * 100,
            "min_oci_utility": min(ocis) if ocis else 0.0,
            "max_oci_utility": max(ocis) if ocis else 0.0,
            "mean_oci": mean_oci_val,
            "mean_oci_percent": mean_oci_val * 100,
            "median_oci": median_oci_val,
            "median_oci_percent": median_oci_val * 100,
            "min_oci": min(ocis) if ocis else 0.0,
            "max_oci": max(ocis) if ocis else 0.0,
        })

    with AGG_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "n_sentences",
                "mean_oci_divergence", "mean_oci_divergence_percent",
                "median_oci_divergence", "median_oci_divergence_percent",
                "min_oci_divergence", "max_oci_divergence",
                "mean_oci_utility", "mean_oci_utility_percent",
                "median_oci_utility", "median_oci_utility_percent",
                "min_oci_utility", "max_oci_utility",
                "mean_oci", "mean_oci_percent",
                "median_oci", "median_oci_percent",
                "min_oci", "max_oci"
            ],
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Saved per-sentence OCI: {PER_SENTENCE_OUT}")
    print(f"Saved aggregate OCI:    {AGG_OUT}")


if __name__ == "__main__":
    main()
