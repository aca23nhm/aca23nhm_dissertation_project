"""Compute final composite OCI for Experiment 1.

The first Experiment 1 OCI script used an early, simple excess-edit score.
This script recomputes Experiment 1 OCI using the final composite definition
used in Experiments 2--4, so Chapter 5 can compare OCI values on one scale.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT
    / "outputs"
    / "experiment_1_prompt_engineering"
    / "style_eval"
    / "per_sentence_style_metrics_simple.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "experiment_1_prompt_engineering" / "oci_eval"
PER_SENTENCE_OUT = OUTPUT_DIR / "per_sentence_oci_composite.csv"
AGGREGATE_OUT = OUTPUT_DIR / "aggregate_oci_composite.csv"

WEIGHTS = {
    "edit_distance": 0.35,
    "edit_density": 0.15,
    "delta_ttr": 0.20,
    "delta_readability": 0.15,
    "cosine_distance": 0.15,
}


def min_max_normalise(values: pd.Series) -> pd.Series:
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return values * 0.0
    return (values - min_value) / (max_value - min_value)


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df["norm_edit_distance"] = min_max_normalise(df["word_levenshtein"].astype(float))
    df["norm_edit_density"] = min_max_normalise(df["edit_density"].astype(float))
    df["norm_delta_ttr"] = min_max_normalise(df["delta_ttr"].astype(float))
    df["norm_delta_readability"] = min_max_normalise(df["delta_readability"].astype(float))
    df["norm_1_minus_cosine"] = (1.0 - df["stylometric_cosine"].astype(float)).clip(0.0, 1.0)

    df["oci"] = (
        WEIGHTS["edit_distance"] * df["norm_edit_distance"]
        + WEIGHTS["edit_density"] * df["norm_edit_density"]
        + WEIGHTS["delta_ttr"] * df["norm_delta_ttr"]
        + WEIGHTS["delta_readability"] * df["norm_delta_readability"]
        + WEIGHTS["cosine_distance"] * df["norm_1_minus_cosine"]
    )
    df["oci_percent"] = df["oci"] * 100

    aggregate = df.groupby("condition", as_index=False).agg(
        n_sentences=("oci", "size"),
        mean_oci=("oci", "mean"),
        mean_oci_percent=("oci_percent", "mean"),
        median_oci=("oci", "median"),
        median_oci_percent=("oci_percent", "median"),
        min_oci=("oci", "min"),
        max_oci=("oci", "max"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PER_SENTENCE_OUT, index=False)
    aggregate.to_csv(AGGREGATE_OUT, index=False)

    print(aggregate.sort_values("mean_oci").to_string(index=False))
    print(f"\nSaved per-sentence OCI to {PER_SENTENCE_OUT}")
    print(f"Saved aggregate OCI to {AGGREGATE_OUT}")


if __name__ == "__main__":
    main()
