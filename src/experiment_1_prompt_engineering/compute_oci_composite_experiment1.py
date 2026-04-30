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
UTILITY_WEIGHTS = {
    "delta_fluency": 0.50,
    "cosine_similarity": 0.50,
}


def min_max_normalise(values: pd.Series) -> pd.Series:
    min_value = values.min()
    max_value = values.max()
    if max_value == min_value:
        return values * 0.0
    return (values - min_value) / (max_value - min_value)


def fluency_score(text: str) -> float:
    stripped = str(text).strip()
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
    return max(
        0.0,
        min(
            1.0,
            1.0
            - 0.35 * repeated_penalty
            - 0.20 * punctuation_penalty
            - 0.10 * word_length_penalty
            - 0.10 * sentence_len_penalty
            - terminal_penalty,
        ),
    )


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    df["norm_edit_distance"] = min_max_normalise(df["word_levenshtein"].astype(float))
    df["norm_edit_density"] = min_max_normalise(df["edit_density"].astype(float))
    df["norm_delta_ttr"] = min_max_normalise(df["delta_ttr"].astype(float))
    df["norm_delta_readability"] = min_max_normalise(df["delta_readability"].astype(float))
    df["source_fluency"] = df.get("source_fluency", df["source"].map(fluency_score)).astype(float)
    df["output_fluency"] = df.get("output_fluency", df["output"].map(fluency_score)).astype(float)
    df["delta_fluency"] = df.get("delta_fluency", df["output_fluency"] - df["source_fluency"]).astype(float)
    df["norm_delta_fluency"] = min_max_normalise(df["delta_fluency"])
    df["norm_1_minus_cosine"] = (1.0 - df["stylometric_cosine"].astype(float)).clip(0.0, 1.0)
    df["cosine_similarity"] = df["stylometric_cosine"].astype(float).clip(0.0, 1.0)

    df["oci_divergence"] = (
        WEIGHTS["edit_distance"] * df["norm_edit_distance"]
        + WEIGHTS["edit_density"] * df["norm_edit_density"]
        + WEIGHTS["delta_ttr"] * df["norm_delta_ttr"]
        + WEIGHTS["delta_readability"] * df["norm_delta_readability"]
        + WEIGHTS["cosine_distance"] * df["norm_1_minus_cosine"]
    ).clip(0.0, 1.0)
    df["oci_divergence_percent"] = df["oci_divergence"] * 100
    df["utility"] = (
        UTILITY_WEIGHTS["delta_fluency"] * df["norm_delta_fluency"]
        + UTILITY_WEIGHTS["cosine_similarity"] * df["cosine_similarity"]
    ).clip(0.0, 1.0)
    df["oci_utility"] = (df["oci_divergence"] * (1.0 - df["utility"])).clip(0.0, 1.0)
    df["oci_utility_percent"] = df["oci_utility"] * 100
    df["oci"] = df["oci_utility"]
    df["oci_percent"] = df["oci"] * 100

    aggregate = df.groupby("condition", as_index=False).agg(
        n_sentences=("oci", "size"),
        mean_oci_divergence=("oci_divergence", "mean"),
        mean_oci_divergence_percent=("oci_divergence_percent", "mean"),
        median_oci_divergence=("oci_divergence", "median"),
        median_oci_divergence_percent=("oci_divergence_percent", "median"),
        min_oci_divergence=("oci_divergence", "min"),
        max_oci_divergence=("oci_divergence", "max"),
        mean_oci_utility=("oci_utility", "mean"),
        mean_oci_utility_percent=("oci_utility_percent", "mean"),
        median_oci_utility=("oci_utility", "median"),
        median_oci_utility_percent=("oci_utility_percent", "median"),
        min_oci_utility=("oci_utility", "min"),
        max_oci_utility=("oci_utility", "max"),
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
