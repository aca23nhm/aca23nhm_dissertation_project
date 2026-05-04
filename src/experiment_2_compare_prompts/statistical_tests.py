"""Statistical tests for prompt-comparison results.

The main prompt-comparison experiment evaluates each sentence under the same
prompt conditions, so baseline-vs-structured comparisons are paired by
sentence_id. The script reports paired OCI reductions, bootstrap confidence
intervals, and correlations between sentence length and OCI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT2_OCI = PROJECT_ROOT / "outputs" / "experiment_2_compare_prompts" / "oci_eval" / "per_sentence_oci.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_OUT = RESULTS_DIR / "statistical_tests.csv"
SUMMARY_OUT = RESULTS_DIR / "statistical_tests_summary.md"

BASELINE = "baseline"
STRUCTURED_PROMPTS = ["instruction", "role", "fewshot"]
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 1234


def bootstrap_ci_mean(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Return a percentile bootstrap 95% CI for a mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.size, size=(iterations, values.size))
    sample_means = values[sample_indices].mean(axis=1)
    lower, upper = np.percentile(sample_means, [2.5, 97.5])
    return float(lower), float(upper)


def paired_cohens_dz(values: pd.Series) -> float:
    """Return paired Cohen's dz for a vector of paired differences."""
    std = values.std(ddof=1)
    if pd.isna(std) or std == 0:
        return 0.0
    return float(values.mean() / std)


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def wilcoxon_greater(differences: pd.Series) -> tuple[float, float]:
    """Run a one-sided Wilcoxon test, handling all-zero differences."""
    non_zero = differences[differences != 0]
    if non_zero.empty:
        return 0.0, 1.0
    result = stats.wilcoxon(non_zero, alternative="greater", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def paired_oci_tests(df: pd.DataFrame) -> list[dict[str, object]]:
    pivot = df.pivot(index="sentence_id", columns="condition", values="oci")
    if BASELINE not in pivot.columns:
        raise ValueError(f"Missing required condition: {BASELINE}")

    records: list[dict[str, object]] = []
    baseline = pivot[BASELINE]

    for condition in STRUCTURED_PROMPTS:
        if condition not in pivot.columns:
            raise ValueError(f"Missing required condition: {condition}")

        prompt_scores = pivot[condition]
        paired = pd.concat([baseline, prompt_scores], axis=1, keys=["baseline", "prompt"]).dropna()
        differences = paired["baseline"] - paired["prompt"]
        ci_low, ci_high = bootstrap_ci_mean(differences.to_numpy())
        wilcoxon_stat, wilcoxon_p = wilcoxon_greater(differences)

        records.append(
            {
                "analysis": "paired_oci_vs_baseline",
                "comparison": f"{BASELINE}_vs_{condition}",
                "condition": condition,
                "n": int(differences.count()),
                "baseline_mean_oci": float(paired["baseline"].mean()),
                "prompt_mean_oci": float(paired["prompt"].mean()),
                "mean_oci_difference": float(differences.mean()),
                "median_oci_difference": float(differences.median()),
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "wilcoxon_p_formatted": format_p_value(wilcoxon_p),
                "cohens_dz": paired_cohens_dz(differences),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "note": "Positive OCI difference means baseline OCI is higher than the structured prompt.",
            }
        )

    return records


def spearman_length_oci(df: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    length_col = "source_word_count" if "source_word_count" in df.columns else "source_sent_len_words"
    if length_col not in df.columns:
        raise ValueError("No sentence-length column found in the OCI file.")

    groups: list[tuple[str, pd.DataFrame]] = [("all_conditions", df)]
    groups.extend((condition, group) for condition, group in df.groupby("condition"))

    for label, group in groups:
        clean = group[[length_col, "oci"]].dropna()
        if clean.empty:
            rho, p_value = np.nan, np.nan
        else:
            result = stats.spearmanr(clean[length_col], clean["oci"])
            rho, p_value = float(result.statistic), float(result.pvalue)

        records.append(
            {
                "analysis": "spearman_sentence_length_oci",
                "comparison": f"sentence_length_vs_oci_{label}",
                "condition": label,
                "n": int(clean.shape[0]),
                "baseline_mean_oci": np.nan,
                "prompt_mean_oci": np.nan,
                "mean_oci_difference": np.nan,
                "median_oci_difference": np.nan,
                "bootstrap_ci_95_low": np.nan,
                "bootstrap_ci_95_high": np.nan,
                "wilcoxon_statistic": np.nan,
                "wilcoxon_p": np.nan,
                "wilcoxon_p_formatted": "NA",
                "cohens_dz": np.nan,
                "spearman_rho": rho,
                "spearman_p": p_value,
                "note": f"Spearman correlation uses {length_col} and sentence-level OCI.",
            }
        )

    return records


def write_summary(results: pd.DataFrame) -> None:
    paired = results[results["analysis"] == "paired_oci_vs_baseline"]
    corr = results[results["analysis"] == "spearman_sentence_length_oci"]

    lines = [
        "# Statistical Tests Summary",
        "",
        f"Input file: `{EXPERIMENT2_OCI.as_posix()}`",
        "",
        "## Paired OCI Comparisons",
        "",
        "Each structured prompt is compared with the baseline on the same sentence IDs. "
        "Positive differences mean the baseline has higher OCI than the structured prompt.",
        "",
        "| Comparison | n | Mean OCI difference | Median OCI difference | 95% bootstrap CI | Wilcoxon p | Cohen's dz |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in paired.iterrows():
        lines.append(
            "| {comparison} | {n} | {mean:.6f} | {median:.6f} | [{low:.6f}, {high:.6f}] | {p} | {dz:.3f} |".format(
                comparison=row["comparison"],
                n=int(row["n"]),
                mean=row["mean_oci_difference"],
                median=row["median_oci_difference"],
                low=row["bootstrap_ci_95_low"],
                high=row["bootstrap_ci_95_high"],
                p=row["wilcoxon_p_formatted"],
                dz=row["cohens_dz"],
            )
        )

    lines.extend(
        [
            "",
            "## Sentence Length and OCI",
            "",
            "| Condition | n | Spearman rho | p-value |",
            "|---|---:|---:|---:|",
        ]
    )

    for _, row in corr.iterrows():
        lines.append(
            "| {condition} | {n} | {rho:.3f} | {p} |".format(
                condition=row["condition"],
                n=int(row["n"]),
                rho=row["spearman_rho"],
                p=format_p_value(row["spearman_p"]),
            )
        )

    lines.extend(
        [
            "",
            "## Availability Notes",
            "",
            "- Sentence-level OCI values are available and were used for paired tests.",
            "- Sentence length is available through the OCI component file and was used for Spearman correlation.",
            "- Sentence-level ERRANT $F_{0.5}$ or edit-level accuracy values were not found in the available result files, so paired/bootstrap accuracy tests were not computed.",
        ]
    )

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not EXPERIMENT2_OCI.exists():
        raise FileNotFoundError(f"Missing sentence-level OCI file: {EXPERIMENT2_OCI}")

    df = pd.read_csv(EXPERIMENT2_OCI)
    required = {"sentence_id", "condition", "oci"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {EXPERIMENT2_OCI}: {sorted(missing)}")

    records = paired_oci_tests(df)
    records.extend(spearman_length_oci(df))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(CSV_OUT, index=False)
    write_summary(results)

    print(results.to_string(index=False))
    print(f"\nSaved statistical test results to: {CSV_OUT}")
    print(f"Saved statistical test summary to: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
