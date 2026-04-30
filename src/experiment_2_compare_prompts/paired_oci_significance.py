"""Paired OCI significance tests for Experiment 2.

This script tests whether each structured prompt reduces sentence-level OCI
relative to the baseline prompt. It uses the paired design of Experiment 2:
the same sentence appears under every prompt condition.
"""

from pathlib import Path

import pandas as pd
from scipy import stats


INPUT_PATH = Path("outputs/experiment_2_compare_prompts/oci_eval/per_sentence_oci.csv")
OUTPUT_DIR = Path("outputs/experiment_2_compare_prompts/statistical_tests")
OUTPUT_PATH = OUTPUT_DIR / "paired_oci_significance.csv"

COMPARISONS = ["instruction", "role", "fewshot"]


def paired_cohens_dz(differences: pd.Series) -> float:
    """Compute paired Cohen's dz from paired differences."""
    std = differences.std(ddof=1)
    if std == 0:
        return 0.0
    return differences.mean() / std


def format_p_value(p_value: float) -> str:
    """Return a compact p-value string for report tables."""
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    pivot = df.pivot(index="sentence_id", columns="condition", values="oci")

    if "baseline" not in pivot.columns:
        raise ValueError("Expected a baseline condition in the OCI file.")

    records = []
    baseline = pivot["baseline"]

    for condition in COMPARISONS:
        if condition not in pivot.columns:
            raise ValueError(f"Expected condition {condition!r} in the OCI file.")

        prompt_scores = pivot[condition]
        differences = baseline - prompt_scores

        # One-sided test: baseline OCI is expected to be greater than prompt OCI.
        wilcoxon = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="greater",
        )
        paired_t = stats.ttest_rel(baseline, prompt_scores, alternative="greater")

        records.append(
            {
                "comparison": f"baseline_vs_{condition}",
                "n": int(differences.count()),
                "baseline_mean_oci": baseline.mean(),
                "prompt_mean_oci": prompt_scores.mean(),
                "mean_oci_reduction": differences.mean(),
                "median_oci_reduction": differences.median(),
                "wilcoxon_statistic": wilcoxon.statistic,
                "wilcoxon_p": wilcoxon.pvalue,
                "wilcoxon_p_formatted": format_p_value(wilcoxon.pvalue),
                "paired_t_p": paired_t.pvalue,
                "cohens_dz": paired_cohens_dz(differences),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved paired OCI significance results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
