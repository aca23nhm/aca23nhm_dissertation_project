"""Analyse manually labelled human validation sample for OCI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = PROJECT_ROOT / "results" / "human_validation_sample.csv"
ANALYSIS_OUT = PROJECT_ROOT / "results" / "human_validation_analysis.csv"
SUMMARY_OUT = PROJECT_ROOT / "results" / "human_validation_summary.md"
FIGURE_OUT = PROJECT_ROOT / "results" / "human_validation_oci_by_label.png"

LABEL_ORDER = [
    "minimal_correct_correction",
    "acceptable_useful_rewrite",
    "under_correction_or_error",
    "over_correction",
    "meaning_change",
]

HIGH_RISK_LABELS = {"over_correction", "meaning_change"}
RISK_SCORE = {
    "minimal_correct_correction": 0,
    "acceptable_useful_rewrite": 1,
    "under_correction_or_error": 1,
    "over_correction": 2,
    "meaning_change": 2,
}


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.001:
        return "< .001"
    return f"{p_value:.3f}"


def load_labelled_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing labelled sample: {SAMPLE_PATH}")

    df = pd.read_csv(SAMPLE_PATH)
    required = {"example_id", "OCI", "human_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {SAMPLE_PATH}: {sorted(missing)}")

    df["human_label"] = df["human_label"].fillna("").str.strip()
    df["OCI"] = pd.to_numeric(df["OCI"], errors="coerce")
    df["high_risk"] = df["human_label"].isin(HIGH_RISK_LABELS)
    df["risk_score"] = df["human_label"].map(RISK_SCORE)
    return df


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df[df["human_label"].isin(LABEL_ORDER)].copy()
    grouped = labelled.groupby("human_label", observed=False)
    summary = grouped.agg(
        n=("OCI", "count"),
        mean_oci=("OCI", "mean"),
        median_oci=("OCI", "median"),
        std_oci=("OCI", "std"),
        min_oci=("OCI", "min"),
        max_oci=("OCI", "max"),
        mean_edit_distance=("edit_distance", "mean") if "edit_distance" in labelled.columns else ("OCI", "size"),
        mean_similarity=("similarity", "mean") if "similarity" in labelled.columns else ("OCI", "size"),
        mean_fluency_delta=("fluency_delta", "mean") if "fluency_delta" in labelled.columns else ("OCI", "size"),
    ).reset_index()
    summary["analysis"] = "label_distribution"
    summary["is_high_risk_label"] = summary["human_label"].isin(HIGH_RISK_LABELS)
    return summary


def spearman_analysis(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["OCI", "risk_score"])
    if valid.empty:
        rho, p_value = np.nan, np.nan
    else:
        result = stats.spearmanr(valid["OCI"], valid["risk_score"])
        rho, p_value = float(result.statistic), float(result.pvalue)

    return pd.DataFrame(
        [
            {
                "analysis": "spearman_oci_vs_ordinal_risk",
                "human_label": "all_valid_labels",
                "n": int(valid.shape[0]),
                "spearman_rho": rho,
                "spearman_p": p_value,
                "spearman_p_formatted": format_p_value(p_value),
            }
        ]
    )


def threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["OCI"]).copy()
    valid = valid[valid["human_label"].isin(LABEL_ORDER)]

    thresholds = [
        ("median_oci", float(valid["OCI"].median())),
        ("upper_tertile_oci", float(valid["OCI"].quantile(2 / 3))),
        ("upper_quartile_oci", float(valid["OCI"].quantile(0.75))),
    ]
    if "oci_band" in valid.columns and (valid["oci_band"] == "high").any():
        band_threshold = float(valid.loc[valid["oci_band"] == "high", "OCI"].min())
        thresholds.append(("sample_high_oci_band_min", band_threshold))

    rows = []
    actual = valid["high_risk"]
    for name, threshold in thresholds:
        predicted = valid["OCI"] >= threshold
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        tn = int((~predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(valid) if len(valid) else np.nan

        rows.append(
            {
                "analysis": "threshold_high_risk_detection",
                "threshold_name": name,
                "threshold_oci": threshold,
                "n": int(len(valid)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def write_figure(df: pd.DataFrame) -> None:
    valid = df[df["human_label"].isin(LABEL_ORDER)].copy()
    values = [valid.loc[valid["human_label"] == label, "OCI"].dropna().to_numpy() for label in LABEL_ORDER]
    labels = [
        "minimal\ncorrect",
        "useful\nrewrite",
        "under/error",
        "over-\ncorrection",
        "meaning\nchange",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True, patch_artist=True)

    rng = np.random.default_rng(1234)
    for idx, group_values in enumerate(values, start=1):
        if len(group_values) == 0:
            continue
        jitter = rng.normal(0, 0.035, size=len(group_values))
        ax.scatter(np.full(len(group_values), idx) + jitter, group_values, alpha=0.75, s=28)

    ax.set_title("OCI Distribution by Human Validation Label")
    ax.set_xlabel("Human label")
    ax.set_ylabel("OCI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_OUT, dpi=200)
    plt.close(fig)


def write_outputs(df: pd.DataFrame) -> None:
    missing_labels = df[df["human_label"] == ""]
    invalid_labels = df[(df["human_label"] != "") & ~df["human_label"].isin(LABEL_ORDER)]

    label_stats = label_distribution(df)
    spearman = spearman_analysis(df)
    thresholds = threshold_metrics(df)

    analysis_frames = [
        label_stats,
        spearman,
        thresholds,
    ]
    pd.concat(analysis_frames, ignore_index=True, sort=False).to_csv(ANALYSIS_OUT, index=False)
    write_figure(df)

    high_risk_count = int(df["high_risk"].sum())
    valid_label_count = int(df["human_label"].isin(LABEL_ORDER).sum())

    lines = [
        "# Human Validation Analysis Summary",
        "",
        f"Input file: `{SAMPLE_PATH.as_posix()}`",
        f"Output CSV: `{ANALYSIS_OUT.as_posix()}`",
        f"Figure: `{FIGURE_OUT.as_posix()}`",
        "",
        "## Label Completeness",
        "",
        f"- Total examples: {len(df)}",
        f"- Valid labelled examples: {valid_label_count}",
        f"- Missing labels: {len(missing_labels)}",
        f"- Invalid labels: {len(invalid_labels)}",
    ]

    if not missing_labels.empty:
        lines.append(f"- Missing label example IDs: {', '.join(missing_labels['example_id'].astype(str))}")
    if not invalid_labels.empty:
        bad = ", ".join(f"{row.example_id}={row.human_label}" for row in invalid_labels.itertuples())
        lines.append(f"- Invalid labels: {bad}")

    lines.extend(
        [
            "",
            "## OCI by Human Label",
            "",
            "| Human label | n | Mean OCI | Median OCI | Min OCI | Max OCI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for label in LABEL_ORDER:
        row = label_stats[label_stats["human_label"] == label]
        if row.empty:
            lines.append(f"| {label} | 0 | NA | NA | NA | NA |")
            continue
        row = row.iloc[0]
        lines.append(
            f"| {label} | {int(row['n'])} | {row['mean_oci']:.6f} | {row['median_oci']:.6f} | {row['min_oci']:.6f} | {row['max_oci']:.6f} |"
        )

    spear = spearman.iloc[0]
    lines.extend(
        [
            "",
            "## OCI and Ordinal Risk Score",
            "",
            "Ordinal risk scores: minimal correct = 0; acceptable useful rewrite = 1; under-correction/error = 1; over-correction = 2; meaning change = 2.",
            f"- Spearman rho: {spear['spearman_rho']:.3f}",
            f"- p-value: {spear['spearman_p_formatted']}",
            "",
            "## High-Risk Labels",
            "",
            "`over_correction` and `meaning_change` are treated as high-risk labels.",
            f"- High-risk examples: {high_risk_count}",
            "",
            "## Threshold Analysis",
            "",
            "| Threshold | OCI cut-off | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in thresholds.itertuples():
        lines.append(
            f"| {row.threshold_name} | {row.threshold_oci:.6f} | {row.tp} | {row.fp} | {row.tn} | {row.fn} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {row.accuracy:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "This analysis checks whether the manually assigned labels align with OCI values in the validation sample. Because the sample has only 60 examples and only a small number of high-risk labels, the threshold results should be interpreted as exploratory rather than definitive.",
        ]
    )

    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_labelled_sample()
    write_outputs(df)
    print(f"Saved analysis CSV to: {ANALYSIS_OUT}")
    print(f"Saved summary to: {SUMMARY_OUT}")
    print(f"Saved figure to: {FIGURE_OUT}")


if __name__ == "__main__":
    main()
