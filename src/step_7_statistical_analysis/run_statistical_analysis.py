from __future__ import annotations

import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import shapiro, ttest_rel, wilcoxon, pearsonr, spearmanr, norm

ROOT = Path(__file__).resolve().parents[2]

OCI_CSV = ROOT / "outputs" / "oci_eval" / "per_sentence_oci.csv"
STYLE_CSV = ROOT / "outputs" / "style_eval" / "per_sentence_style_metrics.csv"
EDITS_CSV = ROOT / "outputs" / "errant_outputs" / "per_sentence_edits.csv"
ERRANT_DIR = ROOT / "outputs" / "errant_outputs"

OUT_DIR = ROOT / "outputs" / "statistical_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRED_VALUES_OUT = OUT_DIR / "paired_oci_values.csv"
PAIRED_RESULTS_OUT = OUT_DIR / "paired_significance_results.csv"
ERROR_TYPE_OUT = OUT_DIR / "error_type_rewriting_analysis.csv"
TRADEOFF_OUT = OUT_DIR / "tradeoff_analysis.csv"

BASELINE_CONDITION = "baseline"

# Freeze thresholds before analysis
EDIT_DENSITY_THRESHOLD = 0.30
OCI_THRESHOLD = 0.50

# Baseline vs these structured-prompt conditions
# Update names if your actual prompt_id values differ.
COMPARE_TO = ["instruction", "role", "fewshot"]


def safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev_sample(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return statistics.stdev(xs)


def parse_f05_from_report(report_path: Path) -> float:
    """
    Extract F0.5 from ERRANT report text.
    Works with lines like:
    F0.5
    or
    F_0.5
    """
    text = report_path.read_text(encoding="utf-8", errors="replace")

    # Try common patterns
    patterns = [
        r"F0\.5\s*[:=]\s*([0-9.]+)",
        r"F_0\.5\s*[:=]\s*([0-9.]+)",
        r"F0\.5\s+([0-9.]+)",
        r"F_0\.5\s+([0-9.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    # Fallback: take last number from lines mentioning F
    for line in text.splitlines():
        if "F0.5" in line or "F_0.5" in line:
            nums = re.findall(r"[0-9]*\.?[0-9]+", line)
            if nums:
                return float(nums[-1])

    raise ValueError(f"Could not extract F0.5 from {report_path}")


def rank_biserial_effect_from_wilcoxon(x: list[float], y: list[float]) -> float:
    """
    Approximate rank-based effect size using z / sqrt(N).
    SciPy wilcoxon may not directly return z-statistic, so derive from p-value and sign.
    This is acceptable for dissertation-level analysis if documented.
    """
    diffs = [a - b for a, b in zip(x, y) if a != b]
    n = len(diffs)
    if n == 0:
        return 0.0

    res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", correction=False)
    p = res.pvalue

    # two-sided p -> |z|
    if p <= 0:
        z_abs = 0.0
    else:
        z_abs = abs(norm.ppf(p / 2.0))

    sign = 1 if mean(diffs) >= 0 else -1
    z = sign * z_abs
    return z / math.sqrt(n)


def cohens_d_paired(x: list[float], y: list[float]) -> float:
    diffs = [a - b for a, b in zip(x, y)]
    sd = stdev_sample(diffs)
    if sd == 0:
        return 0.0
    return mean(diffs) / sd


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_paired_oci() -> tuple[list[dict], dict[str, dict[str, float]]]:
    """
    Returns:
      paired_rows: one row per sentence_id with OCI columns per condition
      oci_by_condition: condition -> sentence_id -> oci
    """
    if not OCI_CSV.exists():
        raise FileNotFoundError(f"Missing {OCI_CSV}. Run Step 6 first.")

    rows = load_csv_rows(OCI_CSV)

    # sentence_id -> condition -> oci
    by_sentence = defaultdict(dict)
    by_condition = defaultdict(dict)

    for r in rows:
        sid = str(r["sentence_id"])
        cond = r["condition"]
        oci = safe_float(r["oci"])
        by_sentence[sid][cond] = oci
        by_condition[cond][sid] = oci

    all_conditions = sorted(by_condition.keys())

    paired_rows = []
    for sid in sorted(by_sentence.keys(), key=lambda s: (len(s), s)):
        row = {"sentence_id": sid}
        for cond in all_conditions:
            row[f"oci_{cond}"] = by_sentence[sid].get(cond, "")
        paired_rows.append(row)

    return paired_rows, by_condition


def run_paired_tests(oci_by_condition: dict[str, dict[str, float]]) -> list[dict]:
    if BASELINE_CONDITION not in oci_by_condition:
        raise ValueError(f"Baseline condition '{BASELINE_CONDITION}' not found in OCI data.")

    baseline_map = oci_by_condition[BASELINE_CONDITION]
    results = []

    for cond in COMPARE_TO:
        if cond not in oci_by_condition:
            print(f"Skipping comparison: {cond} not found.")
            continue

        other_map = oci_by_condition[cond]
        shared_ids = sorted(set(baseline_map) & set(other_map))

        x = [baseline_map[sid] for sid in shared_ids]
        y = [other_map[sid] for sid in shared_ids]
        diffs = [a - b for a, b in zip(x, y)]

        if len(diffs) < 3:
            print(f"Skipping {BASELINE_CONDITION} vs {cond}: too few paired samples.")
            continue

        # normality check on paired differences
        # Shapiro is standard for small/medium paired samples
        shapiro_stat, shapiro_p = shapiro(diffs)

        if shapiro_p > 0.05:
            test_name = "paired_t_test"
            stat, p_value = ttest_rel(x, y)
            effect_size = cohens_d_paired(x, y)
            effect_name = "cohens_d_paired"
        else:
            test_name = "wilcoxon_signed_rank"
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", correction=False)
            stat, p_value = res.statistic, res.pvalue
            effect_size = rank_biserial_effect_from_wilcoxon(x, y)
            effect_name = "r"

        results.append({
            "comparison": f"{BASELINE_CONDITION}_vs_{cond}",
            "baseline_condition": BASELINE_CONDITION,
            "other_condition": cond,
            "n_pairs": len(shared_ids),
            "mean_oci_baseline": mean(x),
            "mean_oci_other": mean(y),
            "mean_difference_baseline_minus_other": mean(diffs),
            "normality_test": "shapiro_wilk",
            "normality_p_value": shapiro_p,
            "selected_test": test_name,
            "test_statistic": stat,
            "p_value": p_value,
            "effect_size_name": effect_name,
            "effect_size": effect_size,
        })

    return results


def run_error_type_rewriting_analysis() -> list[dict]:
    if not EDITS_CSV.exists():
        raise FileNotFoundError(f"Missing {EDITS_CSV}. Run ERRANT edit export first.")
    if not STYLE_CSV.exists():
        raise FileNotFoundError(f"Missing {STYLE_CSV}. Run Step 5 first.")
    if not OCI_CSV.exists():
        raise FileNotFoundError(f"Missing {OCI_CSV}. Run Step 6 first.")

    edits_rows = load_csv_rows(EDITS_CSV)
    style_rows = load_csv_rows(STYLE_CSV)
    oci_rows = load_csv_rows(OCI_CSV)

    # Use gold edits only to define error types present in the sentence
    error_types_by_sentence = defaultdict(set)
    for r in edits_rows:
        if r.get("role") != "gold":
            continue
        sid = str(r["sent_idx"])
        err = r["error_type"]
        if err:
            error_types_by_sentence[sid].add(err)

    # sentence_id + condition -> edit density
    style_map = {}
    for r in style_rows:
        sid = str(r["sentence_id"])
        cond = r["condition"]
        style_map[(sid, cond)] = safe_float(r["edit_density"])

    # sentence_id + condition -> oci
    oci_map = {}
    all_conditions = set()
    for r in oci_rows:
        sid = str(r["sentence_id"])
        cond = r["condition"]
        oci_map[(sid, cond)] = safe_float(r["oci"])
        all_conditions.add(cond)

    # For each error type and condition, count rewriting vs not
    counts = defaultdict(lambda: {"rewriting": 0, "not_rewriting": 0})

    # Use all sentence-condition pairs where both style and oci exist
    for (sid, cond), oci in oci_map.items():
        edit_density = style_map.get((sid, cond), 0.0)
        rewriting_happened = (edit_density > EDIT_DENSITY_THRESHOLD) or (oci > OCI_THRESHOLD)

        sentence_error_types = error_types_by_sentence.get(sid, set())
        for err_type in sentence_error_types:
            key = (cond, err_type)
            if rewriting_happened:
                counts[key]["rewriting"] += 1
            else:
                counts[key]["not_rewriting"] += 1

    rows = []
    for (cond, err_type), c in sorted(counts.items()):
        total = c["rewriting"] + c["not_rewriting"]
        rows.append({
            "condition": cond,
            "error_type": err_type,
            "rewriting_count": c["rewriting"],
            "not_rewriting_count": c["not_rewriting"],
            "rewriting_rate": (c["rewriting"] / total) if total else 0.0,
            "edit_density_threshold": EDIT_DENSITY_THRESHOLD,
            "oci_threshold": OCI_THRESHOLD,
        })

    return rows


def run_tradeoff_analysis() -> list[dict]:
    """
    Correlate grammatical accuracy (F0.5) with style divergence (mean OCI)
    across conditions.
    """
    agg_oci_path = ROOT / "outputs" / "oci_eval" / "aggregate_oci.csv"
    if not agg_oci_path.exists():
        raise FileNotFoundError(f"Missing {agg_oci_path}. Run Step 6 first.")

    agg_rows = load_csv_rows(agg_oci_path)

    # condition -> mean OCI
    oci_by_cond = {r["condition"]: safe_float(r["mean_oci"]) for r in agg_rows}

    # condition -> F0.5
    f05_by_cond = {}
    for report_path in sorted(ERRANT_DIR.glob("*.report.txt")):
        condition = report_path.name.replace(".report.txt", "")
        try:
            f05_by_cond[condition] = parse_f05_from_report(report_path)
        except Exception as e:
            print(f"Warning: could not parse F0.5 for {condition}: {e}")

    shared_conds = sorted(set(oci_by_cond) & set(f05_by_cond))
    if len(shared_conds) < 3:
        raise ValueError("Need at least 3 shared conditions for correlation analysis.")

    x = [f05_by_cond[c] for c in shared_conds]
    y = [oci_by_cond[c] for c in shared_conds]

    # normality checks on each variable
    x_normal_p = shapiro(x).pvalue if len(x) >= 3 else 0.0
    y_normal_p = shapiro(y).pvalue if len(y) >= 3 else 0.0

    if x_normal_p > 0.05 and y_normal_p > 0.05:
        method = "pearson"
        corr, p_value = pearsonr(x, y)
    else:
        method = "spearman"
        corr, p_value = spearmanr(x, y)

    rows = []
    for cond in shared_conds:
        rows.append({
            "analysis_type": "condition_values",
            "condition": cond,
            "f0_5": f05_by_cond[cond],
            "mean_oci": oci_by_cond[cond],
            "correlation_method": "",
            "correlation_coefficient": "",
            "p_value": "",
            "x_normality_p": "",
            "y_normality_p": "",
        })

    rows.append({
        "analysis_type": "overall_correlation",
        "condition": "ALL",
        "f0_5": "",
        "mean_oci": "",
        "correlation_method": method,
        "correlation_coefficient": corr,
        "p_value": p_value,
        "x_normality_p": x_normal_p,
        "y_normality_p": y_normal_p,
    })

    return rows


def main() -> None:
    # 1. Paired OCI dataset
    paired_rows, oci_by_condition = build_paired_oci()
    if paired_rows:
        paired_fieldnames = list(paired_rows[0].keys())
        save_csv(PAIRED_VALUES_OUT, paired_rows, paired_fieldnames)
        print(f"Saved: {PAIRED_VALUES_OUT}")

    # 2. Paired tests
    paired_results = run_paired_tests(oci_by_condition)
    if paired_results:
        save_csv(PAIRED_RESULTS_OUT, paired_results, list(paired_results[0].keys()))
        print(f"Saved: {PAIRED_RESULTS_OUT}")

    # 3. Error-type rewriting analysis
    error_rows = run_error_type_rewriting_analysis()
    if error_rows:
        save_csv(ERROR_TYPE_OUT, error_rows, list(error_rows[0].keys()))
        print(f"Saved: {ERROR_TYPE_OUT}")

    # 4. Trade-off analysis
    tradeoff_rows = run_tradeoff_analysis()
    if tradeoff_rows:
        save_csv(TRADEOFF_OUT, tradeoff_rows, list(tradeoff_rows[0].keys()))
        print(f"Saved: {TRADEOFF_OUT}")

    print("Step 7 statistical analysis complete.")


if __name__ == "__main__":
    main()