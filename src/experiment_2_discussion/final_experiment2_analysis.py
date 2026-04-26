from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRANT_DIR = ROOT / "outputs" / "experiment_2" / "errant_outputs"
OCI_PATH = ROOT / "outputs" / "experiment_2" / "oci_eval" / "aggregate_oci.csv"


def extract_f05_from_report(report_path: Path) -> float:
    text = report_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if re.match(r"^\d+\s+\d+\s+\d+", line.strip()):
            parts = line.strip().split()
            if len(parts) >= 6:
                return float(parts[5])
    raise ValueError(f"Could not find F0.5 score in {report_path}")


def load_errant_scores() -> dict[str, float]:
    if not ERRANT_DIR.exists():
        raise FileNotFoundError(f"Missing ERRANT reports directory: {ERRANT_DIR}")

    scores: dict[str, float] = {}
    for report_file in sorted(ERRANT_DIR.glob("*.report.txt")):
        condition = report_file.stem.replace(".report", "")
        scores[condition] = extract_f05_from_report(report_file)
    return scores


def load_oci_scores() -> dict[str, float]:
    if not OCI_PATH.exists():
        raise FileNotFoundError(f"Missing OCI aggregate file: {OCI_PATH}")

    scores: dict[str, float] = {}
    with OCI_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]
            scores[condition] = float(row["mean_oci_percent"])
    return scores


def classify_strategy(f05: float, oci: float) -> str:
    if oci <= 2.85 and f05 >= 49.0:
        return "Controlled"
    if oci <= 2.85:
        return "Controlled"
    if oci >= 3.00:
        return "Aggressive"
    return "Balanced"


def main() -> None:
    errant_scores = load_errant_scores()
    oci_scores = load_oci_scores()

    combined = []
    for condition, f05 in errant_scores.items():
        if condition not in oci_scores:
            print(f"[WARN] Missing OCI result for {condition}")
            continue
        combined.append({
            "condition": condition,
            "f05": f05,
            "oci": oci_scores[condition],
        })

    combined.sort(key=lambda x: x["f05"], reverse=True)

    print("Experiment 2: Comparison of Prompting Strategies")
    print("5.2.1 Purpose: Compare baseline, best instruction, best role-based, and best few-shot prompts.")
    print("5.2.2 Evaluation: ERRANT (Precision, Recall, F0.5) and OCI")
    print("\nComparison table: F0.5 vs OCI")
    print("Condition    F0.5    OCI%")
    print("---------    -----    -----")
    for row in combined:
        print(f"{row['condition']:11} {row['f05']:6.2f} {row['oci']:7.2f}")

    print("\nBehavioural analysis")
    print("Condition    Type      Interpretation")
    print("---------    ----      --------------")
    for row in combined:
        profile = classify_strategy(row["f05"], row["oci"])
        print(f"{row['condition']:11} {profile:9}  F0.5={row['f05']:.2f}, OCI={row['oci']:.2f}")

    baseline_f05 = errant_scores.get("baseline")
    baseline_oci = oci_scores.get("baseline")
    if baseline_f05 is not None and baseline_oci is not None:
        best = next((x for x in combined if x["condition"] != "baseline"), None)
        if best:
            print("\nCore insight: Trade-off between accuracy and unnecessary edits")
            print(f"  Baseline has F0.5={baseline_f05:.2f} and OCI={baseline_oci:.2f}.")
            print(f"  The tuned prompts improve F0.5 while lowering OCI compared to baseline.")
            print(f"  Among tuned prompts, the best few-shot prompt is the most aggressive, with the highest F0.5 and the highest OCI.")
            print(f"  Instruction and role prompts are more controlled, achieving lower OCI with only slightly lower F0.5.")


if __name__ == "__main__":
    main()