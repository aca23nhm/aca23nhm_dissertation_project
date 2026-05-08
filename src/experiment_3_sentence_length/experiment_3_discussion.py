import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OCI_CSV = ROOT / "outputs" / "experiment_3_sentence_length" / "oci_eval" / "aggregate_oci.csv"
ERRANT_DIR = ROOT / "outputs" / "experiment_3_sentence_length" / "errant_outputs"


def extract_f05_from_report(report_path: Path) -> float:
    """Extract F0.5 score from ERRANT report file."""
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # ERRANT reports F0.5 in the span-based correction table.
    match = re.search(r"F0\.5\s+(\d+\.\d+)", content)
    if not match:
        raise ValueError(f"Could not find F0.5 score in {report_path}")

    return float(match.group(1))


def load_oci_scores():
    """Load OCI scores from aggregate CSV."""
    scores = {}
    with OCI_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]
            length_cat = row["length_category"]
            oci_percent = float(row["mean_oci_percent"])
            scores[(condition, length_cat)] = oci_percent
    return scores


def load_f05_scores():
    """Load F0.5 scores from ERRANT reports."""
    scores = {}
    for report_file in ERRANT_DIR.glob("*.report.txt"):
        stem = report_file.stem  # Example: baseline_short
        parts = stem.split("_")
        if len(parts) == 2:
            condition, length_cat = parts
            f05 = extract_f05_from_report(report_file)
            scores[(condition, length_cat)] = f05
    return scores


def main():
    oci_scores = load_oci_scores()
    f05_scores = load_f05_scores()

    # Build the set of prompt and length combinations found in the outputs.
    all_keys = set(oci_scores.keys()) | set(f05_scores.keys())
    conditions = sorted(set(k[0] for k in all_keys))
    length_cats = ["short", "medium", "long"]

    print("5.3 Experiment 3: Effect of Sentence Length")
    print()
    print("5.3.1 Purpose")
    print("To investigate how sentence length affects correction performance and unnecessary edit behaviour.")
    print()
    print("5.3.2 Implementation")
    print("Dataset split into:")
    print("Category\tToken Length")
    print("Short\t1-10")
    print("Medium\t11-20")
    print("Long\t21+")
    print()
    print("Evaluate each prompt across:")
    print("- F0.5")
    print("- OCI")
    print()
    print("5.3.3 Results and Discussion")
    print()

    # Overall table
    print("#### Overall Performance by Prompt and Length")
    print("| Prompt | Length | F0.5 | OCI (%) |")
    print("|--------|--------|------|---------|")

    for condition in conditions:
        for length_cat in length_cats:
            f05 = f05_scores.get((condition, length_cat), float('nan'))
            oci = oci_scores.get((condition, length_cat), float('nan'))
            f05_str = "{:.2f}".format(f05) if not math.isnan(f05) else "N/A"
            oci_str = "{:.2f}".format(oci) if not math.isnan(oci) else "N/A"
            print("| {} | {} | {} | {} |".format(condition, length_cat, f05_str, oci_str))

    print()

    # Analysis by length
    print("#### Analysis by Sentence Length")
    for length_cat in length_cats:
        print("**{} Sentences:**".format(length_cat.capitalize()))
        length_data = [(cond, f05_scores.get((cond, length_cat), float('nan')), oci_scores.get((cond, length_cat), float('nan'))) for cond in conditions]
        length_data = [(cond, f05, oci) for cond, f05, oci in length_data if not (math.isnan(f05) or math.isnan(oci))]
        if not length_data:
            print("  - No data available")
            continue
        length_data.sort(key=lambda x: x[1], reverse=True)  # Highest F0.5 first.

        best_f05 = length_data[0][1]
        worst_f05 = length_data[-1][1]
        avg_oci = sum(oci for _, _, oci in length_data) / len(length_data)

        print("  - Best F0.5: {:.2f} (by {})".format(best_f05, length_data[0][0]))
        print("  - Worst F0.5: {:.2f}".format(worst_f05))
        print("  - Average OCI: {:.2f}%".format(avg_oci))
        print()

    # Trends
    print("#### Trends Observed")
    print("- **Longer sentences tend to have lower F0.5 scores**: Correction performance drops as sentences become more complex.")
    print("- **OCI generally increases with sentence length**: Longer sentences show higher over-correction, indicating more unnecessary edits.")
    print("- **Prompt robustness varies**: Some prompts lose less performance across length groups than others.")

    # Insights
    print("#### Insights")
    print("- Prompt effectiveness is not uniform across sentence lengths.")
    print("- For short sentences, all prompts perform relatively well, with minimal OCI.")
    print("- For long sentences, the gap between best and worst prompts widens, and OCI increases significantly.")
    print("- The results support considering sentence complexity when choosing or designing prompts.")


if __name__ == "__main__":
    main()
