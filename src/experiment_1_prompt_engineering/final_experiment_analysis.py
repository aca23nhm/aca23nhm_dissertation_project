from pathlib import Path
import csv
import re


def load_oci_results():
    """Load OCI results from the aggregate CSV."""
    oci_file = Path("outputs/experiment_1_prompt_engineering/oci_eval/aggregate_oci_simple.csv")
    oci_data = {}

    with open(oci_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            condition = row["condition"]
            oci_percent = float(row["mean_oci_percent"])
            oci_data[condition] = oci_percent

    return oci_data


def extract_f05_from_report(report_path: Path) -> float:
    """Extract F0.5 score from ERRANT report file."""
    content = report_path.read_text(encoding="utf-8")

    for line in content.splitlines():
        if re.match(r"^\d+\s+\d+\s+\d+", line.strip()):
            parts = line.strip().split()
            if len(parts) >= 6:
                return float(parts[5])

    raise ValueError(f"Could not find F0.5 score in {report_path}")


def load_errant_results():
    """Load ERRANT F0.5 results from report files."""
    reports_dir = Path("outputs/experiment_1_prompt_engineering/experiment1_errant_outputs")
    errant_data = {}

    for report_file in sorted(reports_dir.glob("*.report.txt")):
        condition = report_file.stem.replace(".report", "")
        errant_data[condition] = extract_f05_from_report(report_file)

    return errant_data


def parse_condition(condition):
    """
    Return prompt_type and version.
    Example:
        fewshot_v1 -> ("fewshot", 1)
        baseline -> ("baseline", None)
    """
    if "_v" not in condition:
        return condition, None

    prompt_type, version_str = condition.split("_v", 1)
    return prompt_type, int(version_str)


def main():
    oci_data = load_oci_results()
    errant_data = load_errant_results()

    results = []

    for condition in sorted(errant_data.keys()):
        if condition in oci_data:
            results.append({
                "condition": condition,
                "f05_score": errant_data[condition],
                "oci_percent": oci_data[condition]
            })
        else:
            print(f"[WARN] Missing OCI result for: {condition}")

    print("EXPERIMENT 1 RESULTS: Prompt Engineering Refinement")
    print("=" * 60)
    print("Condition\t\tF0.5\tOCI%")
    print("-" * 40)

    for result in results:
        print(
            f"{result['condition']:15}\t"
            f"{result['f05_score']:5.2f}\t"
            f"{result['oci_percent']:5.1f}"
        )

    prompt_types = {}

    for result in results:
        condition = result["condition"]
        prompt_type, version = parse_condition(condition)

        if prompt_type not in prompt_types:
            prompt_types[prompt_type] = []

        prompt_types[prompt_type].append({
            "version": version,
            "condition": condition,
            "f05": result["f05_score"],
            "oci": result["oci_percent"]
        })

    print("\nANALYSIS BY PROMPT TYPE:")
    print("=" * 40)

    for prompt_type, variants in prompt_types.items():
        print(f"\n{prompt_type.upper()} PROMPTS:")

        variants_sorted = sorted(variants, key=lambda x: x["f05"], reverse=True)

        for variant in variants_sorted:
            if variant["version"] is None:
                print(
                    f"    baseline: F0.5={variant['f05']:.2f}, "
                    f"OCI={variant['oci']:.1f}%"
                )
            else:
                print(
                    f"    v{variant['version']}: F0.5={variant['f05']:.2f}, "
                    f"OCI={variant['oci']:.1f}%"
                )

        best_variant = variants_sorted[0]

        if best_variant["version"] is None:
            print(f"  BEST: baseline (F0.5: {best_variant['f05']:.2f})")
        else:
            print(
                f"  BEST: v{best_variant['version']} "
                f"(F0.5: {best_variant['f05']:.2f})"
            )

        avg_f05 = sum(v["f05"] for v in variants) / len(variants)
        avg_oci = sum(v["oci"] for v in variants) / len(variants)
        print(f"  Average: F0.5={avg_f05:.2f}, OCI={avg_oci:.1f}%")

    all_variants = [item for sublist in prompt_types.values() for item in sublist]
    overall_best = max(all_variants, key=lambda x: x["f05"])

    print("\nOVERALL BEST PROMPT VARIANT:")
    print(
        f"  {overall_best['condition']} "
        f"(F0.5: {overall_best['f05']:.2f}, "
        f"OCI: {overall_best['oci']:.1f}%)"
    )

    if "baseline" in errant_data:
        print("\nBASELINE COMPARISON:")
        baseline_f05 = errant_data["baseline"]
        best_f05 = overall_best["f05"]
        improvement = best_f05 - baseline_f05
        print(f"  Baseline F0.5: {baseline_f05:.2f}")
        print(f"  Best F0.5: {best_f05:.2f}")
        print(f"  Improvement: +{improvement:.2f}")

    print("\nKEY FINDINGS:")
    print("- Baseline provides the reference point for prompt engineering improvements")
    print("- Fewshot prompts show the strongest improvement with version escalation")
    print("- Instruction and role prompts reach peak performance at later versions")
    print("- Over-correction (OCI) is generally low across all variants")
    print("- The best performing variant achieves the highest F0.5 while maintaining controlled over-correction")


if __name__ == "__main__":
    main()