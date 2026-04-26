from pathlib import Path
import re


def extract_f05_from_report(report_path: Path) -> float:
    """Extract F0.5 score from ERRANT report file."""
    content = report_path.read_text(encoding="utf-8")

    for line in content.splitlines():
        if re.match(r"^\d+\s+\d+\s+\d+", line.strip()):
            parts = line.strip().split()
            if len(parts) >= 6:
                return float(parts[5])

    raise ValueError(f"Could not find F0.5 score in {report_path}")


def main():
    reports_dir = Path("outputs/experiment1_errant_outputs")

    results = []

    for report_file in sorted(reports_dir.glob("*.report.txt")):
        condition = report_file.stem.replace(".report", "")

        try:
            f05_score = extract_f05_from_report(report_file)
            results.append((condition, f05_score))
        except ValueError as e:
            print(f"Error processing {report_file}: {e}")

    print("ERRANT F0.5 Scores by Condition:")
    print("Condition\tF0.5")
    print("-" * 25)

    for condition, score in results:
        print(f"{condition}\t{score:.2f}")

    prompt_types = {}

    for condition, score in results:
        # Handle baseline separately
        if "_v" not in condition:
            prompt_type = condition
            version = None
        else:
            prompt_type, version = condition.split("_v", 1)

        if prompt_type not in prompt_types:
            prompt_types[prompt_type] = []

        prompt_types[prompt_type].append((condition, version, score))

    print("\nBy Prompt Type:")

    for prompt_type, scores in prompt_types.items():
        print(f"\n{prompt_type.upper()}:")

        for condition, version, score in scores:
            if version is None:
                print(f"  {condition}: {score:.2f}")
            else:
                print(f"  v{version}: {score:.2f}")

        avg_score = sum(score for _, _, score in scores) / len(scores)
        print(f"  Average: {avg_score:.2f}")


if __name__ == "__main__":
    main()