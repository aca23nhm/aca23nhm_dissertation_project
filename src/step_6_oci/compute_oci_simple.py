from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IN_JSONL = ROOT / "outputs" / "experiment1_500_outputs.jsonl"
OUT_DIR = ROOT / "outputs" / "oci_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SENTENCE_OUT = OUT_DIR / "per_sentence_oci_simple.csv"
AGG_OUT = OUT_DIR / "aggregate_oci_simple.csv"

def word_levenshtein(a: str, b: str) -> int:
    """Simple word-level Levenshtein distance."""
    a_tokens = a.split()
    b_tokens = b.split()
    n = len(a_tokens)
    m = len(b_tokens)

    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[m]

def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0

def main() -> None:
    # Load the experiment outputs
    records = []
    with IN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {IN_JSONL}")

    per_sentence_rows = []
    grouped_oci = defaultdict(list)

    for record in records:
        source = record["source"]
        reference = record["reference"]
        hypothesis = record["clean_output_text"]
        condition = record["prompt_id"]

        # Compute edit distances
        src_to_ref = word_levenshtein(source, reference)
        src_to_hyp = word_levenshtein(source, hypothesis)

        # Edit density (normalized by source length)
        src_tokens = len(source.split())
        ref_tokens = len(reference.split())
        hyp_tokens = len(hypothesis.split())

        src_to_ref_density = safe_div(src_to_ref, src_tokens)
        src_to_hyp_density = safe_div(src_to_hyp, src_tokens)

        # OCI: Over-Correction Index
        # Measures how much more the hypothesis differs from source compared to reference
        # Positive OCI means over-correction (hypothesis changes more than needed)
        oci_raw = src_to_hyp - src_to_ref

        # Normalize OCI to [0,1] range using min-max across all records
        # We'll collect all values first, then normalize

        per_sentence_rows.append({
            "condition": condition,
            "source": source,
            "reference": reference,
            "hypothesis": hypothesis,
            "src_to_ref_edits": src_to_ref,
            "src_to_hyp_edits": src_to_hyp,
            "src_tokens": src_tokens,
            "ref_tokens": ref_tokens,
            "hyp_tokens": hyp_tokens,
            "src_to_ref_density": src_to_ref_density,
            "src_to_hyp_density": src_to_hyp_density,
            "oci_raw": oci_raw,
        })

    # Now normalize OCI values
    oci_raw_vals = [r["oci_raw"] for r in per_sentence_rows]
    oci_min = min(oci_raw_vals)
    oci_max = max(oci_raw_vals)

    print(f"OCI raw values: min={oci_min}, max={oci_max}")

    for row in per_sentence_rows:
        # Normalize OCI to [0,1]
        if oci_max > oci_min:
            oci_norm = (row["oci_raw"] - oci_min) / (oci_max - oci_min)
        else:
            oci_norm = 0.0

        row["oci_normalized"] = oci_norm
        row["oci_percent"] = oci_norm * 100

        grouped_oci[row["condition"]].append(oci_norm)

    # Save per-sentence results
    fieldnames = list(per_sentence_rows[0].keys())
    with PER_SENTENCE_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sentence_rows)

    # Save aggregate results
    agg_rows = []
    for condition, ocis in sorted(grouped_oci.items()):
        mean_oci = sum(ocis) / len(ocis) if ocis else 0.0
        agg_rows.append({
            "condition": condition,
            "n_sentences": len(ocis),
            "mean_oci_normalized": mean_oci,
            "mean_oci_percent": mean_oci * 100,
            "min_oci": min(ocis) if ocis else 0.0,
            "max_oci": max(ocis) if ocis else 0.0,
        })

    with AGG_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "n_sentences", "mean_oci_normalized", "mean_oci_percent", "min_oci", "max_oci"])
        writer.writeheader()
        writer.writerows(agg_rows)

    print(f"Saved per-sentence OCI to {PER_SENTENCE_OUT}")
    print(f"Saved aggregate OCI to {AGG_OUT}")

    # Print summary
    print("\nAggregate OCI Results:")
    print("Condition\tN\tMean OCI %")
    print("-" * 30)
    for row in agg_rows:
        print(f"{row['condition']}\t{row['n_sentences']}\t{row['mean_oci_percent']:.1f}")

if __name__ == "__main__":
    main()