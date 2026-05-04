"""Create a human validation sample for checking OCI.

The sample is drawn from Experiment 2 sentence-level OCI outputs. It contains
20 low-OCI, 20 medium-OCI, and 20 high-OCI examples, with stratification across
prompt conditions where possible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCI_PATH = PROJECT_ROOT / "outputs" / "experiment_2_compare_prompts" / "oci_eval" / "per_sentence_oci.csv"
OUTPUTS_JSONL = PROJECT_ROOT / "outputs" / "experiment_2_compare_prompts" / "experiment2_outputs.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
SAMPLE_OUT = RESULTS_DIR / "human_validation_sample.csv"
INSTRUCTIONS_OUT = RESULTS_DIR / "human_validation_instructions.md"

SEED = 1234
CONDITIONS = ["baseline", "instruction", "role", "fewshot"]
OCI_BANDS = ["low", "medium", "high"]
SAMPLES_PER_BAND = 20
BAND_SEED_OFFSET = {"low": 100, "medium": 200, "high": 300}
CONDITION_SEED_OFFSET = {"baseline": 1, "instruction": 2, "role": 3, "fewshot": 4}

ALLOWED_LABELS = [
    "minimal_correct_correction",
    "acceptable_useful_rewrite",
    "over_correction",
    "meaning_change",
    "under_correction_or_error",
]


def load_references() -> pd.DataFrame:
    records: list[dict[str, str]] = []
    with OUTPUTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "sentence_id": obj.get("sentence_id", ""),
                    "condition": obj.get("prompt_id", ""),
                    "reference_correction": obj.get("reference", ""),
                }
            )

    refs = pd.DataFrame(records).drop_duplicates(["sentence_id", "condition"])
    return refs


def assign_oci_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["oci_band"] = pd.qcut(
        df["oci"],
        q=3,
        labels=OCI_BANDS,
        duplicates="drop",
    )
    if df["oci_band"].isna().any():
        raise ValueError("Could not assign OCI bands for all rows.")
    return df


def sample_band(band_df: pd.DataFrame, band: str) -> pd.DataFrame:
    per_condition = SAMPLES_PER_BAND // len(CONDITIONS)
    sampled_parts: list[pd.DataFrame] = []
    used_indices: set[int] = set()

    for condition in CONDITIONS:
        group = band_df[band_df["condition"] == condition]
        if group.empty:
            continue
        n = min(per_condition, len(group))
        part = group.sample(
            n=n,
            random_state=SEED + BAND_SEED_OFFSET[band] + CONDITION_SEED_OFFSET[condition],
        )
        sampled_parts.append(part)
        used_indices.update(part.index.tolist())

    sampled = pd.concat(sampled_parts, axis=0) if sampled_parts else pd.DataFrame(columns=band_df.columns)

    remaining_needed = SAMPLES_PER_BAND - len(sampled)
    if remaining_needed > 0:
        remaining = band_df.drop(index=list(used_indices), errors="ignore")
        if len(remaining) < remaining_needed:
            raise ValueError(
                f"Not enough rows to sample {SAMPLES_PER_BAND} examples for OCI band {band!r}."
            )
        fill = remaining.sample(n=remaining_needed, random_state=SEED + BAND_SEED_OFFSET[band])
        sampled = pd.concat([sampled, fill], axis=0)

    return sampled


def create_sample() -> pd.DataFrame:
    if not OCI_PATH.exists():
        raise FileNotFoundError(f"Missing OCI file: {OCI_PATH}")
    if not OUTPUTS_JSONL.exists():
        raise FileNotFoundError(f"Missing output JSONL file: {OUTPUTS_JSONL}")

    df = pd.read_csv(OCI_PATH)
    required = {
        "sentence_id",
        "condition",
        "source",
        "output",
        "oci",
        "word_levenshtein",
        "stylometric_cosine",
        "delta_fluency",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {OCI_PATH}: {sorted(missing)}")

    refs = load_references()
    df = df.merge(refs, on=["sentence_id", "condition"], how="left")
    df["reference_correction"] = df["reference_correction"].fillna("")
    df = assign_oci_bands(df)

    sampled = pd.concat(
        [sample_band(df[df["oci_band"] == band], band) for band in OCI_BANDS],
        axis=0,
    )
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sampled.insert(0, "example_id", [f"HV{idx:03d}" for idx in range(1, len(sampled) + 1)])

    out = pd.DataFrame(
        {
            "example_id": sampled["example_id"],
            "oci_band": sampled["oci_band"].astype(str),
            "prompt_condition": sampled["condition"],
            "original_sentence": sampled["source"],
            "model_output": sampled["output"],
            "reference_correction": sampled["reference_correction"],
            "OCI": sampled["oci"],
            "edit_distance": sampled["word_levenshtein"],
            "similarity": sampled["stylometric_cosine"],
            "fluency_delta": sampled["delta_fluency"],
            "human_label": "",
            "human_notes": "",
        }
    )
    return out


def write_instructions() -> None:
    labels = "\n".join(f"- `{label}`" for label in ALLOWED_LABELS)
    text = f"""# Human Validation Instructions for OCI

This file explains how to annotate `results/human_validation_sample.csv`.

## Purpose

The aim is to manually check whether examples with higher OCI are genuinely more likely to show over-correction. OCI should be treated as a comparative signal, not as a final judgement.

## What to Read

For each row, compare:

1. `original_sentence`: the learner's original sentence.
2. `model_output`: the model correction.
3. `reference_correction`: the dataset reference correction, where available.
4. `OCI`, `edit_distance`, `similarity`, and `fluency_delta`: automatic signals for context only.

Do not label an example purely because its OCI is high or low. Use the text comparison as the main evidence.

## Allowed Labels

Use exactly one of the following labels in the `human_label` column:

{labels}

## Label Meanings

- `minimal_correct_correction`: the model fixes the grammatical problem and stays close to the learner's wording.
- `acceptable_useful_rewrite`: the model rewrites more than minimally, but the rewrite is useful, meaning-preserving, and still appropriate.
- `over_correction`: the model makes unnecessary wording, style, or structure changes beyond grammatical correction.
- `meaning_change`: the model changes, removes, or adds meaning compared with the original sentence.
- `under_correction_or_error`: the model leaves important errors uncorrected or introduces a new error.

## Annotation Notes

Use `human_notes` to briefly explain difficult cases. For example:

- "Corrects grammar but changes vocabulary unnecessarily."
- "Meaning preserved; rewrite improves clarity."
- "Reference also rewrites heavily, so judgement is uncertain."
- "Output leaves the main verb error uncorrected."

## Consistency Rules

- Prefer `minimal_correct_correction` when the output makes only necessary grammatical changes.
- Prefer `acceptable_useful_rewrite` when extra changes improve clarity without hiding the learner's original meaning.
- Prefer `over_correction` when extra changes are stylistic, paraphrastic, or unnecessary for GEC.
- Prefer `meaning_change` when the correction changes the content, not just the style.
- Prefer `under_correction_or_error` when the correction is incomplete or introduces a new grammar/meaning problem.

## Sampling

The sample contains 60 examples selected with fixed random seed `{SEED}`:

- 20 low-OCI examples
- 20 medium-OCI examples
- 20 high-OCI examples

The sampler attempts to include all prompt conditions within each OCI band.
"""
    INSTRUCTIONS_OUT.write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = create_sample()
    sample.to_csv(SAMPLE_OUT, index=False)
    write_instructions()

    print(f"Saved human validation sample to: {SAMPLE_OUT}")
    print(f"Saved annotation instructions to: {INSTRUCTIONS_OUT}")
    print("\nSample counts by OCI band and prompt condition:")
    print(sample.groupby(["oci_band", "prompt_condition"]).size().to_string())


if __name__ == "__main__":
    main()
