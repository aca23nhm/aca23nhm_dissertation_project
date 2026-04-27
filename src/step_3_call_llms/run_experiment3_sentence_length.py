from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from src.step_1_dataset_loader.data_loader import Item, load_dataset
from src.step_2_prompt_manager.prompt_manager import load_prompts, render_prompt
from src.step_3_call_llms.model_runner import ModelRunner, load_model_config
from src.step_3_call_llms.save_results import append_jsonl

TARGET_SENTENCES_PER_GROUP = 150
SEED = 1234

LENGTH_BUCKETS = {
    "short": range(1, 11),
    "medium": range(11, 21),
    "long": range(21, 1000),
}

DEFAULT_EXISTING_JSONL_PATHS = [
    Path("outputs/experiment_2_compare_prompts/experiment2_outputs.jsonl"),
]
DEFAULT_DATASET_CSV = Path("data/processed/experiment_1/experiment1_500_sentences.csv")
DEFAULT_PROMPTS_DIR = Path("prompts/experiment2")
DEFAULT_OUTPUT_DIR = Path("outputs/experiment_3_sentence_length")
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "experiment3_sentence_length_outputs.jsonl"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "outputs_with_length_groups.csv"


def length_group_from_source(source: str) -> str:
    token_length = len(source.split())
    for group_name, token_range in LENGTH_BUCKETS.items():
        if token_length in token_range:
            return group_name
    return "unknown"


def load_jsonl_records(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue
    return records


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["source"] = df["source"].astype(str).fillna("").str.strip()
    df["reference"] = df["reference"].astype(str).fillna("").str.strip()
    df["token_length"] = df["source"].apply(lambda text: len(text.split()))
    df["length_group"] = df["source"].apply(length_group_from_source)
    return df


def summarize_unique_sentence_ids(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {group: 0 for group in LENGTH_BUCKETS}
    return (
        df.drop_duplicates(subset=["sentence_id"]).groupby("length_group")["sentence_id"].nunique().reindex(LENGTH_BUCKETS.keys(), fill_value=0).to_dict()
    )


def resolve_candidate_dataset_paths(path: Path) -> List[Path]:
    if path.is_dir():
        csv_paths = sorted(path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return csv_paths
    if path.exists():
        return [path]
    raise FileNotFoundError(f"Dataset path not found: {path}")


def load_candidate_items(dataset_path: Path) -> List[Item]:
    candidate_paths = resolve_candidate_dataset_paths(dataset_path)
    items: List[Item] = []
    seen_ids: set[str] = set()
    for csv_path in candidate_paths:
        for item in load_dataset(csv_path):
            if item.sentence_id not in seen_ids:
                items.append(item)
                seen_ids.add(item.sentence_id)
    return items


def summarize_output_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["prompt_id", "length_group", "record_count"]).astype(object)
    return (
        df.groupby(["prompt_id", "length_group"])
        .size()
        .reset_index(name="record_count")
        .sort_values(["prompt_id", "length_group"])
    )


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_missing_items(
    dataset_items: Sequence[Item],
    existing_sentence_ids: set[str],
    missing_by_group: Dict[str, int],
    seed: int,
) -> List[Item]:
    grouped: Dict[str, List[Item]] = {group: [] for group in LENGTH_BUCKETS}
    for item in dataset_items:
        if item.sentence_id in existing_sentence_ids:
            continue
        group = length_group_from_source(item.source)
        if group in grouped:
            grouped[group].append(item)

    rng = random.Random(seed)
    selected: List[Item] = []
    for group, missing_count in missing_by_group.items():
        candidates = grouped.get(group, [])
        if missing_count <= 0:
            continue
        if not candidates:
            print(
                f"Warning: no candidate sentences remain for length group '{group}' after excluding existing ids."
            )
            continue
        if missing_count > len(candidates):
            print(
                f"Warning: target missing count for '{group}' is {missing_count}, but only {len(candidates)} candidates are available. "
                "Using all available candidate sentences."
            )
            selected.extend(rng.sample(candidates, len(candidates)))
            continue
        selected.extend(rng.sample(candidates, missing_count))
    return selected


def choose_sentence_ids_for_group(sentence_ids: list[str], target: int, seed: int) -> set[str]:
    if len(sentence_ids) <= target:
        return set(sentence_ids)
    rng = random.Random(seed)
    return set(rng.sample(sentence_ids, target))


def get_selected_sentence_ids(existing_df: pd.DataFrame, target: int, seed: int) -> Dict[str, set[str]]:
    selected: Dict[str, set[str]] = {}
    for group in LENGTH_BUCKETS:
        ids = sorted(existing_df.loc[existing_df["length_group"] == group, "sentence_id"].unique())
        selected[group] = choose_sentence_ids_for_group(ids, target, seed)
    return selected


def build_new_records(
    items: Sequence[Item],
    templates: Dict[str, str],
    runner: ModelRunner,
    existing_pairs: set[tuple[str, str]],
) -> List[Dict[str, Any]]:
    new_records: List[Dict[str, Any]] = []
    for item in items:
        for prompt_id, template in templates.items():
            pair = (item.sentence_id, prompt_id)
            if pair in existing_pairs:
                continue
            rendered_prompt = render_prompt(template, item.source)
            rec = runner.run_one(
                sentence_id=item.sentence_id,
                prompt_id=prompt_id,
                rendered_prompt=rendered_prompt,
            )
            rec["source"] = item.source
            rec["reference"] = item.reference
            rec["token_length"] = len(item.source.split())
            rec["length_group"] = length_group_from_source(item.source)
            new_records.append(rec)
            existing_pairs.add(pair)
    return new_records


def print_counts(title: str, counts: Dict[str, int]) -> None:
    print(title)
    for group, count in counts.items():
        print(f"  {group:>6}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 3 sentence-length outputs with reuse of previous JSONL records."
    )
    parser.add_argument(
        "--prompt-dir",
        type=Path,
        default=DEFAULT_PROMPTS_DIR,
        help="Directory containing prompt templates for Experiment 3.",
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=DEFAULT_DATASET_CSV,
        help="Original dataset CSV used for sentence selection.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Experiment 3 JSONL output file path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Experiment 3 grouped CSV output file path.",
    )
    args = parser.parse_args()
    templates = load_prompts(args.prompt_dir)

    existing_records = [
        rec for rec in load_jsonl_records(DEFAULT_EXISTING_JSONL_PATHS)
        if rec.get("prompt_id") in templates
    ]
    experiment3_records = [
        rec for rec in load_jsonl_records([args.output_jsonl])
        if args.output_jsonl.exists() and rec.get("prompt_id") in templates
    ] if args.output_jsonl.exists() else []

    combined_records = {(
        rec["sentence_id"], rec["prompt_id"]
    ): rec for rec in existing_records + experiment3_records}

    existing_df = build_dataframe(list(combined_records.values()))
    selected_sentence_ids_by_group = get_selected_sentence_ids(existing_df, TARGET_SENTENCES_PER_GROUP, SEED)
    selected_ids = set().union(*selected_sentence_ids_by_group.values())
    missing_counts = {
        group: max(0, TARGET_SENTENCES_PER_GROUP - len(selected_sentence_ids_by_group[group]))
        for group in LENGTH_BUCKETS
    }

    print("Prompt set:", list(templates.keys()))
    print_counts("Selected unique sentence IDs by length group:", {g: len(ids) for g, ids in selected_sentence_ids_by_group.items()})
    print_counts("Missing sentence counts by length group:", missing_counts)

    generate_items: list[Item] = []
    if sum(missing_counts.values()) > 0:
        if not args.dataset_csv.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {args.dataset_csv}")

        dataset_items = load_candidate_items(args.dataset_csv)
        existing_sentence_ids = set(existing_df["sentence_id"].unique())

        available_items = [
            item for item in dataset_items
            if item.sentence_id not in existing_sentence_ids
        ]
        available_df = build_dataframe([
            {"sentence_id": item.sentence_id, "source": item.source, "reference": item.reference}
            for item in available_items
        ])
        available_counts = summarize_unique_sentence_ids(available_df)
        print_counts("Available candidate sentence counts by length group:", available_counts)

        if any(available_counts[group] < missing_counts[group] for group in LENGTH_BUCKETS):
            fallback_source = Path("data/processed")
            if fallback_source.exists() and fallback_source.is_dir():
                print("Falling back to all CSVs in data/processed to find more candidates.")
                dataset_items = load_candidate_items(fallback_source)
                available_items = [
                    item for item in dataset_items
                    if item.sentence_id not in existing_sentence_ids
                ]
                available_df = build_dataframe([
                    {"sentence_id": item.sentence_id, "source": item.source, "reference": item.reference}
                    for item in available_items
                ])
                available_counts = summarize_unique_sentence_ids(available_df)
                print_counts("Fallback available candidate sentence counts by length group:", available_counts)

        generate_items = select_missing_items(
            dataset_items=available_items,
            existing_sentence_ids=existing_sentence_ids,
            missing_by_group=missing_counts,
            seed=SEED,
        )

    print(f"Generating outputs for {len(generate_items)} new sentences.")

    cfg = load_model_config("configs/model.yaml")
    runner = ModelRunner(cfg)
    templates = load_prompts(args.prompt_dir)

    new_records = build_new_records(
        items=generate_items,
        templates=templates,
        runner=runner,
        existing_pairs=set(combined_records.keys()),
    )

    if new_records:
        combined_records.update({(rec["sentence_id"], rec["prompt_id"]): rec for rec in new_records})
        print(f"Generated {len(new_records)} new output records.")
    else:
        print("No new records were generated.")

    final_records = [
        rec for rec in combined_records.values()
        if rec["sentence_id"] in selected_ids or any(rec["sentence_id"] == item.sentence_id for item in generate_items)
    ]

    final_df = build_dataframe(final_records)
    final_counts = summarize_unique_sentence_ids(final_df)
    print_counts("Final unique sentence counts by length group:", final_counts)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(final_records, args.output_jsonl)
    print(f"Saved consolidated Experiment 3 JSONL to: {args.output_jsonl}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved grouped Experiment 3 output to: {args.output_csv}")

    counts_by_prompt_and_group = (
        final_df.groupby(["prompt_id", "length_group"]).size().reset_index(name="record_count")
    )
    print("Outputs per prompt_id and length_group:")
    print(counts_by_prompt_and_group.to_string(index=False))


if __name__ == "__main__":
    main()
