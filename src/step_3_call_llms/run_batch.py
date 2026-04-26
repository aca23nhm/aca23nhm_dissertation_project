from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Set, Iterable

from src.step_1_dataset_loader.data_loader import load_dataset, get_by_ids, Item
from src.step_2_prompt_manager.prompt_manager import load_prompts, render_prompt
from src.step_3_call_llms.model_runner import load_model_config, ModelRunner
from src.step_3_call_llms.save_results import append_jsonl

DATASET_CSV = Path("data/processed/wi_locness_sentences.csv")
L2_IDS_JSON = Path("data/processed/all_5000_subset_ids.json")
OUT_JSONL = Path("outputs/model_outputs.jsonl")


def load_done_pairs(path: Path) -> Set[tuple[str, str]]:
    """
    Read existing outputs JSONL (if any) and return completed
    (sentence_id, prompt_id) pairs for resume support.
    """
    done: Set[tuple[str, str]] = set()

    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("sentence_id")
                pid = obj.get("prompt_id")
                if sid and pid:
                    done.add((sid, pid))
            except json.JSONDecodeError:
                continue

    return done


def iter_l2_items() -> Iterable[Item]:
    """
    Only run the L2-only subset listed in l2_subset_ids.json
    to avoid accidentally calling the full dataset.
    """
    if not L2_IDS_JSON.exists():
        raise FileNotFoundError(
            f"L2 subset id file not found: {L2_IDS_JSON}\n"
            "Refusing to run full dataset to protect API credits.\n"
            "Generate it first by running your create_subset script."
        )

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATASET_CSV}\n"
            "Make sure wi_locness_sentences.csv exists."
        )

    items = load_dataset(DATASET_CSV)
    ids = json.loads(L2_IDS_JSON.read_text(encoding="utf-8"))

    if not isinstance(ids, list):
        raise ValueError(
            f"Expected a JSON list of sentence IDs in {L2_IDS_JSON}, "
            f"but got {type(ids).__name__}."
        )

    selected_items = get_by_ids(items, ids)

    if not selected_items:
        raise ValueError(
            f"No matching dataset items found for IDs in {L2_IDS_JSON}."
        )

    return selected_items


def main() -> None:
    templates = load_prompts("prompts")
    cfg = load_model_config("configs/model.yaml")
    runner = ModelRunner(cfg)

    items = list(iter_l2_items())
    done = load_done_pairs(OUT_JSONL)

    print(f"Loaded L2 subset from: {L2_IDS_JSON}")
    print(f"Loaded {len(items)} sentences.")
    print(f"Already completed pairs (resume): {len(done)}")
    print(f"Prompts loaded: {list(templates.keys())}")

    total = len(items) * len(templates)
    completed = len(done)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    for it in items:
        for prompt_id, template in templates.items():
            key = (it.sentence_id, prompt_id)
            if key in done:
                continue

            rendered = render_prompt(template, it.source)

            rec: Dict[str, Any] = runner.run_one(
                sentence_id=it.sentence_id,
                prompt_id=prompt_id,
                rendered_prompt=rendered,
            )

            rec["source"] = it.source
            rec["reference"] = it.reference

            append_jsonl(str(OUT_JSONL), rec)

            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"Progress: {completed}/{total}")

    print("Done.")


if __name__ == "__main__":
    main()