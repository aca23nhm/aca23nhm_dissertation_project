# src/run_batch.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Set, Iterable

from src.step_1_dataset_loader.data_loader import load_dataset, get_by_ids, Item
from src.step_2_prompt_manager.prompt_manager import load_prompts, render_prompt
from src.step_3_call_llms.model_runner import load_model_config, ModelRunner
from src.step_3_call_llms.save_results import append_jsonl

DATASET_CSV = Path("data/processed/wi_locness_sentences.csv")
FIXED_IDS_JSON = Path("data/processed/fixed_subset_ids.json")
OUT_JSONL = Path("outputs/model_outputs.jsonl")


def load_done_pairs(path: Path) -> Set[tuple[str, str]]:
    """
    Read existing outputs JSONL (if any) and return completed (sentence_id, prompt_id) pairs.
    This allows resume after crash.
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
                # Skip broken lines (rare, but possible if crash mid-write)
                continue

    return done


def iter_fixed_items() -> Iterable[Item]:
    """
    SAFETY: Only run the fixed subset to protect API credits.
    If fixed subset file is missing, refuse to run.
    """
    if not FIXED_IDS_JSON.exists():
        raise FileNotFoundError(
            f"Fixed subset id file not found: {FIXED_IDS_JSON}\n"
            "Refusing to run full dataset to protect API credits.\n"
            "Generate it first (e.g., run your fixed subset script)."
        )

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATASET_CSV}\n"
            "Make sure you generated wi_locness_sentences.csv first."
        )

    items = load_dataset(DATASET_CSV)
    ids = json.loads(FIXED_IDS_JSON.read_text(encoding="utf-8"))
    return get_by_ids(items, ids)


def main() -> None:
    templates = load_prompts("prompts")
    cfg = load_model_config("configs/model.yaml")
    runner = ModelRunner(cfg)

    # Only fixed subset (safe)
    items = list(iter_fixed_items())

    done = load_done_pairs(OUT_JSONL)

    print(f"Loaded FIXED subset: {len(items)} sentences.")
    print(f"Already completed pairs (resume): {len(done)}")
    print(f"Prompts loaded: {list(templates.keys())}")

    total = len(items) * len(templates)
    completed = len(done)

    for it in items:
        for prompt_id, template in templates.items():
            key = (it.sentence_id, prompt_id)
            if key in done:
                continue  # resume support

            rendered = render_prompt(template, it.source)

            rec: Dict[str, Any] = runner.run_one(
                sentence_id=it.sentence_id,
                prompt_id=prompt_id,
                rendered_prompt=rendered,
            )

            # Include source/reference too (useful later for ERRANT + style metrics)
            rec["source"] = it.source
            rec["reference"] = it.reference

            append_jsonl(str(OUT_JSONL), rec)

            completed += 1
            if completed % 20 == 0:
                print(f"Progress: {completed}/{total}")

    print("Done.")


if __name__ == "__main__":
    main()