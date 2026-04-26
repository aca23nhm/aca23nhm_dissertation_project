from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Set, Iterable

from src.step_1_dataset_loader.data_loader import load_dataset, Item
from src.step_2_prompt_manager.prompt_manager import load_prompts, render_prompt
from src.step_3_call_llms.model_runner import load_model_config, ModelRunner
from src.step_3_call_llms.save_results import append_jsonl

# Experiment 2 configuration
DATASET_CSV = Path("data/processed/experiment_1/experiment1_500_sentences.csv")
PROMPTS_DIR = Path("prompts/experiment2")
OUT_JSONL = Path("outputs/experiment_2/experiment2_outputs.jsonl")


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


def iter_experiment2_items() -> Iterable[Item]:
    """
    Load the 500-sentence experiment 2 dataset (same as experiment 1).
    """
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Experiment 2 dataset CSV not found: {DATASET_CSV}\n"
            "Make sure experiment1_500_sentences.csv exists."
        )

    items = load_dataset(DATASET_CSV)

    if not items:
        raise ValueError(
            f"No dataset items found in {DATASET_CSV}."
        )

    return items


def main() -> None:
    templates = load_prompts(PROMPTS_DIR)
    cfg = load_model_config("configs/model.yaml")
    runner = ModelRunner(cfg)

    items = list(iter_experiment2_items())
    done = load_done_pairs(OUT_JSONL)

    print(f"Loaded Experiment 2 dataset from: {DATASET_CSV}")
    print(f"Loaded {len(items)} sentences.")
    print(f"Already completed pairs (resume): {len(done)}")
    print(f"Prompts loaded: {list(templates.keys())}")

    total = len(items) * len(templates)
    completed = len(done)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    for it in items:
        for prompt_id, template in templates.items():
            pair = (it.sentence_id, prompt_id)

            if pair in done:
                continue

            rendered = render_prompt(template, it.source)

            try:
                rec = runner.run_one(
                    sentence_id=it.sentence_id,
                    prompt_id=prompt_id,
                    rendered_prompt=rendered,
                )

                # Add source and reference for evaluation
                rec["source"] = it.source
                rec["reference"] = it.reference

                append_jsonl(OUT_JSONL, rec)
                completed += 1

                print(f"[{completed}/{total}] {it.sentence_id} + {prompt_id}")

            except Exception as e:
                print(f"ERROR on {it.sentence_id} + {prompt_id}: {e}")
                continue

    print(f"Experiment 2 completed. Results saved to: {OUT_JSONL}")


if __name__ == "__main__":
    main()