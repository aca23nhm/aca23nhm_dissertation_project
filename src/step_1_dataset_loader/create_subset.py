from pathlib import Path
import json

from src.step_1_dataset_loader.data_loader import load_dataset, sample_subset

DATASET_CSV = Path("data/processed/wi_locness_sentences.csv")
OUT_IDS = Path("data/processed/l2_5000_subset_ids.json")

N = 5000
SEED = 42

items = load_dataset(DATASET_CSV)

# Keep ONLY pure L2 learner data
l2_pool = [
    item for item in items
    if (
        not item.sentence_id.startswith("N.")
        and not item.sentence_id.startswith("ABCN.")
    )
]

if len(l2_pool) < N:
    raise ValueError(
        f"Not enough L2-only items to sample {N}. "
        f"Found only {len(l2_pool)} after filtering."
    )

subset = sample_subset(l2_pool, n=N, seed=SEED)

OUT_IDS.parent.mkdir(parents=True, exist_ok=True)
OUT_IDS.write_text(
    json.dumps([it.sentence_id for it in subset], indent=2),
    encoding="utf-8"
)

print(f"Saved {len(subset)} L2-only ids to {OUT_IDS}")