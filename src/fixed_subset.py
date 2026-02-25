from pathlib import Path
import json

from data_loader import load_dataset, sample_subset

DATASET_CSV = Path("data/processed/wi_locness_sentences.csv")
OUT_IDS = Path("data/processed/fixed_subset_ids.json")

N = 500
SEED = 42

items = load_dataset(DATASET_CSV)

# Keep ONLY pure L2 learner data
subset_pool = [
    item for item in items
    if (
        not item.sentence_id.startswith("N.") and
        not item.sentence_id.startswith("ABCN.")
    )
]

if len(subset_pool) < N:
    raise ValueError(
        f"Not enough L2-only items to sample {N}. "
        f"Found only {len(subset_pool)} after filtering."
    )

subset = sample_subset(subset_pool, n=N, seed=SEED)

OUT_IDS.parent.mkdir(parents=True, exist_ok=True)
OUT_IDS.write_text(
    json.dumps([it.sentence_id for it in subset], indent=2),
    encoding="utf-8"
)

print(f"Saved {len(subset)} L2-only ids to {OUT_IDS}")