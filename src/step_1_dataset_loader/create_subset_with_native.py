from pathlib import Path
import json

from src.step_1_dataset_loader.data_loader import load_dataset, sample_subset

DATASET_CSV = Path("data/processed/wi_locness_sentences.csv")
OUT_IDS = Path("data/processed/all_5000_subset_ids.json")

N = 5000
SEED = 42

# Load all items (NO filtering)
items = load_dataset(DATASET_CSV)

# Ensure enough data
if len(items) < N:
    raise ValueError(
        f"Not enough items to sample {N}. "
        f"Found only {len(items)} in the dataset."
    )

# Sample from FULL dataset (includes native + L2)
subset = sample_subset(items, n=N, seed=SEED)

# Save sampled sentence IDs
OUT_IDS.parent.mkdir(parents=True, exist_ok=True)
OUT_IDS.write_text(
    json.dumps([it.sentence_id for it in subset], indent=2),
    encoding="utf-8"
)

# Summary (optional, for analysis)
native_count = sum(
    1
    for it in subset
    if it.sentence_id.startswith("N.") or it.sentence_id.startswith("ABCN.")
)
l2_count = len(subset) - native_count

print(f"Saved {len(subset)} mixed ids to {OUT_IDS}")
print(f"L2 items: {l2_count}")
print(f"Native items: {native_count}")