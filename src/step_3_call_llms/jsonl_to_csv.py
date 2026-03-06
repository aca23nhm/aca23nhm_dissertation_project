import json
import csv
from pathlib import Path

IN_PATH = Path("outputs/model_outputs.jsonl")
OUT_PATH = Path("outputs/model_outputs.csv")

with IN_PATH.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

# Ensure stable column order
fieldnames = sorted({k for r in rows for k in r.keys()})

with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"Saved: {OUT_PATH} ({len(rows)} rows)")