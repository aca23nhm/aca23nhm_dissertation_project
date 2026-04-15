from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRANT_OUT = ROOT / "outputs" / "errant_outputs"
CSV_OUT = ERRANT_OUT / "error_type_counts.csv"

def iter_m2_edits(m2_path: Path):
    """
    Yield (sent_idx, err_type) from an M2 file.
    Skips 'noop' edits.
    """
    sent_idx = -1
    for line in m2_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("S "):
            sent_idx += 1
        elif line.startswith("A "):
            # Format: A o_start o_end|||TYPE|||correction|||REQUIRED|||...
            parts = line.split("|||")
            if len(parts) >= 2:
                err_type = parts[1].strip()
                if err_type != "noop":
                    yield sent_idx, err_type

def main() -> None:
    if not ERRANT_OUT.exists():
        raise FileNotFoundError(f"Missing directory: {ERRANT_OUT}. Run src/run_errant.py first.")

    rows = []
    # For each condition, compare hyp vs ref counts
    for ref_m2 in sorted(ERRANT_OUT.glob("*.ref.m2")):
        condition = ref_m2.name.replace(".ref.m2", "")
        hyp_m2 = ERRANT_OUT / f"{condition}.hyp.m2"
        if not hyp_m2.exists():
            continue

        gold_counts = Counter(err for _, err in iter_m2_edits(ref_m2))
        sys_counts = Counter(err for _, err in iter_m2_edits(hyp_m2))

        all_types = sorted(set(gold_counts) | set(sys_counts))
        for t in all_types:
            rows.append({
                "condition": condition,
                "error_type": t,
                "gold_count": gold_counts.get(t, 0),
                "system_count": sys_counts.get(t, 0),
            })

    # Write CSV
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "error_type", "gold_count", "system_count"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {CSV_OUT}")
    print(f"Rows: {len(rows)}")

if __name__ == "__main__":
    main()