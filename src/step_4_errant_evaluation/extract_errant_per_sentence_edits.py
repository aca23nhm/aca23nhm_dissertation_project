from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRANT_OUT = ROOT / "outputs" / "errant_outputs"
CSV_OUT = ERRANT_OUT / "per_sentence_edits.csv"

def parse_m2(m2_path: Path):
    """
    Parse M2 into per-sentence rows.
    Returns list of dict rows with: sent_idx, source_sentence, edit fields.
    """
    rows = []
    sent_idx = -1
    current_sent = ""

    for line in m2_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("S "):
            sent_idx += 1
            current_sent = line[2:].strip()
        elif line.startswith("A "):
            # A o_start o_end|||TYPE|||correction|||REQUIRED|||...
            parts = line.split("|||")
            if len(parts) < 4:
                continue

            span = parts[0].split()
            # span: ["A", o_start, o_end]
            o_start = int(span[1]) if len(span) > 1 else -1
            o_end = int(span[2]) if len(span) > 2 else -1

            err_type = parts[1].strip()
            cor = parts[2].strip()
            required = parts[3].strip()

            if err_type == "noop":
                continue

            rows.append({
                "sent_idx": sent_idx,
                "source_sentence": current_sent,
                "o_start": o_start,
                "o_end": o_end,
                "error_type": err_type,
                "correction": cor,
                "required": required,
            })

    return rows

def main() -> None:
    if not ERRANT_OUT.exists():
        raise FileNotFoundError(f"Missing directory: {ERRANT_OUT}. Run src/run_errant.py first.")

    out_rows = []

    # Export both system and gold edits (tagged)
    for m2_path in sorted(ERRANT_OUT.glob("*.m2")):
        name = m2_path.name
        if name.endswith(".hyp.m2"):
            condition = name.replace(".hyp.m2", "")
            role = "system"
        elif name.endswith(".ref.m2"):
            condition = name.replace(".ref.m2", "")
            role = "gold"
        else:
            continue

        rows = parse_m2(m2_path)
        for r in rows:
            r2 = {"condition": condition, "role": role, **r}
            out_rows.append(r2)

    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "role",
                "sent_idx", "source_sentence",
                "o_start", "o_end",
                "error_type", "correction", "required",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"Saved: {CSV_OUT}")
    print(f"Rows: {len(out_rows)}")

if __name__ == "__main__":
    main()