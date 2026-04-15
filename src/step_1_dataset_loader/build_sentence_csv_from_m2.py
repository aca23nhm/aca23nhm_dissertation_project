import os
import glob
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class M2Edit:
    start: int
    end: int
    cor: str
    annotator: int

def parse_m2_edit(line: str) -> Optional[M2Edit]:
    """
    Parse an M2 'A' line:
    A start end|||TYPE|||correction|||REQUIRED|||...|||annotator_id
    """
    if not line.startswith("A "):
        return None
    # Remove leading "A "
    rest = line[2:].strip()
    parts = rest.split("|||")
    if len(parts) < 2:
        return None

    # span is: "start end"
    span = parts[0].strip().split()
    start, end = int(span[0]), int(span[1])

    # correction string is usually parts[2], but be defensive:
    cor = parts[2].strip() if len(parts) > 2 else ""
    # annotator is last field
    try:
        annotator = int(parts[-1].strip())
    except ValueError:
        annotator = 0

    return M2Edit(start=start, end=end, cor=cor, annotator=annotator)

def apply_edits_to_tokens(tokens: List[str], edits: List[M2Edit]) -> List[str]:
    """
    Apply edits to token list.
    We apply from right-to-left (descending start index) to avoid index shift.
    Rules:
    - If cor == "-NONE-" => delete tokens[start:end]
    - Else replace tokens[start:end] with cor tokens (cor may be empty for deletion)
    - Insertions are usually start == end, cor non-empty => insert at start
    """
    # Sort by start desc, end desc
    edits_sorted = sorted(edits, key=lambda e: (e.start, e.end), reverse=True)

    out = tokens[:]
    for e in edits_sorted:
        cor = e.cor
        if cor == "-NONE-":
            cor_tokens = []
        else:
            # Sometimes multiple alternatives exist; take the first variant
            # Common separators in some M2 variants: "||" or "|||", but we already split by "|||"
            cor = cor.split("||")[0].strip()
            cor_tokens = cor.split() if cor.strip() else []

        # Replace span [start:end] with cor_tokens
        out[e.start:e.end] = cor_tokens

    return out

def iter_m2_sentences(m2_path: str):
    """
    Yields tuples: (source_sentence_str, edits_list)
    Sentences are separated by blank lines.
    """
    with open(m2_path, "r", encoding="utf-8") as f:
        block_lines = []
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if block_lines:
                    yield block_lines
                    block_lines = []
            else:
                block_lines.append(line)
        if block_lines:
            yield block_lines

def build_csv_from_m2_folder(
    m2_folder: str,
    output_csv: str,
    annotator_id: int = 0
):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    m2_files = sorted(glob.glob(os.path.join(m2_folder, "*.m2")))
    if not m2_files:
        raise FileNotFoundError(f"No .m2 files found in: {m2_folder}")

    rows = []
    for m2_file in m2_files:
        base = os.path.splitext(os.path.basename(m2_file))[0]
        sent_idx = 0

        for block in iter_m2_sentences(m2_file):
            # Find source line
            s_lines = [ln for ln in block if ln.startswith("S ")]
            if not s_lines:
                continue
            source = s_lines[0][2:].strip()  # remove leading "S "
            source_tokens = source.split()

            # Collect edits for chosen annotator
            edits = []
            for ln in block:
                if ln.startswith("A "):
                    e = parse_m2_edit(ln)
                    if e and e.annotator == annotator_id:
                        edits.append(e)

            # Apply edits to reconstruct gold reference
            ref_tokens = apply_edits_to_tokens(source_tokens, edits)
            reference = " ".join(ref_tokens)

            sent_idx += 1
            sentence_id = f"{base}_{sent_idx:06d}"

            rows.append((sentence_id, source, reference))

    # Write CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence_id", "source", "reference"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_csv}")

if __name__ == "__main__":
    # Paths for output
    build_csv_from_m2_folder(
        m2_folder="data/raw/m2",
        output_csv="data/processed/wi_locness_sentences.csv",
        annotator_id=0
    )