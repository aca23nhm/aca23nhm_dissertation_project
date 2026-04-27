from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

INPUT_JSONL = Path("outputs/experiment_3_sentence_length/experiment3_sentence_length_outputs.jsonl")
OUTPUT_DIR = Path("outputs/experiment_3_sentence_length/errant_inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def detokenize_punctuation(s: str) -> str:
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)
    s = re.sub(r'"\s+', '"', s)
    s = re.sub(r"\s+\"", '"', s)
    s = re.sub(r"'\s+", "'", s)
    s = re.sub(r"\s+'", "'", s)
    return s


def sanitize(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = detokenize_punctuation(s)
    return s


def length_category(source: str) -> str:
    tokens = source.split()
    if len(tokens) <= 10:
        return "short"
    if len(tokens) <= 20:
        return "medium"
    return "long"


def main() -> None:
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_JSONL}. Run Experiment 3 generation first."
        )

    by_prompt = defaultdict(list)
    skipped = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping line {line_num}: invalid JSON ({e})")
                skipped += 1
                continue

            prompt_id = r.get("prompt_id")
            src = sanitize(r.get("source", ""))
            hyp = sanitize(r.get("clean_output_text", ""))
            ref = sanitize(r.get("reference", ""))
            category = length_category(src)

            if not prompt_id:
                print(f"[WARN] Skipping line {line_num}: missing prompt_id")
                skipped += 1
                continue

            if not src or not hyp or not ref:
                print(f"[WARN] Skipping line {line_num}: empty source/hyp/ref")
                skipped += 1
                continue

            by_prompt[(prompt_id, category)].append((src, hyp, ref))

    for prompt_key, triples in by_prompt.items():
        prompt_id, category = prompt_key
        src_lines = [s for s, _, _ in triples]
        hyp_lines = [h for _, h, _ in triples]
        ref_lines = [t for _, _, t in triples]

        if not (len(src_lines) == len(hyp_lines) == len(ref_lines)):
            raise ValueError(
                f"Line count mismatch for {prompt_id}:{category}: src={len(src_lines)}, hyp={len(hyp_lines)}, ref={len(ref_lines)}"
            )

        file_stem = f"{prompt_id}_{category}"
        src_path = OUTPUT_DIR / f"{file_stem}.src"
        hyp_path = OUTPUT_DIR / f"{file_stem}.hyp"
        ref_path = OUTPUT_DIR / f"{file_stem}.ref"

        src_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8", newline="\n")
        hyp_path.write_text("\n".join(hyp_lines) + "\n", encoding="utf-8", newline="\n")
        ref_path.write_text("\n".join(ref_lines) + "\n", encoding="utf-8", newline="\n")

        print(f"{prompt_id}_{category} => {len(triples)} examples")

    print(f"\nSaved ERRANT input files to: {OUTPUT_DIR}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()
