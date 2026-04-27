from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

INPUT_JSONL = Path("outputs/experiment_1_prompt_engineering/experiment1_500_outputs.jsonl")
OUTPUT_DIR = Path("outputs/experiment_1_prompt_engineering/experiment1_errant_inputs")

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def detokenize_punctuation(s: str) -> str:
    """
    Convert tokenised punctuation like:
    'Hi my friend !' -> 'Hi my friend!'
    'For example , this is good .' -> 'For example, this is good.'
    """
    # remove spaces before punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    # fix brackets
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\[\s+", "[", s)
    s = re.sub(r"\s+\]", "]", s)

    # fix quotes a bit
    s = re.sub(r'"\s+', '"', s)
    s = re.sub(r"\s+\"", '"', s)
    s = re.sub(r"'\s+", "'", s)
    s = re.sub(r"\s+'", "'", s)

    return s


def sanitize(s: str) -> str:
    """
    Normalise text for ERRANT input:
    - Unicode normalisation
    - remove control chars
    - flatten newlines
    - collapse whitespace
    - detokenise punctuation for consistency
    """
    if s is None:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = _CTRL.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = detokenize_punctuation(s)
    return s


def main() -> None:
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

            if not prompt_id:
                print(f"[WARN] Skipping line {line_num}: missing prompt_id")
                skipped += 1
                continue

            if not src or not hyp or not ref:
                print(f"[WARN] Skipping line {line_num}: empty source/hyp/ref")
                skipped += 1
                continue

            by_prompt[prompt_id].append((src, hyp, ref))

    for prompt_id, triples in by_prompt.items():
        src_lines = [s for s, _, _ in triples]
        hyp_lines = [h for _, h, _ in triples]
        ref_lines = [t for _, _, t in triples]

        if not (len(src_lines) == len(hyp_lines) == len(ref_lines)):
            raise ValueError(
                f"Line count mismatch for {prompt_id}: "
                f"src={len(src_lines)}, hyp={len(hyp_lines)}, ref={len(ref_lines)}"
            )

        src_path = OUTPUT_DIR / f"{prompt_id}.src"
        hyp_path = OUTPUT_DIR / f"{prompt_id}.hyp"
        ref_path = OUTPUT_DIR / f"{prompt_id}.ref"

        src_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8", newline="\n")
        hyp_path.write_text("\n".join(hyp_lines) + "\n", encoding="utf-8", newline="\n")
        ref_path.write_text("\n".join(ref_lines) + "\n", encoding="utf-8", newline="\n")

        print(f"{prompt_id} => {len(triples)} examples")

        # print first 3 samples for sanity check
        print(f"\n[Sample check: {prompt_id}]")
        for i, (s, h, t) in enumerate(triples[:3], start=1):
            print(f"Example {i}")
            print("SRC:", s)
            print("HYP:", h)
            print("REF:", t)
            print("-" * 60)

    print(f"\nSaved ERRANT input files to: {OUTPUT_DIR}")
    print(f"Skipped rows: {skipped}")


if __name__ == "__main__":
    main()