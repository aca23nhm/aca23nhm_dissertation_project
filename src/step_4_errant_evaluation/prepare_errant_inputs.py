# src/prepare_errant_inputs.py
from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

IN_PATH = Path("outputs/model_outputs.jsonl")
OUT_DIR = Path("outputs/errant_inputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Remove control characters (includes the nasty Windows \x9d-style chars)
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize(s: str) -> str:
    """
    Normalise text and remove control characters that break ERRANT on Windows.
    This does NOT change grammatical content; it only stabilises encoding.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL.sub("", s)
    return s


def main() -> None:
    # per prompt_id: lists of (source, hyp, ref)
    by_prompt = defaultdict(list)

    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            src = sanitize(r.get("source", ""))
            hyp = sanitize(r.get("clean_output_text", ""))
            ref = sanitize(r.get("reference", ""))

            by_prompt[r["prompt_id"]].append((src, hyp, ref))

    for prompt_id, triples in by_prompt.items():
        src_path = OUT_DIR / f"{prompt_id}.src"
        hyp_path = OUT_DIR / f"{prompt_id}.hyp"
        ref_path = OUT_DIR / f"{prompt_id}.ref"

        src_path.write_text("\n".join(s for s, _, _ in triples), encoding="utf-8", newline="\n")
        hyp_path.write_text("\n".join(h for _, h, _ in triples), encoding="utf-8", newline="\n")
        ref_path.write_text("\n".join(t for _, _, t in triples), encoding="utf-8", newline="\n")

        print(prompt_id, "=>", len(triples))

    print(f"Saved ERRANT input files to: {OUT_DIR}")


if __name__ == "__main__":
    main()