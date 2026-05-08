from __future__ import annotations

import csv
import json
import math
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "outputs" / "experiment1_500_outputs.jsonl"
OUT_DIR = ROOT / "outputs" / "style_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_SENTENCE_CSV = OUT_DIR / "per_sentence_style_metrics.csv"
AGG_CSV = OUT_DIR / "aggregate_style_metrics.csv"

# Load the spaCy model used for POS-based style features.
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' is not installed. "
        "Install it with:\n"
        "python -m spacy download en_core_web_sm"
    )

WORD_RE = re.compile(r"\S+")
VOWEL_RE = re.compile(r"[aeiouy]+", re.I)


# ----------------------------
# Basic tokenisation utilities
# ----------------------------
def whitespace_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.strip())


def normalise_word_for_ttr(token: str) -> str:
    return token.strip(string.punctuation).lower()


def safe_div(num: float, den: float) -> float:
    return num / den if den != 0 else 0.0


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ----------------------------
# Word-level Levenshtein
# ----------------------------
def levenshtein_words(a_tokens: list[str], b_tokens: list[str]) -> int:
    """
    Standard Levenshtein distance at word level.
    Counts insertions, deletions, substitutions.
    """
    n = len(a_tokens)
    m = len(b_tokens)

    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev

    return prev[m]


# ----------------------------
# TTR
# ----------------------------
def ttr(text: str) -> float:
    tokens = [normalise_word_for_ttr(t) for t in whitespace_tokens(text)]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


# ----------------------------
# Readability: Flesch-Kincaid Grade
# ----------------------------
def count_syllables_in_word(word: str) -> int:
    """
    Simple heuristic syllable counter.
    Good enough for comparative sentence-level evaluation.
    """
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0

    groups = VOWEL_RE.findall(word)
    count = len(groups)

    # Silent 'e' heuristic
    if word.endswith("e") and not word.endswith(("le", "ye")) and count > 1:
        count -= 1

    return max(1, count)


def split_sentences_simple(text: str) -> list[str]:
    """
    Lightweight sentence split for FK.
    """
    parts = re.split(r"[.!?]+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def flesch_kincaid_grade(text: str) -> float:
    words = [normalise_word_for_ttr(t) for t in whitespace_tokens(text)]
    words = [w for w in words if w]
    num_words = len(words)
    if num_words == 0:
        return 0.0

    sentences = split_sentences_simple(text)
    num_sentences = max(1, len(sentences))

    syllables = sum(count_syllables_in_word(w) for w in words)

    # FKGL = 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    return 0.39 * (num_words / num_sentences) + 11.8 * (syllables / num_words) - 15.59


def fluency_score(text: str) -> float:
    stripped = text.strip()
    words = [normalise_word_for_ttr(t) for t in whitespace_tokens(stripped)]
    words = [w for w in words if w]
    if not words:
        return 0.0
    repeated_adjacent = sum(1 for a, b in zip(words, words[1:]) if a == b)
    repeated_penalty = safe_div(repeated_adjacent, max(1, len(words) - 1))
    punctuation_density = safe_div(sum(1 for ch in stripped if ch in string.punctuation), max(1, len(words)))
    punctuation_penalty = max(0.0, punctuation_density - 0.35)
    avg_word_len = safe_div(sum(len(w) for w in words), len(words))
    word_length_penalty = max(0.0, abs(avg_word_len - 5.0) / 20.0)
    sentence_len_penalty = min(0.25, max(0, len(words) - 40) / 120.0)
    terminal_penalty = 0.0 if stripped[-1:] in {".", "!", "?", '"', "'"} else 0.05
    return clip01(
        1.0
        - 0.35 * repeated_penalty
        - 0.20 * punctuation_penalty
        - 0.10 * word_length_penalty
        - 0.10 * sentence_len_penalty
        - terminal_penalty
    )


# ----------------------------
# Stylometric features
# ----------------------------
POS_GROUPS = {
    "noun_prop": {"NOUN", "PROPN"},
    "verb_prop": {"VERB", "AUX"},
    "adj_prop": {"ADJ"},
    "adv_prop": {"ADV"},
    "pron_prop": {"PRON"},
    "det_prop": {"DET"},
    "adp_prop": {"ADP"},
    "punct_prop": {"PUNCT"},
}


def punctuation_count(text: str) -> int:
    return sum(1 for ch in text if ch in string.punctuation)


def stylometric_features(text: str) -> dict[str, float]:
    doc = NLP(text)

    words = [t.text for t in doc if not t.is_space and not t.is_punct]
    all_tokens = [t for t in doc if not t.is_space]

    sent_len_words = len(words)
    avg_word_len = safe_div(sum(len(w) for w in words), sent_len_words)
    punct_count = punctuation_count(text)

    pos_total = len(all_tokens)
    pos_counts = Counter(t.pos_ for t in all_tokens)

    feats = {
        "sent_len_words": float(sent_len_words),
        "avg_word_len": float(avg_word_len),
        "punct_count": float(punct_count),
    }

    for feat_name, pos_set in POS_GROUPS.items():
        feats[feat_name] = safe_div(sum(pos_counts[p] for p in pos_set), pos_total)

    return feats


def cosine_similarity_dict(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    norm_b = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------
# Main per-row computation
# ----------------------------
def compute_metrics(source: str, output: str) -> dict[str, float]:
    src_tokens = whitespace_tokens(source)
    out_tokens = whitespace_tokens(output)

    edit_dist = levenshtein_words(src_tokens, out_tokens)
    edit_density = edit_dist / max(1, len(src_tokens))

    src_ttr = ttr(source)
    out_ttr = ttr(output)
    delta_ttr = abs(out_ttr - src_ttr)

    src_fk = flesch_kincaid_grade(source)
    out_fk = flesch_kincaid_grade(output)
    delta_r = abs(out_fk - src_fk)
    source_fluency = fluency_score(source)
    output_fluency = fluency_score(output)
    delta_fluency = output_fluency - source_fluency

    src_style = stylometric_features(source)
    out_style = stylometric_features(output)
    cos_sim = cosine_similarity_dict(src_style, out_style)

    result = {
        "source_word_count": len(src_tokens),
        "output_word_count": len(out_tokens),
        "word_levenshtein": edit_dist,
        "edit_density": edit_density,
        "source_ttr": src_ttr,
        "output_ttr": out_ttr,
        "delta_ttr": delta_ttr,
        "source_fk": src_fk,
        "output_fk": out_fk,
        "delta_readability": delta_r,
        "source_fluency": source_fluency,
        "output_fluency": output_fluency,
        "delta_fluency": delta_fluency,
        "stylometric_cosine": cos_sim,
    }

    for k, v in src_style.items():
        result[f"source_{k}"] = v
    for k, v in out_style.items():
        result[f"output_{k}"] = v

    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH}")

    per_sentence_rows = []
    grouped = defaultdict(list)

    with IN_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            row = json.loads(line)

            source = row.get("source", "").strip()
            output = row.get("clean_output_text", "").strip()
            condition = row.get("prompt_id", "unknown")
            sentence_id = row.get("sentence_id", idx)

            metrics = compute_metrics(source, output)

            out_row = {
                "sentence_id": sentence_id,
                "condition": condition,
                "source": source,
                "output": output,
                **metrics,
            }
            per_sentence_rows.append(out_row)
            grouped[condition].append(out_row)

    # Write per-sentence CSV
    if per_sentence_rows:
        fieldnames = list(per_sentence_rows[0].keys())
        with PER_SENTENCE_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_sentence_rows)

    # Aggregate per condition
    agg_rows = []
    metric_names = [
        "source_word_count",
        "output_word_count",
        "word_levenshtein",
        "edit_density",
        "source_ttr",
        "output_ttr",
        "delta_ttr",
        "source_fk",
        "output_fk",
        "delta_readability",
        "source_fluency",
        "output_fluency",
        "delta_fluency",
        "source_sent_len_words",
        "output_sent_len_words",
        "source_avg_word_len",
        "output_avg_word_len",
        "source_punct_count",
        "output_punct_count",
        "source_noun_prop",
        "output_noun_prop",
        "source_verb_prop",
        "output_verb_prop",
        "source_adj_prop",
        "output_adj_prop",
        "source_adv_prop",
        "output_adv_prop",
        "source_pron_prop",
        "output_pron_prop",
        "source_det_prop",
        "output_det_prop",
        "source_adp_prop",
        "output_adp_prop",
        "source_punct_prop",
        "output_punct_prop",
        "stylometric_cosine",
    ]

    for condition, rows in grouped.items():
        agg = {"condition": condition, "n_sentences": len(rows)}
        for m in metric_names:
            agg[f"mean_{m}"] = mean([float(r[m]) for r in rows])
        agg_rows.append(agg)

    if agg_rows:
        fieldnames = list(agg_rows[0].keys())
        with AGG_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_rows)

    print(f"Saved per-sentence metrics: {PER_SENTENCE_CSV}")
    print(f"Saved aggregate metrics:    {AGG_CSV}")


if __name__ == "__main__":
    main()
