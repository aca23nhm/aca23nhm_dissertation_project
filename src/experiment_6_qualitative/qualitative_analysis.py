"""
Experiment 6: Qualitative Analysis of Model Behaviour

This script selects qualitative examples from the saved experiment outputs.
It does not call any LLM APIs or regenerate model outputs.

Purpose: compare baseline and best-prompt outputs in examples where edit behaviour differs.
"""

import json
import math
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# Saved inputs from earlier experiments
EXPERIMENT_2_JSONL = Path('outputs/experiment_2_compare_prompts/experiment2_outputs.jsonl')
TRADEOFF_TABLE = Path('outputs/experiment_5_tradeoff/f05_oci_tradeoff_table.csv')

# Files written by this analysis
OUTPUT_DIR = Path('outputs/experiment_6_qualitative')
CSV_PATH = OUTPUT_DIR / 'qualitative_examples.csv'
MD_PATH = OUTPUT_DIR / 'qualitative_examples.md'

# Keep the qualitative section short enough to inspect manually.
MAX_EXAMPLES = 8


def word_tokens(text):
    """Split text into simple word tokens for edit distance."""
    return [token for token in text.strip().split() if token]


def word_edit_distance(a, b):
    """Compute word-level Levenshtein distance."""
    a_words = word_tokens(a)
    b_words = word_tokens(b)
    n, m = len(a_words), len(b_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a_words[i - 1] == b_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]


def similarity_ratio(a, b):
    """Compute similarity ratio between token sequences."""
    return SequenceMatcher(None, word_tokens(a), word_tokens(b)).ratio()


def load_experiment_outputs():
    """Load experiment outputs from JSONL into a DataFrame."""
    records = []
    with EXPERIMENT_2_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            records.append({
                'sentence_id': data.get('sentence_id'),
                'condition': data.get('prompt_id'),
                'source': data.get('source', ''),
                'reference': data.get('reference', ''),
                'clean_output_text': data.get('clean_output_text', '')
            })
    return pd.DataFrame(records)


def choose_best_prompt():
    """Choose the best prompt condition from the trade-off table."""
    if TRADEOFF_TABLE.exists():
        df = pd.read_csv(TRADEOFF_TABLE)
        # Prefer a high-F0.5 prompt whose OCI is not above the median.
        median_oci = df['mean_oci'].median()
        balanced = df[df['mean_oci'] <= median_oci]
        oci_map = df.set_index('condition')['mean_oci'].to_dict()
        if not balanced.empty:
            best = balanced.sort_values(['f05', 'mean_oci'], ascending=[False, True]).iloc[0]
            return best['condition'], oci_map
        best = df.sort_values(['f05', 'mean_oci'], ascending=[False, True]).iloc[0]
        return best['condition'], oci_map
    raise FileNotFoundError('Trade-off table not found for best-prompt selection.')


def select_examples(df, baseline_condition, best_condition, oci_map):
    """Select examples where the baseline edits more than the selected prompt."""
    grouped = df[df['condition'].isin([baseline_condition, best_condition])].groupby('sentence_id')
    rows = []

    for sentence_id, group in grouped:
        if set(group['condition']) != {baseline_condition, best_condition}:
            continue

        baseline_row = group[group['condition'] == baseline_condition].iloc[0]
        best_row = group[group['condition'] == best_condition].iloc[0]

        if baseline_row['clean_output_text'] == best_row['clean_output_text']:
            continue

        source = baseline_row['source']

        baseline_dist = word_edit_distance(source, baseline_row['clean_output_text'])
        best_dist = word_edit_distance(source, best_row['clean_output_text'])
        baseline_similarity = similarity_ratio(source, baseline_row['clean_output_text'])
        best_similarity = similarity_ratio(source, best_row['clean_output_text'])

        if baseline_dist <= best_dist:
            continue

        score = (
            (baseline_dist - best_dist) * 2
            + (baseline_similarity - best_similarity) * 10
            + int(baseline_row['clean_output_text'] != best_row['clean_output_text'])
        )

        rows.append({
            'sentence_id': sentence_id,
            'source': source,
            'reference': baseline_row['reference'],
            'baseline_output': baseline_row['clean_output_text'],
            'best_prompt_output': best_row['clean_output_text'],
            'baseline_edit_distance': baseline_dist,
            'best_prompt_edit_distance': best_dist,
            'baseline_oci': oci_map.get(baseline_condition, float('nan')),
            'best_prompt_oci': oci_map.get(best_condition, float('nan')),
            'baseline_similarity': baseline_similarity,
            'best_similarity': best_similarity,
            'score': score
        })

    selected = sorted(rows, key=lambda x: (x['score'], x['baseline_edit_distance'] - x['best_prompt_edit_distance']), reverse=True)
    return selected[:MAX_EXAMPLES]


def explain_example(example, baseline_condition, best_condition):
    """Write a short note explaining the difference between the two outputs."""
    changes = []
    if example['baseline_edit_distance'] > example['best_prompt_edit_distance']:
        changes.append('baseline edits the sentence more aggressively')
    if example['baseline_similarity'] < example['best_similarity']:
        changes.append('baseline deviates from the original wording more than the best prompt')
    if example['baseline_output'].split() != example['best_prompt_output'].split():
        changes.append('baseline and best prompt outputs differ in structure or wording')

    if not changes:
        return 'The two outputs differ, with the baseline making the less conservative edit.'

    return f"The baseline {', and the baseline '.join(changes)}. In this example, the selected prompt stays closer to the original sentence."


def write_markdown(examples, baseline_condition, best_condition):
    """Write a markdown file with the selected examples."""
    lines = [
        '# Experiment 6: Qualitative Analysis of Model Behaviour',
        '',
        'This qualitative analysis uses existing experiment outputs only. It compares baseline and best prompt outputs to show how unnecessary edits appear in practice.',
        '',
        f'- Baseline prompt: `{baseline_condition}`',
        f'- Best prompt: `{best_condition}`',
        '',
    ]

    for idx, example in enumerate(examples, 1):
        lines.extend([
            f'## Example {idx}: Qualitative difference',
            '',
            '**Original:**',
            example['source'],
            '',
            '**Reference:**',
            example['reference'] or '_No reference available_',
            '',
            '**Baseline output:**',
            example['baseline_output'],
            '',
            '**Best prompt output:**',
            example['best_prompt_output'],
            '',
            '**Explanation:**',
            explain_example(example, baseline_condition, best_condition),
            '',
            '**Metadata:**',
            f'- baseline_edit_distance: {example["baseline_edit_distance"]}',
            f'- best_prompt_edit_distance: {example["best_prompt_edit_distance"]}',
            f'- baseline_oci: {example["baseline_oci"]:.6f}',
            f'- best_prompt_oci: {example["best_prompt_oci"]:.6f}',
            '',
        ])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text('\n'.join(lines), encoding='utf-8')


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_experiment_outputs()
    baseline_condition = 'baseline'
    best_condition, oci_map = choose_best_prompt()

    print(f'Baseline prompt: {baseline_condition}')
    print(f'Best prompt: {best_condition}')

    examples = select_examples(df, baseline_condition, best_condition, oci_map)
    if not examples:
        raise RuntimeError('No qualitative examples found that satisfy the selection criteria.')

    output_df = pd.DataFrame([{
        'sentence_id': ex['sentence_id'],
        'source': ex['source'],
        'reference': ex['reference'],
        'baseline_output': ex['baseline_output'],
        'best_prompt_output': ex['best_prompt_output'],
        'baseline_edit_distance': ex['baseline_edit_distance'],
        'best_prompt_edit_distance': ex['best_prompt_edit_distance'],
        'baseline_oci': ex['baseline_oci'],
        'best_prompt_oci': ex['best_prompt_oci'],
        'explanation': explain_example(ex, baseline_condition, best_condition)
    } for ex in examples])

    output_df.to_csv(CSV_PATH, index=False)
    write_markdown(examples, baseline_condition, best_condition)

    print(f'Saved qualitative CSV examples to {CSV_PATH}')
    print(f'Saved qualitative markdown examples to {MD_PATH}')
    print('\nSelected examples:')
    print(output_df[['sentence_id', 'baseline_edit_distance', 'best_prompt_edit_distance', 'baseline_oci', 'best_prompt_oci']].to_string(index=False))

if __name__ == '__main__':
    main()
