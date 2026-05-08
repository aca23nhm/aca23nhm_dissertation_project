"""
Experiment 4: OCI Robustness and Sensitivity Analysis

This script checks how OCI changes when the component weights are varied.
It uses existing Experiment 2 style metrics and does not call the model again.

Purpose: check whether prompt rankings change under alternative OCI weighting schemes.

Data Source: Experiment 2 style metrics (outputs/experiment_2_compare_prompts/style_eval/per_sentence_style_metrics.csv)
- Uses only the final selected prompts from Experiment 2: baseline, fewshot, instruction, role
- 500 sentences per condition

Weighting Schemes Tested:
A. Default: edit_distance=0.35, edit_density=0.15, delta_ttr=0.20, delta_readability=0.15, inverted_cosine=0.15
B. Edit-heavy: edit_distance=0.50, edit_density=0.20, delta_ttr=0.10, delta_readability=0.10, inverted_cosine=0.10
C. Style-heavy: edit_distance=0.25, edit_density=0.10, delta_ttr=0.25, delta_readability=0.25, inverted_cosine=0.15
D. Similarity-heavy: edit_distance=0.25, edit_density=0.10, delta_ttr=0.15, delta_readability=0.15, inverted_cosine=0.35
E. Equal weights: edit_distance=0.20, edit_density=0.20, delta_ttr=0.20, delta_readability=0.20, inverted_cosine=0.20

Outputs:
- oci_weight_sensitivity_results.csv: Per-sentence OCI under each weighting scheme
- oci_prompt_rankings.csv: Mean OCI and rankings per prompt per scheme
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Map OCI component names to the corresponding CSV columns.
COLUMN_MAPPINGS = {
    'edit_distance': 'word_levenshtein',
    'edit_density': 'edit_density',
    'delta_ttr': 'delta_ttr',
    'delta_readability': 'delta_readability',
    'cosine_similarity': 'stylometric_cosine'
}

UTILITY_WEIGHTS = {
    'delta_fluency': 0.50,
    'cosine_similarity': 0.50,
}


def fluency_score(text):
    stripped = str(text).strip()
    words = [token.strip(".,!?;:\"'()[]{}").lower() for token in stripped.split()]
    words = [word for word in words if word]
    if not words:
        return 0.0
    repeated = sum(1 for left, right in zip(words, words[1:]) if left == right)
    repeated_penalty = repeated / max(1, len(words) - 1)
    punctuation_density = sum(1 for ch in stripped if ch in ".,!?;:\"'()[]{}") / max(1, len(words))
    punctuation_penalty = max(0.0, punctuation_density - 0.35)
    avg_word_len = sum(len(word) for word in words) / len(words)
    word_length_penalty = max(0.0, abs(avg_word_len - 5.0) / 20.0)
    sentence_len_penalty = min(0.25, max(0, len(words) - 40) / 120.0)
    terminal_penalty = 0.0 if stripped[-1:] in {".", "!", "?", '"', "'"} else 0.05
    return max(
        0.0,
        min(
            1.0,
            1.0
            - 0.35 * repeated_penalty
            - 0.20 * punctuation_penalty
            - 0.10 * word_length_penalty
            - 0.10 * sentence_len_penalty
            - terminal_penalty,
        ),
    )

# Weighting schemes
WEIGHTING_SCHEMES = {
    'A_Default': {
        'edit_distance': 0.35,
        'edit_density': 0.15,
        'delta_ttr': 0.20,
        'delta_readability': 0.15,
        'inverted_cosine': 0.15
    },
    'B_Edit_heavy': {
        'edit_distance': 0.50,
        'edit_density': 0.20,
        'delta_ttr': 0.10,
        'delta_readability': 0.10,
        'inverted_cosine': 0.10
    },
    'C_Style_heavy': {
        'edit_distance': 0.25,
        'edit_density': 0.10,
        'delta_ttr': 0.25,
        'delta_readability': 0.25,
        'inverted_cosine': 0.15
    },
    'D_Similarity_heavy': {
        'edit_distance': 0.25,
        'edit_density': 0.10,
        'delta_ttr': 0.15,
        'delta_readability': 0.15,
        'inverted_cosine': 0.35
    },
    'E_Equal_weights': {
        'edit_distance': 0.20,
        'edit_density': 0.20,
        'delta_ttr': 0.20,
        'delta_readability': 0.20,
        'inverted_cosine': 0.20
    }
}

def load_experiment2_style_metrics():
    """Load the per-sentence style metrics from Experiment 2."""
    input_path = Path('outputs/experiment_2_compare_prompts/style_eval/per_sentence_style_metrics.csv')
    df = pd.read_csv(input_path)
    return df

def prepare_oci_components(df):
    """Extract and prepare OCI component metrics."""
    if 'source_fluency' not in df.columns:
        df['source_fluency'] = df['source'].map(fluency_score)
    if 'output_fluency' not in df.columns:
        df['output_fluency'] = df['output'].map(fluency_score)
    if 'delta_fluency' not in df.columns:
        df['delta_fluency'] = df['output_fluency'] - df['source_fluency']

    # Map column names
    components = {}
    for component, col in COLUMN_MAPPINGS.items():
        components[component] = df[col].copy()

    # Create inverted cosine similarity
    components['inverted_cosine'] = (1 - components['cosine_similarity']).clip(0.0, 1.0)
    components['delta_fluency'] = df['delta_fluency'].copy()

    return components

def min_max_normalize(values):
    """Min-max normalize a series of values."""
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - min_val) / (max_val - min_val)

def compute_weighted_oci(norm_components, weights):
    """Compute OCI using the supplied divergence weights."""
    divergence = np.zeros(len(norm_components['norm_edit_distance']))
    for component, weight in weights.items():
        norm_key = f'norm_{component}'
        if norm_key in norm_components:
            divergence += weight * norm_components[norm_key]
    divergence = np.clip(divergence, 0.0, 1.0)
    utility = (
        UTILITY_WEIGHTS['delta_fluency'] * norm_components['norm_delta_fluency']
        + UTILITY_WEIGHTS['cosine_similarity'] * norm_components['cosine_similarity'].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return divergence, np.clip(divergence * (1.0 - utility), 0.0, 1.0), utility

def main():
    # Load data
    print("Loading Experiment 2 style metrics...")
    df = load_experiment2_style_metrics()
    print(f"Loaded {len(df)} sentences from {df['condition'].nunique()} conditions")

    # Prepare OCI components
    components = prepare_oci_components(df)

    # Min-max normalize each component across the entire dataset
    print("Normalizing OCI components...")
    norm_components = {}
    for component, values in components.items():
        norm_components[f'norm_{component}'] = min_max_normalize(values)
    norm_components['norm_inverted_cosine'] = components['inverted_cosine']
    norm_components['cosine_similarity'] = components['cosine_similarity']

    # Compute OCI under each weighting scheme
    oci_results = {}
    for scheme_name, weights in WEIGHTING_SCHEMES.items():
        print(f"Computing OCI for scheme: {scheme_name}")
        oci_results[scheme_name] = compute_weighted_oci(norm_components, weights)

    # Create results dataframe
    results_df = df[['sentence_id', 'condition']].copy()
    for scheme_name, (divergence_values, utility_oci_values, utility_values) in oci_results.items():
        results_df[f'oci_divergence_{scheme_name}'] = divergence_values
        results_df[f'utility_{scheme_name}'] = utility_values
        results_df[f'oci_utility_{scheme_name}'] = utility_oci_values
        results_df[f'oci_{scheme_name}'] = utility_oci_values

    # Save per-sentence results
    output_dir = Path('outputs/experiment_4_oci_robustness')
    results_path = output_dir / 'oci_weight_sensitivity_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Saved per-sentence OCI results to {results_path}")

    # Compute mean OCI per condition per scheme
    rankings_data = []
    for scheme_name in WEIGHTING_SCHEMES.keys():
        oci_col = f'oci_{scheme_name}'
        mean_oci_per_condition = results_df.groupby('condition')[oci_col].mean().reset_index()
        mean_oci_per_condition.columns = ['condition', 'mean_oci']
        divergence_col = f'oci_divergence_{scheme_name}'
        mean_divergence = results_df.groupby('condition')[divergence_col].mean().reset_index()
        mean_divergence.columns = ['condition', 'mean_oci_divergence']
        mean_oci_per_condition = mean_oci_per_condition.merge(mean_divergence, on='condition')
        mean_oci_per_condition['scheme'] = scheme_name
        rankings_data.append(mean_oci_per_condition)

    rankings_df = pd.concat(rankings_data, ignore_index=True)

    # Add rankings (lower OCI is better)
    rankings_df['rank'] = rankings_df.groupby('scheme')['mean_oci'].rank(method='dense', ascending=True).astype(int)

    # Save rankings
    rankings_path = output_dir / 'oci_prompt_rankings.csv'
    rankings_df.to_csv(rankings_path, index=False)
    print(f"Saved prompt rankings to {rankings_path}")

    # Print summary
    print("\n" + "="*60)
    print("OCI ROBUSTNESS AND SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)

    for scheme_name in WEIGHTING_SCHEMES.keys():
        print(f"\nScheme: {scheme_name}")
        scheme_data = rankings_df[rankings_df['scheme'] == scheme_name].sort_values('rank')
        for _, row in scheme_data.iterrows():
            print(f"  {row['rank']}. {row['condition']}: OCI={row['mean_oci']:.4f}")

    # Check if best prompt changes
    best_prompts = {}
    for scheme_name in WEIGHTING_SCHEMES.keys():
        scheme_data = rankings_df[rankings_df['scheme'] == scheme_name]
        best_prompt = scheme_data.loc[scheme_data['rank'].idxmin(), 'condition']
        best_prompts[scheme_name] = best_prompt

    print(f"\nBest (lowest OCI) prompt per scheme:")
    for scheme, prompt in best_prompts.items():
        print(f"  {scheme}: {prompt}")

    unique_best = set(best_prompts.values())
    if len(unique_best) == 1:
        print("Ranking is stable - same best prompt across all weighting schemes")
    else:
        print("Ranking varies - best prompt changes with weighting scheme")
        print(f"  Unique best prompts: {', '.join(unique_best)}")

    print("\nAnalysis complete. The results show how much the OCI rankings depend on the weighting choices.")

if __name__ == "__main__":
    main()
