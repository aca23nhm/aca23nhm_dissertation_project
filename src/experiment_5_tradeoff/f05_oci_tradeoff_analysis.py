"""
Experiment 5: F0.5 and OCI Trade-off Analysis

Section 5.5 reuses Experiment 2 ERRANT results and Experiment 2 / Experiment 4 OCI results.
This analysis is purely post-hoc and does not call any LLMs or regenerate outputs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Column name mappings (adjust if needed)
COLUMN_MAPPINGS = {
    'precision': 'precision',
    'recall': 'recall',
    'f05': 'f05'
}

# Input paths
ERRANT_REPORT_DIR = Path('outputs/experiment_2_compare_prompts/errant_outputs')
OCI_AGGREGATE_PATH = Path('outputs/experiment_2_compare_prompts/oci_eval/aggregate_oci.csv')

# Output paths
TRADEOFF_DIR = Path('outputs/experiment_5_tradeoff')
INTERPRETATION_DIR = Path('outputs/experiment_5_tradeoff')
TRADEOFF_TABLE_PATH = TRADEOFF_DIR / 'f05_oci_tradeoff_table.csv'
SCATTER_PLOT_PATH = TRADEOFF_DIR / 'f05_vs_oci_scatter.png'
INTERPRETATION_PATH = INTERPRETATION_DIR / 'f05_oci_tradeoff_interpretation.csv'


def extract_metrics_from_report(report_path: Path) -> dict:
    """Extract precision, recall, and F0.5 from an ERRANT report file."""
    content = report_path.read_text(encoding='utf-8')
    result = {}

    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            result['precision'] = float(parts[3])
            result['recall'] = float(parts[4])
            result['f05'] = float(parts[5])
            return result

    raise ValueError(f"Could not parse ERRANT metrics from {report_path}")


def load_errant_results():
    """Load Experiment 2 ERRANT precision, recall, and F0.5 results."""
    records = []

    for report_file in sorted(ERRANT_REPORT_DIR.glob('*.report.txt')):
        condition = report_file.name.replace('.report.txt', '')
        metrics = extract_metrics_from_report(report_file)
        records.append({
            'condition': condition,
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f05': metrics['f05']
        })

    return pd.DataFrame(records)


def load_oci_results():
    """Load aggregate OCI results from Experiment 2."""
    df = pd.read_csv(OCI_AGGREGATE_PATH)
    if 'condition' not in df.columns or 'mean_oci' not in df.columns:
        raise ValueError('Expected columns condition and mean_oci in aggregate OCI file')
    return df[['condition', 'mean_oci']].copy()


def categorize_tradeoff(row, median_f05, median_oci):
    """Assign an interpretation category based on median thresholds."""
    high_f05 = row['f05'] >= median_f05
    high_oci = row['mean_oci'] >= median_oci

    if high_f05 and high_oci:
        return 'aggressive'
    if high_f05 and not high_oci:
        return 'balanced/desirable'
    if not high_f05 and not high_oci:
        return 'conservative'
    return 'weak/inefficient'


def plot_tradeoff(df):
    """Save a scatter plot of OCI vs F0.5."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df['mean_oci'], df['f05'], s=120, c='tab:blue', edgecolors='black')

    for _, row in df.iterrows():
        plt.text(row['mean_oci'] + 0.0005, row['f05'] + 0.0005, row['condition'], fontsize=10)

    plt.title('Experiment 5: Correction Accuracy and OCI')
    plt.xlabel('Mean OCI (lower over-correction risk)')
    plt.ylabel('F0.5')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT_PATH, dpi=300)
    plt.close()


def main():
    TRADEOFF_DIR.mkdir(parents=True, exist_ok=True)
    INTERPRETATION_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading Experiment 2 ERRANT results...')
    errant_df = load_errant_results()

    print('Loading Experiment 2 OCI results...')
    oci_df = load_oci_results()

    print('Merging ERRANT and OCI results...')
    merged_df = errant_df.merge(oci_df, on='condition', how='inner')

    if merged_df.empty:
        raise RuntimeError('Merged trade-off table is empty. Check input files and condition names.')

    merged_df.to_csv(TRADEOFF_TABLE_PATH, index=False)
    print(f'Merged trade-off table saved to {TRADEOFF_TABLE_PATH}')

    median_f05 = merged_df['f05'].median()
    median_oci = merged_df['mean_oci'].median()

    merged_df['interpretation'] = merged_df.apply(
        lambda row: categorize_tradeoff(row, median_f05, median_oci), axis=1
    )
    merged_df.to_csv(INTERPRETATION_PATH, index=False)

    plot_tradeoff(merged_df)
    print(f'Scatter plot saved to {SCATTER_PLOT_PATH}')

    print('\nMerged trade-off table:')
    print(merged_df.to_string(index=False))
    print('\nMedian F0.5:', round(median_f05, 4))
    print('Median OCI:', round(median_oci, 6))

    print('\nInterpretation category per prompt:')
    for _, row in merged_df.iterrows():
        print(f"  {row['condition']}: {row['interpretation']}")

    balanced_df = merged_df[
        (merged_df['f05'] >= median_f05) & (merged_df['mean_oci'] <= median_oci)
    ]

    best_balanced = None
    if not balanced_df.empty:
        best_balanced = balanced_df.sort_values(['f05', 'mean_oci'], ascending=[False, True]).iloc[0]
        print(f"\nBest balanced prompt: {best_balanced['condition']}")
    else:
        print('\nBest balanced prompt: none found (no prompt is high F0.5 and low OCI)')


if __name__ == '__main__':
    main()
