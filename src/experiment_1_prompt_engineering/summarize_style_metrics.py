import csv
from pathlib import Path

def main():
    csv_file = Path("outputs/experiment_1_prompt_engineering/style_eval/aggregate_style_metrics_simple.csv")

    print("STYLE EVALUATION METRICS PER PROMPT VERSION")
    print("=" * 60)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by prompt type
    prompt_types = {}
    for row in rows:
        condition = row['condition']
        prompt_type = condition.split('_v')[0]
        if prompt_type not in prompt_types:
            prompt_types[prompt_type] = []
        prompt_types[prompt_type].append(row)

    for prompt_type, variants in prompt_types.items():
        print(f"\n{prompt_type.upper()} PROMPTS:")
        print("-" * 40)

        for variant in sorted(variants, key=lambda x: x['condition']):
            condition = variant['condition']
            print(f"\n{condition}:")

            # Show the key metrics from the result dict
            metrics = {
                "source_word_count": float(variant['mean_source_word_count']),
                "output_word_count": float(variant['mean_output_word_count']),
                "word_levenshtein": float(variant['mean_word_levenshtein']),
                "edit_density": float(variant['mean_edit_density']),
                "source_ttr": float(variant['mean_source_ttr']),
                "output_ttr": float(variant['mean_output_ttr']),
                "delta_ttr": float(variant['mean_delta_ttr']),
                "source_fk": float(variant['mean_source_fk']),
                "output_fk": float(variant['mean_output_fk']),
                "delta_readability": float(variant['mean_delta_readability']),
                "stylometric_cosine": float(variant['mean_stylometric_cosine']),
            }

            print("result = {")
            for key, value in metrics.items():
                if key in ['word_levenshtein', 'edit_density', 'delta_ttr', 'delta_readability', 'stylometric_cosine']:
                    print(f'        "{key}": {value:.4f},')
                else:
                    print(f'        "{key}": {value:.2f},')
            print("    }")

if __name__ == "__main__":
    main()