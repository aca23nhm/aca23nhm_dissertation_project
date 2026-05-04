# Statistical Tests Summary

Input file: `C:/Users/Swift 3/aca23nhm_dissertation_project/outputs/experiment_2_compare_prompts/oci_eval/per_sentence_oci.csv`

## Paired OCI Comparisons

Each structured prompt is compared with the baseline on the same sentence IDs. Positive differences mean the baseline has higher OCI than the structured prompt.

| Comparison | n | Mean OCI difference | Median OCI difference | 95% bootstrap CI | Wilcoxon p | Cohen's dz |
|---|---:|---:|---:|---:|---:|---:|
| baseline_vs_instruction | 500 | 0.007038 | 0.000000 | [0.005010, 0.009452] | < .001 | 0.274 |
| baseline_vs_role | 500 | 0.006340 | 0.000031 | [0.004723, 0.008208] | < .001 | 0.314 |
| baseline_vs_fewshot | 500 | 0.005661 | 0.000000 | [0.003339, 0.008313] | < .001 | 0.198 |

## Sentence Length and OCI

| Condition | n | Spearman rho | p-value |
|---|---:|---:|---:|
| all_conditions | 2000 | 0.567 | < .001 |
| baseline | 500 | 0.574 | < .001 |
| fewshot | 500 | 0.591 | < .001 |
| instruction | 500 | 0.585 | < .001 |
| role | 500 | 0.534 | < .001 |

## Availability Notes

- Sentence-level OCI values are available and were used for paired tests.
- Sentence length is available through the OCI component file and was used for Spearman correlation.
- Sentence-level ERRANT $F_{0.5}$ or edit-level accuracy values were not found in the available result files, so paired/bootstrap accuracy tests were not computed.
