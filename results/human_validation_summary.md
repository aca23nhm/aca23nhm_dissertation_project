# Human Validation Analysis Summary

Input file: `C:/Users/Swift 3/aca23nhm_dissertation_project/results/human_validation_sample.csv`
Output CSV: `C:/Users/Swift 3/aca23nhm_dissertation_project/results/human_validation_analysis.csv`
Figure: `C:/Users/Swift 3/aca23nhm_dissertation_project/results/human_validation_oci_by_label.png`

## Label Completeness

- Total examples: 60
- Valid labelled examples: 60
- Missing labels: 0
- Invalid labels: 0

## OCI by Human Label

| Human label | n | Mean OCI | Median OCI | Min OCI | Max OCI |
|---|---:|---:|---:|---:|---:|
| minimal_correct_correction | 22 | 0.006062 | 0.004301 | 0.002312 | 0.027157 |
| acceptable_useful_rewrite | 9 | 0.015376 | 0.007422 | 0.003388 | 0.035538 |
| under_correction_or_error | 25 | 0.016508 | 0.008270 | 0.002313 | 0.050318 |
| over_correction | 3 | 0.005556 | 0.005174 | 0.004897 | 0.006597 |
| meaning_change | 1 | 0.005435 | 0.005435 | 0.005435 | 0.005435 |

## OCI and Ordinal Risk Score

Ordinal risk scores: minimal correct = 0; acceptable useful rewrite = 1; under-correction/error = 1; over-correction = 2; meaning change = 2.
- Spearman rho: 0.368
- p-value: 0.004

## High-Risk Labels

`over_correction` and `meaning_change` are treated as high-risk labels.
- High-risk examples: 4

## Threshold Analysis

| Threshold | OCI cut-off | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median_oci | 0.005515 | 1 | 29 | 27 | 3 | 0.033 | 0.250 | 0.059 | 0.467 |
| upper_tertile_oci | 0.008873 | 0 | 20 | 36 | 4 | 0.000 | 0.000 | 0.000 | 0.600 |
| upper_quartile_oci | 0.013536 | 0 | 15 | 41 | 4 | 0.000 | 0.000 | 0.000 | 0.683 |
| sample_high_oci_band_min | 0.010080 | 0 | 20 | 36 | 4 | 0.000 | 0.000 | 0.000 | 0.600 |

## Interpretation Note

This analysis checks whether the manually assigned labels align with OCI values in the validation sample. Because the sample has only 60 examples and only a small number of high-risk labels, the threshold results should be interpreted as exploratory rather than definitive.
