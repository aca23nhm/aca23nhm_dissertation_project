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
| minimal_correct_correction | 17 | 0.003213 | 0.003388 | 0.002312 | 0.004405 |
| acceptable_useful_rewrite | 16 | 0.005736 | 0.005496 | 0.003476 | 0.008270 |
| under_correction_or_error | 7 | 0.009404 | 0.005425 | 0.004897 | 0.033010 |
| over_correction | 15 | 0.026889 | 0.026026 | 0.006419 | 0.050318 |
| meaning_change | 5 | 0.018200 | 0.013732 | 0.012618 | 0.027157 |

## OCI and Ordinal Risk Score

Ordinal risk scores: minimal correct = 0; acceptable useful rewrite = 1; under-correction/error = 1; over-correction = 2; meaning change = 2.
- Spearman rho: 0.899
- p-value: < .001

## High-Risk Labels

`over_correction` and `meaning_change` are treated as high-risk labels.
- High-risk examples: 20

## Threshold Analysis

| Threshold | OCI cut-off | TP | FP | TN | FN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| median_oci | 0.005515 | 20 | 10 | 30 | 0 | 0.667 | 1.000 | 0.800 | 0.833 |
| upper_tertile_oci | 0.008873 | 19 | 1 | 39 | 1 | 0.950 | 0.950 | 0.950 | 0.967 |
| upper_quartile_oci | 0.013536 | 14 | 1 | 39 | 6 | 0.933 | 0.700 | 0.800 | 0.883 |
| sample_high_oci_band_min | 0.010080 | 19 | 1 | 39 | 1 | 0.950 | 0.950 | 0.950 | 0.967 |

## Interpretation Note

This analysis checks whether the manually assigned labels align with OCI values in the validation sample. Because the sample has only 60 examples and only a small number of high-risk labels, the threshold results should be interpreted as exploratory rather than definitive.
