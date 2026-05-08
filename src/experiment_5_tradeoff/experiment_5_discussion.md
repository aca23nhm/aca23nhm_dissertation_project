# Experiment 5: Trade-off Analysis between Correction Performance and Unnecessary Edits

## 5.5 Trade-off Analysis: Correction Performance vs Unnecessary Edits

### 5.5.1 Purpose

The goal of Experiment 5 is to analyse the relationship between grammatical correction performance and unnecessary editing behaviour. This study examines whether prompts that improve ERRANT F0.5 also increase OCI, or whether a balanced prompt can achieve high correction quality with lower unnecessary editing.

### 5.5.2 Implementation

- Existing Experiment 2 results are reused for this analysis.
- ERRANT reports from `outputs/experiment_2_compare_prompts/errant_outputs` provide precision, recall, and F0.5 for each prompt condition.
- Aggregate OCI values are loaded from `outputs/experiment_2/oci_eval/aggregate_oci.csv`.
- The two sources are merged by prompt condition.
- A scatter plot is created with:
  - X-axis: mean OCI
  - Y-axis: F0.5
  - One point per prompt condition
- Interpretation categories are assigned using the median F0.5 and median OCI across prompt conditions.

### 5.5.3 Results and Discussion

#### Trade-off Table

| Condition | Precision | Recall | F0.5 | Mean OCI | Category |
|-----------|-----------|--------|------|----------|----------|
| baseline | 40.98 | 52.73 | 42.89 | 0.01678 | weak/inefficient |
| fewshot | 54.47 | 43.34 | 51.81 | 0.01119 | aggressive |
| instruction | 52.82 | 39.63 | 49.52 | 0.01053 | conservative |
| role | 55.03 | 38.39 | 50.64 | 0.01050 | balanced/desirable |

#### Median Thresholds

- Median F0.5: **50.08**
- Median OCI: **0.01086**

#### Cluster Interpretation

- **Aggressive**: `fewshot` has the highest F0.5, but its OCI is also above the median, so the extra accuracy comes with more editing.
- **Balanced / Desirable**: `role` has high F0.5 while keeping OCI below the median, giving the best trade-off in this table.
- **Weak / Inefficient**: `baseline` has low F0.5 and above-median OCI, so it corrects less successfully while still making extra edits.
- **Conservative**: `instruction` keeps OCI below the median, with slightly lower F0.5 than `role`.

#### Discussion

- The scatter plot shows the trade-off between correction performance and unnecessary edits.
- `role` gives the clearest balance here: high F0.5 with lower OCI.
- `fewshot` is most aggressive, delivering the highest F0.5 but with more unnecessary edits.
- `baseline` performs worst on this comparison, with the lowest F0.5 and the highest OCI.

### Insight

The results show that ERRANT F0.5 and OCI need to be considered together. A prompt can improve correction quality while also increasing the risk of unnecessary edits, so prompt choice should account for both measures.
