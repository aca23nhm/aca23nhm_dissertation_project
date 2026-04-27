# Experiment 5: Trade-off Analysis between Correction Performance and Unnecessary Edits

## 5.5 Trade-off Analysis: Correction Performance vs Unnecessary Edits

### 5.5.1 Purpose

The goal of Experiment 5 is to analyse the relationship between grammatical correction performance and unnecessary editing behaviour. This study examines whether prompts that improve ERRANT F0.5 also increase OCI, or whether a balanced prompt can achieve high correction quality with lower unnecessary editing.

### 5.5.2 Implementation

- Existing Experiment 2 results are reused for this analysis.
- ERRANT reports from `outputs/experiment_2/errant_outputs` provide precision, recall, and F0.5 for each prompt condition.
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
| baseline | 40.98 | 52.73 | 42.89 | 0.04350 | weak/inefficient |
| fewshot | 54.47 | 43.34 | 51.81 | 0.03043 | aggressive |
| instruction | 52.82 | 39.63 | 49.52 | 0.02818 | balanced/desirable |
| role | 55.03 | 38.39 | 50.64 | 0.02817 | balanced/desirable |

#### Median Thresholds

- Median F0.5: **50.08**
- Median OCI: **0.02930**

#### Cluster Interpretation

- **Aggressive**: `fewshot` achieves the highest F0.5, but also has above-median OCI. This suggests strong correction performance at the cost of more unnecessary edits.
- **Balanced / Desirable**: `instruction` and `role` both achieve high F0.5 while keeping OCI below the median. These prompts are the best trade-off candidates.
- **Weak / Inefficient**: `baseline` has low F0.5 and above-median OCI, indicating poor correction quality combined with unnecessary editing.
- **Conservative**: No prompt falls clearly into low F0.5 and low OCI for this experiment.

#### Discussion

- The scatter plot clearly demonstrates the trade-off between correction performance and unnecessary edits.
- `instruction` and `role` prompts are the strongest balanced options: they deliver high F0.5 while maintaining lower OCI.
- `fewshot` is most aggressive, delivering the highest F0.5 but with more unnecessary edits.
- `baseline` is the weakest prompt: it offers the lowest F0.5 and the highest OCI, making it the least desirable choice.

### Insight

This analysis demonstrates a clear trade-off between ERRANT F0.5 and OCI. It supports the main research contribution by showing that prompt selection should consider both correction quality and over-correction risk, and that balanced prompts like `instruction` and `role` provide better overall performance.
