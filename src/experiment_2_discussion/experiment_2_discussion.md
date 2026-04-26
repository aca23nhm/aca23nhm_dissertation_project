# Experiment 2 Discussion

## 5.2 Experiment 2: Comparison of Prompting Strategies

### 5.2.1 Purpose
Compare four prompt strategies using their optimal configurations:
- `baseline`
- `instruction`
- `role`
- `fewshot`

The goal is to understand how prompt design affects grammatical correction accuracy and unnecessary edits.

### 5.2.2 Implementation
Evaluate each strategy with:
- ERRANT reports for Precision, Recall, F0.5
- OCI for over-correction / unnecessary edits

Data sources:
- `outputs/experiment_2/errant_outputs/*.report.txt`
- `outputs/experiment_2/oci_eval/aggregate_oci.csv`

### 5.2.3 Results and Discussion

#### Comparison table: F0.5 vs OCI
| Condition   | F0.5  | OCI (%) |
|-------------|:-----:|:-------:|
| baseline    | 42.89 | 4.35    |
| instruction | 49.52 | 2.82    |
| role        | 50.64 | 2.82    |
| fewshot     | 51.81 | 3.04    |

#### Behavioural differences
- **Aggressive**: `fewshot`
  - Highest F0.5, highest OCI among tuned prompts.
  - Best for maximum correction power, but more likely to produce unnecessary edits.
- **Controlled**: `instruction`
  - Lowest OCI while still achieving strong F0.5.
  - Best choice when keeping edits minimal is important.
- **Balanced**: `role`
  - Strong F0.5 with low OCI, offering a good middle ground.
  - Useful when both accuracy and edit conservatism matter.

#### Core insight: Trade-off between accuracy vs unnecessary edits
- The baseline prompt is the weakest performer: lower F0.5 and higher OCI than all optimized prompts.
- Prompt optimization improved correction accuracy while reducing unnecessary edits.
- Among the optimized strategies, higher accuracy is associated with slightly higher OCI.
- This confirms the expected trade-off: a more aggressive correction strategy can gain accuracy at the cost of more unnecessary edits, while a more controlled strategy keeps OCI lower.

### Practical recommendation
- Use `role` for a balanced correction strategy.
- Use `instruction` when minimizing over-correction is the priority.
- Use `fewshot` when maximizing F0.5 is the priority and slightly higher OCI is acceptable.
