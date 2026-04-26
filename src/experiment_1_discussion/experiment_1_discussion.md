# Experiment 1 Discussion

## 5.1 Experiment 1: Prompt Engineering Refinement

### 5.1.1 Purpose
The goal of Experiment 1 was to refine prompt design and identify optimal prompt variants for grammatical correction. The comparison included different prompt families:
- Baseline prompt
- Instruction prompts (v1–v4)
- Role-based prompts (v1–v4)
- Few-shot prompts (v1–v4)

### 5.1.2 Implementation
Evaluation used two metrics:
- ERRANT for grammatical correction quality, especially F0.5
- OCI for over-correction / unnecessary edits

### 5.1.3 Results and Discussion

#### Comparison table: F0.5 vs OCI
| Condition       | F0.5  | OCI (%) |
|-----------------|:-----:|:-------:|
| baseline        | 44.34 | 17.28   |
| instruction_v2  | 49.83 | 16.11   |
| instruction_v4  | 49.52 | 16.25   |
| role_v4         | 50.64 | 16.09   |
| fewshot_v4      | 51.81 | 16.28   |

> Best optimization per strategy:
> - **Best Instruction**: `instruction_v2` (F0.5 49.83, OCI 16.11)
> - **Best Role-based**: `role_v4` (F0.5 50.64, OCI 16.09)
> - **Best Few-shot**: `fewshot_v4` (F0.5 51.81, OCI 16.28)

#### Behavioural differences
- **Aggressive**: `fewshot_v4`
  - Highest F0.5, but still slightly higher OCI than other tuned variants.
  - Indicates the model is making the most corrections, including some unnecessary edits.
- **Controlled**: `instruction_v2`
  - Strong F0.5 with the lowest OCI among the top instruction variants.
  - Best when minimizing over-correction is important.
- **Balanced**: `role_v4`
  - High F0.5 and low OCI, providing the best balance between accuracy and edit conservatism.

#### Core insight: Trade-off between accuracy and unnecessary edits
- The baseline prompt had the lowest F0.5 and the highest OCI.
- All optimized prompt variants improved F0.5 while reducing OCI compared to baseline.
- The best few-shot prompt achieved the highest correction accuracy, but at a modest cost in OCI relative to the best role/instruction prompts.
- Therefore, prompt refinement successfully moved the system towards higher accuracy with less over-correction.

### Practical recommendations
- **Use `fewshot_v4`** when maximizing correction quality is the priority.
- **Use `role_v4`** when a balanced strategy is needed, especially if both accuracy and edit restraint matter.
- **Use `instruction_v2`** when controlled correction with minimal unnecessary edits is desirable.
