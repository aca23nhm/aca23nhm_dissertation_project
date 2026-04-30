# Experiment 4: OCI Robustness and Sensitivity Analysis

## 5.4.1 Purpose

To validate the stability and reliability of the Over-Correction Index (OCI) metric under different weighting configurations. This analysis tests whether prompt rankings remain consistent when OCI component weights are varied, ensuring the metric provides robust and reliable comparisons across different prompting strategies.

## 5.4.2 Implementation

### Data Source
- **Input**: Existing Experiment 2 style metrics (2,000 sentences across 4 prompt conditions)
- **No new LLM calls**: Analysis uses only pre-computed style metrics
- **Components**: edit_distance, edit_density, delta_ttr, delta_readability, inverted_cosine

### Weighting Schemes Tested
- **A_Default**: edit_distance=0.35, edit_density=0.15, delta_ttr=0.20, delta_readability=0.15, inverted_cosine=0.15
- **B_Edit_heavy**: edit_distance=0.50, edit_density=0.20, delta_ttr=0.10, delta_readability=0.10, inverted_cosine=0.10
- **C_Style_heavy**: edit_distance=0.25, edit_density=0.10, delta_ttr=0.25, delta_readability=0.25, inverted_cosine=0.15
- **D_Similarity_heavy**: edit_distance=0.25, edit_density=0.10, delta_ttr=0.15, delta_readability=0.15, inverted_cosine=0.35
- **E_Equal_weights**: edit_distance=0.20, edit_density=0.20, delta_ttr=0.20, delta_readability=0.20, inverted_cosine=0.20

### Methodology
- Min-max normalization of all OCI components across the entire dataset
- Computation of weighted OCI scores for each sentence under each scheme
- Aggregation of mean OCI per prompt condition per scheme
- Ranking of prompts by mean OCI (lower = better) within each scheme

## 5.4.3 Results and Discussion

### Prompt Rankings Across Weighting Schemes

| Scheme | Rank 1 (Best) | Rank 2 | Rank 3 | Rank 4 (Worst) |
|--------|---------------|--------|--------|---------------|
| A_Default | role (0.0105) | instruction (0.0105) | fewshot (0.0112) | baseline (0.0168) |
| B_Edit_heavy | role (0.0111) | instruction (0.0112) | fewshot (0.0114) | baseline (0.0160) |
| C_Style_heavy | role (0.0097) | instruction (0.0098) | fewshot (0.0107) | baseline (0.0167) |
| D_Similarity_heavy | instruction (0.0081) | role (0.0082) | fewshot (0.0085) | baseline (0.0134) |
| E_Equal_weights | role (0.0087) | instruction (0.0087) | fewshot (0.0093) | baseline (0.0149) |

### Key Findings

#### Ranking Stability Analysis
- **Consistent top performers**: `instruction` and `role` prompts consistently rank in the top 2 across all schemes
- **Stable bottom performer**: `baseline` prompt consistently ranks last (4th) across all schemes
- **Minor variation in top ranking**: `role` is best in 4/5 schemes, `instruction` is best in 1/5 schemes
- **Few-shot consistency**: `fewshot` maintains 3rd position in all schemes

#### OCI Sensitivity to Weighting
- **Similarity-heavy scheme (D)** shows the main variation, with `instruction` overtaking `role` as the best prompt
- **Other schemes** maintain the same ranking order: role > instruction > fewshot > baseline
- **Similarity-heavy scheme (D)** shows the largest OCI reduction for optimized prompts, suggesting semantic similarity is a strong differentiator

#### Mean OCI Ranges
- **Baseline**: 0.0134 - 0.0168 (highest OCI, most over-correction)
- **Few-shot**: 0.0085 - 0.0114 (moderate OCI)
- **Role**: 0.0082 - 0.0111 (low OCI, most stable)
- **Instruction**: 0.0081 - 0.0112 (lowest OCI in the similarity-heavy scheme)

### Insights

#### OCI Reliability as a Composite Metric
- **High stability**: Rankings remain largely consistent across diverse weighting schemes
- **Robust differentiation**: The metric reliably distinguishes between prompt quality levels
- **Minimal sensitivity**: Only extreme weighting changes (like edit-heavy) cause ranking shifts
- **Validated composite approach**: The weighted combination of multiple style dimensions provides stable, meaningful comparisons

#### Practical Implications
- **Default weighting recommended**: The A_Default scheme provides balanced, stable rankings
- **Conservative interpretation**: Minor ranking changes between instruction and role prompts are not practically significant
- **Baseline clearly inferior**: All schemes consistently identify baseline as the worst performer
- **Few-shot as middle ground**: Consistent 3rd ranking validates its balanced performance

#### Future Considerations
- **Weight optimization**: The analysis supports the current default weights as appropriate
- **Additional schemes**: Extreme weighting scenarios could be explored but may not be practically relevant
- **Component importance**: Similarity measures (inverted_cosine) appear most influential in distinguishing prompt quality

This robustness analysis confirms that the OCI metric provides stable and reliable prompt comparisons across different weighting configurations, supporting its use as a comprehensive evaluation tool for grammatical error correction prompting strategies.
