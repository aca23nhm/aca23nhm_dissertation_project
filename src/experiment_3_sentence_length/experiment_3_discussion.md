# Experiment 3: Sentence Length Effects

## Purpose

This experiment examines whether sentence length affects grammatical error correction performance. Sentences are grouped as short (≤10 words), medium (11-20 words), and long (>20 words) so that prompt performance can be compared across different levels of sentence complexity.

## Implementation

### Data Preparation
- Sentences from the BEA-2019 dataset were categorized by length:
  - Short: ≤10 words (150 sentences)
  - Medium: 11-20 words (150 sentences) 
  - Long: >20 words (150 sentences)
- Each category was evaluated separately to prevent length-based aggregation artifacts

### Evaluation Metrics
- **Grammatical Accuracy**: ERRANT F0.5 score (harmonic mean favoring precision)
- **Over-Correction Index (OCI)**: Normalized composite metric measuring excessive edits across style dimensions (lexical sophistication, syntactic complexity, formality)

### Prompt Conditions
- **Baseline**: Standard correction prompt
- **Few-shot**: Examples of error correction pairs
- **Instruction**: Detailed correction guidelines
- **Role-based**: LLM assumes expert corrector role

## Results and Discussion

### Overall Performance Trends

| Prompt | Length | F0.5 | OCI (%) |
|--------|--------|------|---------|
| baseline | short | 42.11 | 2.76 |
| baseline | medium | 46.18 | 2.90 |
| baseline | long | 37.11 | 5.61 |
| fewshot | short | 49.81 | 1.58 |
| fewshot | medium | 58.24 | 2.34 |
| fewshot | long | 44.85 | 4.52 |
| instruction | short | 48.55 | 1.77 |
| instruction | medium | 52.50 | 2.23 |
| instruction | long | 43.13 | 4.23 |
| role | short | 47.57 | 1.95 |
| role | medium | 57.19 | 2.20 |
| role | long | 43.37 | 4.10 |

### Key Findings

#### 1. Sentence Length Impact on Correction Quality
- **Medium-length sentences show highest F0.5 scores** across all prompts:
  - Few-shot: 58.24 (medium) > 49.81 (short) > 44.85 (long)
  - Role: 57.19 (medium) > 47.57 (short) > 43.37 (long)
  - Instruction: 52.50 (medium) > 48.55 (short) > 43.13 (long)
  - Baseline: 46.18 (medium) > 42.11 (short) > 37.11 (long)

- **Long sentences consistently perform worst**: F0.5 scores drop for sentences >20 words, which indicates that longer inputs are harder to correct accurately.

- **Short sentences show moderate performance**: While not the best, short sentences maintain reasonable correction quality across all prompts.

#### 2. Over-Correction Patterns by Length
- **Long sentences exhibit highest OCI**: 4.10-5.61% across prompts, indicating more aggressive editing behavior on complex sentences.
- **Short sentences show lowest OCI**: 1.58-2.76%, suggesting more conservative corrections.
- **Medium sentences balance both metrics**: Moderate OCI (2.20-2.90%) with highest F0.5 scores.

#### 3. Prompt Effectiveness Across Length Categories

**Few-shot Prompting:**
- Best performer on medium sentences (F0.5: 58.24, OCI: 2.34%)
- Strong on short sentences (F0.5: 49.81, OCI: 1.58%)
- Moderate decline on long sentences (F0.5: 44.85, OCI: 4.52%)
- Most aggressive overall, with highest F0.5 but variable OCI

**Role-based Prompting:**
- Excellent on medium sentences (F0.5: 57.19, OCI: 2.20%)
- Good balance on short sentences (F0.5: 47.57, OCI: 1.95%)
- Consistent performance degradation on long sentences (F0.5: 43.37, OCI: 4.10%)
- Most stable OCI across categories

**Instruction Prompting:**
- Solid medium performance (F0.5: 52.50, OCI: 2.23%)
- Strong short sentence results (F0.5: 48.55, OCI: 1.77%)
- Significant long sentence challenges (F0.5: 43.13, OCI: 4.23%)
- Balanced approach with controlled over-correction

**Baseline Prompting:**
- Moderate medium performance (F0.5: 46.18, OCI: 2.90%)
- Lowest short sentence F0.5 (42.11) but reasonable OCI (2.76%)
- Worst long sentence performance (F0.5: 37.11, OCI: 5.61%)
- Most inconsistent, with highest OCI variance

### Insights and Implications

#### Complexity-Performance Relationship
- **Medium sentences perform best** for grammatical error correction, with the highest accuracy and moderate over-correction risk.
- **Long sentence correction remains challenging** for all prompting strategies, suggesting the need for specialized approaches for complex sentences.
- **Short sentences benefit from conservative editing**, as aggressive corrections may introduce unnecessary changes.

#### Prompt Strategy Recommendations
- **For medium-length sentences**: Use few-shot or role-based prompting for maximum accuracy.
- **For short sentences**: The optimized prompts perform similarly well; instruction prompting gives a good balance.
- **For long sentences**: Consider hybrid approaches or additional context provision, as all current strategies show reduced effectiveness.

#### Future Research Directions
- Investigate sentence-level features (syntactic complexity, error density) beyond simple word count.
- Develop length-adaptive prompting strategies that adjust correction aggressiveness based on sentence characteristics.
- Explore multi-stage correction pipelines for long sentences, breaking them into manageable segments.

These results show that sentence length affects prompt performance. Medium-length sentences are corrected most successfully in this setup, while long sentences remain more difficult and show higher OCI.
