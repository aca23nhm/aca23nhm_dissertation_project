# Human Validation Instructions for OCI

This file explains how to annotate `results/human_validation_sample.csv`.

## Purpose

The aim is to manually check whether examples with higher OCI are genuinely more likely to show over-correction. OCI should be treated as a comparative signal, not as a final judgement.

## What to Read

For each row, compare:

1. `original_sentence`: the learner's original sentence.
2. `model_output`: the model correction.
3. `reference_correction`: the dataset reference correction, where available.
4. `OCI`, `edit_distance`, `similarity`, and `fluency_delta`: automatic signals for context only.

Do not label an example purely because its OCI is high or low. Use the text comparison as the main evidence.

## Allowed Labels

Use exactly one of the following labels in the `human_label` column:

- `minimal_correct_correction`
- `acceptable_useful_rewrite`
- `over_correction`
- `meaning_change`
- `under_correction_or_error`

## Label Meanings

- `minimal_correct_correction`: the model fixes the grammatical problem and stays close to the learner's wording.
- `acceptable_useful_rewrite`: the model rewrites more than minimally, but the rewrite is useful, meaning-preserving, and still appropriate.
- `over_correction`: the model makes unnecessary wording, style, or structure changes beyond grammatical correction.
- `meaning_change`: the model changes, removes, or adds meaning compared with the original sentence.
- `under_correction_or_error`: the model leaves important errors uncorrected or introduces a new error.

## Annotation Notes

Use `human_notes` to briefly explain difficult cases. For example:

- "Corrects grammar but changes vocabulary unnecessarily."
- "Meaning preserved; rewrite improves clarity."
- "Reference also rewrites heavily, so judgement is uncertain."
- "Output leaves the main verb error uncorrected."

## Consistency Rules

- Prefer `minimal_correct_correction` when the output makes only necessary grammatical changes.
- Prefer `acceptable_useful_rewrite` when extra changes improve clarity without hiding the learner's original meaning.
- Prefer `over_correction` when extra changes are stylistic, paraphrastic, or unnecessary for GEC.
- Prefer `meaning_change` when the correction changes the content, not just the style.
- Prefer `under_correction_or_error` when the correction is incomplete or introduces a new grammar/meaning problem.

## Sampling

The sample contains 60 examples selected with fixed random seed `1234`:

- 20 low-OCI examples
- 20 medium-OCI examples
- 20 high-OCI examples

The sampler attempts to include all prompt conditions within each OCI band.
