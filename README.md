# Dissertation Project

This repository contains the code, prompts, data preparation scripts, evaluation outputs, and analysis files for the dissertation project.

## Project Structure

- `configs/`: model configuration used for generation.
- `data/`: raw and processed BEA/WI+LOCNESS data files.
- `prompts/`: prompt templates used in the experiments.
- `src/`: data loading, prompt rendering, model calls, ERRANT evaluation, OCI calculation, and analysis scripts.
- `outputs/`: generated model outputs and evaluation artefacts.
- `results/`: final summary tables, validation files, and figures.
- `tests/`: focused checks for utility functions and evaluation helpers.

## Reproducibility Notes

The default model configuration is in `configs/model.yaml`. Scripts that call the OpenAI API require `OPENAI_API_KEY` to be set in the environment. Generated outputs are included so that the analysis can be inspected without rerunning all model calls.
