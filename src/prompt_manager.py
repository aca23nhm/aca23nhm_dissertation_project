from __future__ import annotations

from pathlib import Path
from typing import Dict

PLACEHOLDER = "<SENTENCE>"


def load_prompts(prompts_dir: str | Path) -> Dict[str, str]:
    """
    Load all .txt files in prompts_dir.

    Returns:
        A dictionary mapping prompt_id -> template_text
        where prompt_id is the filename stem, e.g.:
          prompts/baseline.txt  -> "baseline"
          prompts/role.txt      -> "role"
    """
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    templates: Dict[str, str] = {}

    for file_path in sorted(prompts_dir.glob("*.txt")):
        prompt_id = file_path.stem
        template = file_path.read_text(encoding="utf-8").strip()

        if not template:
            raise ValueError(f"Prompt template is empty: {file_path}")

        templates[prompt_id] = template

    if not templates:
        raise ValueError(f"No .txt prompt templates found in: {prompts_dir}")

    return templates


def render_prompt(template: str, sentence: str) -> str:
    """
    Replace <SENTENCE> placeholder in the template with the actual sentence.

    Args:
        template: prompt template string containing <SENTENCE>
        sentence: input sentence to insert

    Returns:
        Rendered prompt string
    """
    sentence = sentence.strip()
    if not sentence:
        raise ValueError("Cannot render prompt: sentence is empty.")

    if PLACEHOLDER not in template:
        raise ValueError(f"Template does not contain required placeholder: {PLACEHOLDER}")

    return template.replace(PLACEHOLDER, sentence)