import pytest

from src.step_2_prompt_manager.prompt_manager import load_prompts, render_prompt


def test_load_prompts_reads_text_files(tmp_path):
    prompt_file = tmp_path / "baseline.txt"
    prompt_file.write_text("Correct this: <SENTENCE>", encoding="utf-8")

    assert load_prompts(tmp_path) == {"baseline": "Correct this: <SENTENCE>"}


def test_render_prompt_replaces_placeholder_and_adds_output_rule():
    rendered = render_prompt("Correct this: <SENTENCE>", "She go home.")

    assert "She go home." in rendered
    assert "Return only the corrected sentence. No explanation." in rendered


def test_render_prompt_requires_placeholder():
    with pytest.raises(ValueError, match="required placeholder"):
        render_prompt("Correct this sentence.", "She go home.")
