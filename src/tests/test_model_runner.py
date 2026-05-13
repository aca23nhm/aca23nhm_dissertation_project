from src.step_3_call_llms.model_runner import clean_model_output, load_model_config


def test_clean_model_output_removes_labels_and_extra_lines():
    raw = "Corrected sentence: 'She goes to university every day.'\nExplanation: fixed verb agreement."

    assert clean_model_output(raw) == "She goes to university every day."


def test_load_model_config_reads_yaml():
    cfg = load_model_config("configs/model.yaml")

    assert cfg.provider == "openai"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 128
