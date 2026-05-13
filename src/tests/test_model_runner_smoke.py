import pytest

from src.step_3_call_llms.model_runner import ModelConfig, ModelRunner


def test_model_runner_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = ModelConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.0,
        max_tokens=128,
    )

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        ModelRunner(cfg)
