from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 1.0     # initial delay
    backoff_factor: float = 2.0   # exponential backoff multiplier


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    seed: Optional[int] = None
    retry: RetryPolicy = RetryPolicy()


def clean_output(text: str) -> str:
    """
    Normalise model output for evaluation:
    - strip whitespace
    - remove surrounding quotes
    - keep first line (helps if model outputs extra lines)
    """
    if text is None:
        return ""

    out = text.strip()

    # Remove common surrounding quotes
    if (out.startswith('"') and out.endswith('"')) or (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()

    # Keep only first non-empty line (helps avoid "Corrected sentence: ..." blocks)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _retry_call(fn: Callable[[], str], retry: RetryPolicy) -> str:
    """
    Retry wrapper with exponential backoff.
    """
    last_err = None
    delay = retry.base_delay_s

    for attempt in range(retry.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= retry.max_retries:
                break
            time.sleep(delay)
            delay *= retry.backoff_factor

    raise RuntimeError(f"LLM call failed after {retry.max_retries} retries: {last_err}") from last_err


class ModelRunner:
    """
    A thin wrapper that exposes generate(prompt_text, config) -> output_text.
    Swap the backend implementation depending on how you run the model (API vs local).
    """

    def __init__(self, backend: str = "dummy"):
        """
        backend options:
          - "dummy": returns a deterministic placeholder (pipeline testing)
          - "openai": example backend stub (you fill in with actual API call)
        """
        self.backend = backend

    def generate(self, prompt_text: str, config: ModelConfig) -> str:
        if not prompt_text or not prompt_text.strip():
            raise ValueError("prompt_text is empty")

        if self.backend == "dummy":
            return clean_output(self._dummy_generate(prompt_text, config))

        if self.backend == "openai":
            return clean_output(self._openai_generate_with_retry(prompt_text, config))

        raise ValueError(f"Unknown backend: {self.backend}")

    # --------------------------
    # Dummy backend (for testing)
    # --------------------------
    def _dummy_generate(self, prompt_text: str, config: ModelConfig) -> str:
        # This is NOT a real correction, it just lets you test the pipeline.
        return "DUMMY_OUTPUT"

    # --------------------------
    # OpenAI backend (stub)
    # --------------------------
    def _openai_generate_with_retry(self, prompt_text: str, config: ModelConfig) -> str:
        def do_call() -> str:
            return self._openai_generate(prompt_text, config)

        return _retry_call(do_call, config.retry)

    def _openai_generate(self, prompt_text: str, config: ModelConfig) -> str:
        """
        Stub for OpenAI API call.

        You will need:
          - OPENAI_API_KEY in environment variables
          - openai python package installed
          - update this call depending on the API you use (Chat Completions / Responses)

        Keep this as a single isolated function so it’s easy to document + swap.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

        # Import inside to avoid crashing if user is using dummy backend first
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)

        # NOTE: This is an example pattern. You may need to adjust based on your chosen endpoint.
        response = client.responses.create(
            model=config.model_name,
            input=prompt_text,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p,
            # seed=config.seed,  # only include if your chosen model/endpoint supports it
        )

        # Extract text safely (Responses API returns output list)
        # This is defensive; you can simplify once you confirm your response structure.
        try:
            out_text = response.output[0].content[0].text
        except Exception:
            out_text = str(response)

        return out_text