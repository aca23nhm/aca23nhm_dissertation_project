# src/model_runner.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml
from openai import OpenAI

from .logger import get_logger

log = get_logger(__name__)


# ----------------------------
# Output cleaning (deterministic)
# ----------------------------

_LABEL_PREFIXES = (
    "corrected sentence:",
    "corrected:",
    "correction:",
    "output:",
    "answer:",
)


def clean_model_output(text: Optional[str]) -> str:
    """
    Convert the model output to a single corrected sentence string.

    Rules:
      - trim whitespace
      - keep only the first non-empty line
      - remove wrapping quotes/backticks
      - remove leading labels like 'Corrected:'
      - collapse repeated whitespace
    """
    if not text:
        return ""

    t = text.strip()

    # Keep only the first non-empty line (models sometimes add extra lines)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = lines[0] if lines else ""

    # Remove wrapping quotes or backticks
    t = t.strip().strip("`").strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()

    # Remove leading label prefixes (case-insensitive)
    lowered = t.lower()
    for pref in _LABEL_PREFIXES:
        if lowered.startswith(pref):
            t = t[len(pref):].strip()
            break

    # Collapse multiple spaces/tabs into one
    t = re.sub(r"\s+", " ", t).strip()

    return t


# ----------------------------
# Config + Retry Policy
# ----------------------------

@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    seed: Optional[int] = None
    timeout_s: int = 60


@dataclass(frozen=True)
class RetryPolicy:
    """
    Simple retry policy for transient API failures.
    You can make this exponential later if you want.
    """
    max_retries: int = 2
    backoff_s: float = 1.0


def load_model_config(path: str) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return ModelConfig(**cfg)


# ----------------------------
# Model Runner
# ----------------------------

class ModelRunner:
    """
    Wrapper that:
      - fixes decoding params globally
      - calls the LLM
      - returns raw + cleaned output
      - logs metadata for reproducibility
    """

    def __init__(self, config: ModelConfig, retry: Optional[RetryPolicy] = None):
        self.config = config
        self.retry = retry or RetryPolicy()

        if self.config.provider != "openai":
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. In PowerShell you can set it for this session with:\n"
                '$env:OPENAI_API_KEY="sk-..."'
            )

        self.client = OpenAI(api_key=api_key)

    def run_one(self, sentence_id: str, prompt_id: str, rendered_prompt: str) -> Dict[str, Any]:
        start = time.time()

        # Fixed decoding parameters (never change per condition)
        model = self.config.model
        temperature = self.config.temperature
        max_tokens = self.config.max_tokens
        top_p = self.config.top_p
        seed = self.config.seed
        timeout_s = self.config.timeout_s

        last_error: Optional[str] = None

        for attempt in range(self.retry.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": rendered_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    seed=seed,
                    timeout=timeout_s,
                )

                raw = resp.choices[0].message.content or ""
                clean = clean_model_output(raw)

                elapsed_ms = int((time.time() - start) * 1000)

                record = {
                    "sentence_id": sentence_id,
                    "prompt_id": prompt_id,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "seed": seed,
                    "timeout_s": timeout_s,
                    "attempt": attempt,
                    "elapsed_ms": elapsed_ms,
                    "raw_output_text": raw,
                    "clean_output_text": clean,
                }

                log.info(
                    "Model run done",
                    extra={
                        "sentence_id": sentence_id,
                        "prompt_id": prompt_id,
                        "model": model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "seed": seed,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                    },
                )

                return record

            except Exception as e:
                last_error = str(e)
                log.exception(
                    "Model run failed",
                    extra={
                        "sentence_id": sentence_id,
                        "prompt_id": prompt_id,
                        "model": model,
                        "attempt": attempt,
                    },
                )

                # Retry if we still have attempts left
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.backoff_s)
                    continue

        # If all attempts failed, return a safe record
        elapsed_ms = int((time.time() - start) * 1000)
        return {
            "sentence_id": sentence_id,
            "prompt_id": prompt_id,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "seed": seed,
            "timeout_s": timeout_s,
            "attempt": self.retry.max_retries,
            "elapsed_ms": elapsed_ms,
            "raw_output_text": "",
            "clean_output_text": "",
            "error": last_error or "Unknown error",
        }