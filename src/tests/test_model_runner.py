from src.model_runner import ModelRunner, ModelConfig, RetryPolicy

def main():
    """runner = ModelRunner(backend="openai")"""
    runner = ModelRunner(backend="dummy")

    config = ModelConfig(
        model_name="gpt-4.1-mini",
        temperature=0.0,
        max_tokens=128,
        top_p=1.0,
        seed=42,
        retry=RetryPolicy(max_retries=3, base_delay_s=1.0, backoff_factor=2.0),
    )

    out = runner.generate("Correct this: I has a pen.", config)
    print(out)

if __name__ == "__main__":
    main()