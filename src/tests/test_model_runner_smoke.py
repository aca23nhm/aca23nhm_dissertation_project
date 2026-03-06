from src.step_3_call_llms.model_runner import load_model_config, ModelRunner

def test_model_runner_smoke():
    cfg = load_model_config("configs/model.yaml")
    runner = ModelRunner(cfg)

    rendered_prompt = (
        "Correct the grammatical errors in the following sentence:\n"
        "She go to university every day.\n"
        "Return only the corrected sentence. No explanation."
    )

    rec = runner.run_one(
        sentence_id="demo-1",
        prompt_id="baseline",
        rendered_prompt=rendered_prompt,
    )

    assert "raw_output_text" in rec
    assert "clean_output_text" in rec
    assert rec["clean_output_text"] != ""