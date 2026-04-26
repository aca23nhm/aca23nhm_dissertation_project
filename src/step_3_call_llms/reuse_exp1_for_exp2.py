import json
from pathlib import Path

# Paths
EXP1_JSONL = Path("outputs/experiment_1/experiment1_500_outputs.jsonl")
EXP2_JSONL = Path("outputs/experiment_2/experiment2_outputs.jsonl")

# Prompts to reuse from Experiment 1
REUSE_PROMPTS = {"instruction_v4", "role_v4", "fewshot_v4"}

def main():
    EXP2_JSONL.parent.mkdir(parents=True, exist_ok=True)

    reused_count = 0
    with EXP1_JSONL.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt_id = obj.get("prompt_id")
            if prompt_id in REUSE_PROMPTS:
                # Change prompt_id to match experiment2 naming
                if prompt_id == "instruction_v4":
                    obj["prompt_id"] = "instruction"
                elif prompt_id == "role_v4":
                    obj["prompt_id"] = "role"
                elif prompt_id == "fewshot_v4":
                    obj["prompt_id"] = "fewshot"
                # Append to exp2
                with EXP2_JSONL.open("a", encoding="utf-8") as f_out:
                    json.dump(obj, f_out)
                    f_out.write("\n")
                reused_count += 1

    print(f"Reused {reused_count} records from Experiment 1 for Experiment 2.")

if __name__ == "__main__":
    main()