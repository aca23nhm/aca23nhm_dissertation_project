from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

STEPS = [
    "src.step_4_errant_evaluation.prepare_experiment2_errant_inputs",
    "src.step_4_errant_evaluation.run_experiment2_errant",
    "src.step_5_style_evaluation.evaluate_style_experiment2",
    "src.step_6_oci.compute_oci_experiment2",
]


def run_step(module: str) -> None:
    print(f"\n=== Running {module} ===")
    cmd = [PYTHON, "-m", module]
    res = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Step failed: {module}")


def main() -> None:
    for module in STEPS:
        run_step(module)
    print("\nExperiment 2 evaluation complete.")


if __name__ == "__main__":
    main()