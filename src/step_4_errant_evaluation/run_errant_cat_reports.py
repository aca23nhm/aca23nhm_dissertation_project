from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


# Correct project root (go up 2 levels)
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "errant_outputs"

# Use active venv python
VENV_PY = Path(sys.executable)


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    print(" ".join(cmd))

    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )


def convert_scores_to_percentage(output: str) -> str:
    """
    Convert decimal precision/recall/F0.5 to percentage format
    and align columns for readability.
    """
    lines = output.splitlines()
    new_lines = []

    for line in lines:
        parts = line.strip().split()

        # Match rows like: Category TP FP FN P R F0.5
        if len(parts) >= 6:
            try:
                category = parts[0]
                tp = int(parts[1])
                fp = int(parts[2])
                fn = int(parts[3])
                p = float(parts[4]) * 100
                r = float(parts[5]) * 100
                f = float(parts[6]) * 100

                formatted_line = (
                    f"{category:<12}"
                    f"{tp:>8}"
                    f"{fp:>8}"
                    f"{fn:>8}"
                    f"{p:>9.2f}"
                    f"{r:>9.2f}"
                    f"{f:>9.2f}"
                )

                new_lines.append(formatted_line)
                continue
            except (ValueError, IndexError):
                pass

        new_lines.append(line)

    return "\n".join(new_lines)

def main(cat_level: int = 3) -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing: {OUT_DIR}. Run run_errant.py first."
        )

    ref_files = sorted(OUT_DIR.glob("*.ref.m2"))
    if not ref_files:
        raise FileNotFoundError(
            f"No *.ref.m2 found in {OUT_DIR}. Run run_errant.py first."
        )

    for ref_m2 in ref_files:
        condition = ref_m2.name.replace(".ref.m2", "")
        hyp_m2 = OUT_DIR / f"{condition}.hyp.m2"

        if not hyp_m2.exists():
            print(f"Skipping {condition}: missing {hyp_m2.name}")
            continue

        out_txt = OUT_DIR / f"{condition}.cat{cat_level}.txt"

        cmd = [
            str(VENV_PY),
            "-m",
            "errant.commands.compare_m2",
            "-hyp", str(hyp_m2.resolve()),
            "-ref", str(ref_m2.resolve()),
            "-cat", str(cat_level),
        ]

        res = run_cmd(cmd)

        if res.returncode != 0:
            print(f"\n--- FAILED for {condition} ---")
            print("STDOUT:\n", (res.stdout or "")[:2000])
            print("STDERR:\n", (res.stderr or "")[:2000])
            raise RuntimeError(f"errant_compare failed for {condition}")

        formatted_output = convert_scores_to_percentage(res.stdout or "")

        out_txt.write_text(
            formatted_output,
            encoding="utf-8",
            newline="\n",
        )

        print(f"Saved (percentage format): {out_txt}")

    print("Done.")


if __name__ == "__main__":
    main(cat_level=3)