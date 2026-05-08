from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


# Repo root
ROOT = Path(__file__).resolve().parents[2]

IN_DIR = ROOT / "outputs" / "experiment_1_prompt_engineering" / "experiment1_errant_inputs"
OUT_DIR = ROOT / "outputs" / "experiment_1_prompt_engineering" / "experiment1_errant_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENV_PY = Path(sys.executable)

VENV_DIR = ROOT / ".venv"
ERRANT_PARALLEL = VENV_DIR / "Scripts" / "errant_parallel.exe"
ERRANT_COMPARE = VENV_DIR / "Scripts" / "errant_compare.exe"


def _env_utf8() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cmd(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    print(" ".join(cmd))

    if capture:
        return subprocess.run(
            cmd,
            env=_env_utf8(),
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
    else:
        return subprocess.run(
            cmd,
            env=_env_utf8(),
            cwd=str(ROOT),
        )


def build_parallel_cmd(orig: Path, cor: Path, out: Path) -> list[str]:
    orig_s = str(orig.resolve())
    cor_s = str(cor.resolve())
    out_s = str(out.resolve())

    if ERRANT_PARALLEL.exists():
        return [str(ERRANT_PARALLEL), "-orig", orig_s, "-cor", cor_s, "-out", out_s]

    return [
        str(VENV_PY),
        "-m",
        "errant.commands.parallel_to_m2",
        "-orig",
        orig_s,
        "-cor",
        cor_s,
        "-out",
        out_s,
    ]


def build_compare_cmd(hyp_m2: Path, ref_m2: Path) -> list[str]:
    hyp_s = str(hyp_m2.resolve())
    ref_s = str(ref_m2.resolve())

    if ERRANT_COMPARE.exists():
        return [str(ERRANT_COMPARE), "-hyp", hyp_s, "-ref", ref_s]

    return [
        str(VENV_PY),
        "-m",
        "errant.commands.compare_m2",
        "-hyp",
        hyp_s,
        "-ref",
        ref_s,
    ]


def convert_scores_to_percentage(output: str) -> str:
    """
    Convert decimal Prec/Rec/F0.5 values (0–1)
    into percentage format (0–100).
    """
    lines = output.splitlines()
    new_lines = []

    for line in lines:
        if line.strip().startswith("TP"):
            # Header row
            new_lines.append(line)
            continue

        if line.strip() and any(x in line for x in ["Prec", "Rec", "F0.5"]):
            new_lines.append(line)
            continue

        parts = line.strip().split()

        # Look for metric rows with 6 columns (TP FP FN Prec Rec F0.5)
        if len(parts) == 6:
            try:
                tp, fp, fn = parts[0], parts[1], parts[2]
                p = float(parts[3]) * 100
                r = float(parts[4]) * 100
                f = float(parts[5]) * 100

                formatted_line = (
                    f"{tp}  {fp}  {fn}  "
                    f"{p:.2f}  {r:.2f}  {f:.2f}"
                )
                new_lines.append(formatted_line)
                continue
            except ValueError:
                pass

        new_lines.append(line)

    return "\n".join(new_lines)


def main() -> None:
    if not IN_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {IN_DIR}. Run prepare_experiment1_errant_inputs.py first."
        )

    src_files = sorted(IN_DIR.glob("*.src"))
    if not src_files:
        raise FileNotFoundError(
            f"No .src files found in {IN_DIR}. Run prepare_experiment1_errant_inputs.py first."
        )

    for src_file in src_files:
        prompt_id = src_file.stem
        hyp_file = IN_DIR / f"{prompt_id}.hyp"
        ref_file = IN_DIR / f"{prompt_id}.ref"

        if not hyp_file.exists() or not ref_file.exists():
            raise FileNotFoundError(
                f"Missing hyp/ref for {prompt_id}:\n  {hyp_file}\n  {ref_file}"
            )

        hyp_m2 = OUT_DIR / f"{prompt_id}.hyp.m2"
        ref_m2 = OUT_DIR / f"{prompt_id}.ref.m2"

        # Generate hyp m2
        cmd = build_parallel_cmd(src_file, hyp_file, hyp_m2)
        res = run_cmd(cmd)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(f"parallel_to_m2 failed for hyp: {prompt_id}")

        # Generate ref m2
        cmd = build_parallel_cmd(src_file, ref_file, ref_m2)
        res = run_cmd(cmd)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(f"parallel_to_m2 failed for ref: {prompt_id}")

        # Compare
        report = OUT_DIR / f"{prompt_id}.report.txt"
        compare_cmd = build_compare_cmd(hyp_m2, ref_m2)
        res = run_cmd(compare_cmd)

        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(f"compare_m2 failed for: {prompt_id}")

        formatted_output = convert_scores_to_percentage(res.stdout or "")
        report.write_text(formatted_output, encoding="utf-8")

        print(f"Saved report (percentage format): {report}")

    print(f"\nAll Experiment 1 reports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()