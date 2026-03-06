# src/run_errant.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Repo root = parent of /src
ROOT = Path(__file__).resolve().parents[1]

IN_DIR = ROOT / "outputs" / "errant_inputs"
OUT_DIR = ROOT / "outputs" / "errant_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / "Scripts" / "python.exe"
if not VENV_PY.exists():
    raise FileNotFoundError(f"Cannot find venv python at: {VENV_PY}")

# Prefer ERRANT console scripts on Windows
ERRANT_PARALLEL = VENV_DIR / "Scripts" / "errant_parallel.exe"
ERRANT_COMPARE = VENV_DIR / "Scripts" / "errant_compare.exe"


def _env_utf8() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cmd(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command with UTF-8 enforced, rooted at repo ROOT.
    If capture=False, output streams directly to console (useful for debugging).
    """
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
        # inherit stdout/stderr so we can see real error messages live
        return subprocess.run(
            cmd,
            env=_env_utf8(),
            cwd=str(ROOT),
        )


def ensure_created_or_debug(path: Path, context: str, cmd: list[str], res: subprocess.CompletedProcess) -> None:
    if path.exists():
        return

    print(f"\n--- FAILED to create {path.name} ---")
    print(f"Context: {context}")
    print(f"Expected at: {path}")
    print(f"Repo ROOT:   {ROOT}")
    print(f"IN_DIR:      {IN_DIR}")
    print(f"OUT_DIR:     {OUT_DIR}")
    print("OUT_DIR contents (first 50):", [p.name for p in OUT_DIR.glob("*")][:50])

    # Show whatever we captured (often empty in your case)
    stdout = getattr(res, "stdout", "") or ""
    stderr = getattr(res, "stderr", "") or ""
    print("Captured STDOUT:\n", stdout[:2000])
    print("Captured STDERR:\n", stderr[:2000])

    # Crucial: rerun WITHOUT capture so any hidden error shows in console
    print("\nRe-running the same command without capture to reveal errors...\n")
    run_cmd(cmd, capture=False)

    raise FileNotFoundError(f"Expected file was not created: {path}")


def write_report(cmd: list[str], report_path: Path) -> None:
    res = run_cmd(cmd, capture=True)
    if res.returncode != 0:
        print("\n--- ERRANT compare FAILED ---")
        print("STDOUT:\n", (res.stdout or "")[:2000])
        print("STDERR:\n", (res.stderr or "")[:2000])

        print("\nRe-running compare without capture to reveal errors...\n")
        run_cmd(cmd, capture=False)

        raise RuntimeError(f"ERRANT compare failed. Command: {' '.join(cmd)}")

    report_path.write_text(res.stdout or "", encoding="utf-8", newline="\n")
    print(f"Saved report: {report_path}")


def build_parallel_cmd(orig: Path, cor: Path, out: Path) -> list[str]:
    orig_s = str(orig.resolve())
    cor_s = str(cor.resolve())
    out_s = str(out.resolve())

    # Prefer the venv console script if present
    if ERRANT_PARALLEL.exists():
        return [str(ERRANT_PARALLEL), "-orig", orig_s, "-cor", cor_s, "-out", out_s]

    # Fallback to python -m invocation
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


def main() -> None:
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {IN_DIR}. Run prepare_errant_inputs.py first.")

    src_files = sorted(IN_DIR.glob("*.src"))
    if not src_files:
        raise FileNotFoundError(f"No .src files found in {IN_DIR}. Run prepare_errant_inputs.py first.")

    for src_file in src_files:
        prompt_id = src_file.stem
        hyp_file = IN_DIR / f"{prompt_id}.hyp"
        ref_file = IN_DIR / f"{prompt_id}.ref"

        if not hyp_file.exists() or not ref_file.exists():
            raise FileNotFoundError(f"Missing hyp/ref for {prompt_id}:\n  {hyp_file}\n  {ref_file}")

        hyp_m2 = OUT_DIR / f"{prompt_id}.hyp.m2"
        ref_m2 = OUT_DIR / f"{prompt_id}.ref.m2"

        # 1) Generate hyp m2
        cmd = build_parallel_cmd(src_file, hyp_file, hyp_m2)
        res = run_cmd(cmd, capture=True)
        if res.returncode != 0:
            print("\n--- ERRANT parallel_to_m2 FAILED (hyp) ---")
            print("STDOUT:\n", (res.stdout or "")[:2000])
            print("STDERR:\n", (res.stderr or "")[:2000])
            print("\nRe-running without capture to reveal errors...\n")
            run_cmd(cmd, capture=False)
            raise RuntimeError(f"parallel_to_m2 failed for hyp: {prompt_id}")
        ensure_created_or_debug(hyp_m2, f"{prompt_id} hyp_m2", cmd, res)

        # 2) Generate ref m2
        cmd = build_parallel_cmd(src_file, ref_file, ref_m2)
        res = run_cmd(cmd, capture=True)
        if res.returncode != 0:
            print("\n--- ERRANT parallel_to_m2 FAILED (ref) ---")
            print("STDOUT:\n", (res.stdout or "")[:2000])
            print("STDERR:\n", (res.stderr or "")[:2000])
            print("\nRe-running without capture to reveal errors...\n")
            run_cmd(cmd, capture=False)
            raise RuntimeError(f"parallel_to_m2 failed for ref: {prompt_id}")
        ensure_created_or_debug(ref_m2, f"{prompt_id} ref_m2", cmd, res)

        # 3) Compare
        report = OUT_DIR / f"{prompt_id}.report.txt"
        compare_cmd = build_compare_cmd(hyp_m2, ref_m2)
        write_report(compare_cmd, report)

    print(f"All reports saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()