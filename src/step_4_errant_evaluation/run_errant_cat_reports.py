from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "errant_outputs"

VENV_DIR = ROOT / ".venv"
ERRANT_COMPARE = VENV_DIR / "Scripts" / "errant_compare.exe"

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

def main(cat_level: int = 3) -> None:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"Missing: {OUT_DIR}. Run src/run_errant.py first.")

    if not ERRANT_COMPARE.exists():
        raise FileNotFoundError(f"Missing ERRANT compare exe: {ERRANT_COMPARE}")

    ref_files = sorted(OUT_DIR.glob("*.ref.m2"))
    if not ref_files:
        raise FileNotFoundError(f"No *.ref.m2 found in {OUT_DIR}. Run src/run_errant.py first.")

    for ref_m2 in ref_files:
        condition = ref_m2.name.replace(".ref.m2", "")
        hyp_m2 = OUT_DIR / f"{condition}.hyp.m2"
        if not hyp_m2.exists():
            print(f"Skipping {condition}: missing {hyp_m2.name}")
            continue

        out_txt = OUT_DIR / f"{condition}.cat{cat_level}.txt"

        cmd = [
            str(ERRANT_COMPARE),
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

        out_txt.write_text(res.stdout or "", encoding="utf-8", newline="\n")
        print(f"Saved: {out_txt}")

    print("Done.")

if __name__ == "__main__":
    main(cat_level=3)