"""Run every self-check. Type `python tests/run_all.py`.

The script starts each check in its own process, prints the output, and then gives one
table. The exit code is 0 if no check failed. A known defect of the section 8 of
NOTES.md is not a failure. If a defect check passes, the table says FIXED, because
somebody corrected the defect and the check must move to check().

Type `python tests/run_all.py --fast` to leave out the end-to-end check, which trains
a small model and needs about half a minute.
"""

import re
import sys
import time
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The fast checks first, the end-to-end check last.
SCRIPTS = [
    ROOT / "fp_spectrogram.py",           # the self-check in its __main__ block
    HERE / "test_geometry.py",
    HERE / "test_spectrogram.py",
    HERE / "test_dsp.py",
    HERE / "test_snr_aug.py",
    HERE / "test_model.py",
    HERE / "test_dataset.py",
    HERE / "test_worker.py",
    HERE / "test_prepare_clip.py",
]
SLOW = [HERE / "test_end_to_end.py"]

_SUMMARY = re.compile(r"^\s*->\s*(.*)$", re.M)


def main(argv):
    scripts = list(SCRIPTS)
    if "--fast" not in argv:
        scripts += SLOW
    else:
        print("[--fast] the end-to-end check is left out\n")

    rows, failed = [], 0
    for path in scripts:
        t0 = time.time()
        p = subprocess.run([sys.executable, str(path)], cwd=str(ROOT),
                           capture_output=True, text=True)
        dt = time.time() - t0
        out = (p.stdout or "") + (p.stderr or "")
        print(out.rstrip())
        m = _SUMMARY.findall(out)
        detail = m[-1].split("::", 1)[-1].strip() if m else (
            "ok" if p.returncode == 0 else "crashed")
        state = "PASS" if p.returncode == 0 else "FAIL"
        if p.returncode != 0:
            failed += 1
        rows.append((state, path.name, f"{dt:5.1f}s", detail))

    width = max(len(r[1]) for r in rows)
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for state, name, dt, detail in rows:
        print(f"  {state}  {name:<{width}}  {dt}  {detail}")
    total = sum(float(r[2].strip('s')) for r in rows)
    print("-" * 78)
    if failed:
        print(f"  {failed} of {len(rows)} scripts FAILED   ({total:.0f}s total)")
    else:
        print(f"  all {len(rows)} scripts passed   ({total:.0f}s total)")
    print("  A 'defect' line is a known defect. See the section 8 of NOTES.md.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
