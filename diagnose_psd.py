"""
diagnose_psd.py  –  Is the drone/noise dataset separable, or is it capped by
                    label noise from a bursty signal?

A bursty emitter sliced into ~1 ms FFT snapshots produces many "drone" windows
that were captured *between* bursts — they contain only the noise floor and are
effectively mislabeled.  This script quantifies that:

  1. Mean PSD per class (drone vs noise), with ±1σ band.
     If the two means sit on top of each other, the collapsed-PSD features do
     not separate the classes.

  2. Per-window "burst score" = peak-above-floor in dB
        score = max(psd) - median(psd)
     Bursts have a high score; silent gaps look like noise.  Plotted as
     overlaid histograms for drone vs noise.

  3. Estimated label-noise rate: fraction of DRONE windows whose burst score
     falls below the 95th percentile of NOISE windows — i.e. drone windows
     that are statistically indistinguishable from noise (captured in a gap).

Outputs two PNGs next to this script and prints a summary table.

    python diagnose_psd.py
    python diagnose_psd.py --cache_dir ./psd_cache
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — just write PNGs
import matplotlib.pyplot as plt


def load_class_windows(cache_dir: Path, cls: str):
    """Return (all_windows (N, BINS) float32, per_file_list) for one class."""
    cls_dir = cache_dir / cls
    per_file = []
    for npy in sorted(cls_dir.glob("*.npy")):
        per_file.append(np.load(str(npy)).astype(np.float32))
    if not per_file:
        raise RuntimeError(f"No .npy windows found in {cls_dir}")
    return np.concatenate(per_file, axis=0), per_file


def burst_score(windows: np.ndarray) -> np.ndarray:
    """Peak-above-floor (dB) per window: high = signal burst, low = silent."""
    return windows.max(axis=1) - np.median(windows, axis=1)


def main(args):
    cache_dir = Path(args.cache_dir)
    classes = args.classes

    data = {c: load_class_windows(cache_dir, c) for c in classes}

    # ── 1. mean PSD per class ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in classes:
        w = data[c][0]
        m, s = w.mean(0), w.std(0)
        bins = np.arange(w.shape[1])
        ax.plot(bins, m, label=f"{c}  (n={len(w):,})", lw=1.5)
        ax.fill_between(bins, m - s, m + s, alpha=0.15)
    ax.set_title("Mean PSD per class (±1σ).  Overlap ⇒ collapsed PSD doesn't separate classes")
    ax.set_xlabel("FFT bin"); ax.set_ylabel("Power (dBFS)")
    ax.legend(); ax.grid(alpha=0.3)
    out1 = Path(__file__).with_name("diag_mean_psd.png")
    fig.tight_layout(); fig.savefig(out1, dpi=110); plt.close(fig)

    # ── 2. burst-score distributions ───────────────────────────────────────────
    scores = {c: burst_score(data[c][0]) for c in classes}
    fig, ax = plt.subplots(figsize=(11, 5))
    lo = min(s.min() for s in scores.values())
    hi = max(s.max() for s in scores.values())
    edges = np.linspace(lo, hi, 60)
    for c in classes:
        ax.hist(scores[c], bins=edges, alpha=0.5, label=f"{c}  (n={len(scores[c]):,})", density=True)
    ax.set_title("Per-window burst score = peak − median (dB).  Drone bursts should sit right of noise")
    ax.set_xlabel("Peak-above-floor (dB)"); ax.set_ylabel("density")
    ax.legend(); ax.grid(alpha=0.3)
    out2 = Path(__file__).with_name("diag_burst_score.png")
    fig.tight_layout(); fig.savefig(out2, dpi=110); plt.close(fig)

    # ── 3. label-noise estimate ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"{'class':<10}{'windows':>10}{'burst score  mean':>20}{'p50':>8}{'p95':>8}")
    print("-" * 64)
    for c in classes:
        s = scores[c]
        print(f"{c:<10}{len(s):>10,}{s.mean():>16.1f} dB"
              f"{np.percentile(s,50):>8.1f}{np.percentile(s,95):>8.1f}")
    print("=" * 64)

    if "drone" in scores and "noise" in scores:
        noise_p95 = np.percentile(scores["noise"], 95)
        drone_silent = float((scores["drone"] <= noise_p95).mean())
        print(f"\nNoise 95th-pct burst score : {noise_p95:.1f} dB")
        print(f"Drone windows below it     : {drone_silent:6.1%}")
        print("  -> these drone windows are statistically indistinguishable from")
        print("     noise (captured in a gap between bursts) => likely MISLABELED.\n")
        ceiling = 1.0 - 0.5 * drone_silent
        print(f"Rough best-case accuracy if these stay mislabeled: ~{ceiling:.0%}")
        print("(A perfect model still scores ~50% on the silent drone windows.)\n")

    print(f"Wrote {out1.name} and {out2.name}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default="./psd_cache")
    p.add_argument("--classes", nargs="+", default=["drone", "noise"])
    main(p.parse_args())
