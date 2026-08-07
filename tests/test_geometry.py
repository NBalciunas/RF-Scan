"""Self-check for the sweep geometry. Type `python tests/test_geometry.py`.

The program calculates the FFT bin of a tone at the frequency F in each hop. Then it
joins the hops as SweepWorker._sweep_once does. The linear bin-to-Hz map of the
composite must give F again. The check fails if composite_geometry and the slice
operation do not agree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # find the modules

from terminal_v2 import compute_hop_freqs, composite_geometry, FFT_BINS


def check(sample_rate, rx_bw, center, span, overlap_pct):
    cfg = {"sample_rate": sample_rate, "rx_bw": rx_bw, "overlap_pct": overlap_pct}
    bw = min(sample_rate, rx_bw)
    cfg["hop_freqs"] = compute_hop_freqs(center, span, bw, overlap_pct)
    n_keep, f0, f1 = composite_geometry(cfg)
    hops  = cfg["hop_freqs"]
    total = len(hops) * n_keep
    binw  = sample_rate / FFT_BINS
    b0    = (FFT_BINS - n_keep) // 2
    for F in np.linspace(f0 + binw, f1 - binw, 97):
        errs = []
        for i, hc in enumerate(hops):
            b = int(round((F - (hc - sample_rate / 2)) / binw))   # the FFT bin of F
            if b0 <= b < b0 + n_keep:                             # inside the slice?
                idx  = i * n_keep + (b - b0)
                back = f0 + (idx / total) * (f1 - f0)             # the map of the app
                errs.append(abs(back - F))
        assert errs, f"{F/1e6:.3f} MHz falls in no slot (gap in coverage)"
        assert min(errs) < 3 * binw, \
            f"{F/1e6:.3f} MHz maps back {min(errs)/1e3:.1f} kHz off"
    print(f"  ok: sr={sample_rate/1e6:g}M bw={rx_bw/1e6:g}M span={span/1e6:g}M "
          f"olap={overlap_pct}% -> {len(hops)} hops, n_keep={n_keep}")


if __name__ == "__main__":
    check(10_000_000, 4_000_000, 2_400_000_000, 20_000_000, 30)   # the app defaults
    check(10_000_000, 10_000_000, 2_400_000_000, 20_000_000, 0)   # bw = sr, no overlap
    check(2_000_000, 4_000_000, 915_000_000, 5_000_000, 50)       # large overlap
    check(10_000_000, 4_000_000, 2_400_000_000, 4_000_000, 30)    # one hop only
    print("all geometry checks passed")
