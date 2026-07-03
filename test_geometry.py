"""Self-check for the sweep-composite geometry (run: python test_geometry.py).

Simulates where a tone at frequency F lands in each hop's FFT, stitches it the
way SweepWorker._sweep_once does, and asserts the composite's linear bin->Hz map
gives F back. Fails if composite_geometry / the slicing ever drift apart.
"""
import numpy as np

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
            b = int(round((F - (hc - sample_rate / 2)) / binw))   # FFT bin of F in hop i
            if b0 <= b < b0 + n_keep:                             # survives the slice?
                idx  = i * n_keep + (b - b0)
                back = f0 + (idx / total) * (f1 - f0)             # the app/worker map
                errs.append(abs(back - F))
        assert errs, f"{F/1e6:.3f} MHz falls in no slot (gap in coverage)"
        assert min(errs) < 3 * binw, \
            f"{F/1e6:.3f} MHz maps back {min(errs)/1e3:.1f} kHz off"
    print(f"  ok: sr={sample_rate/1e6:g}M bw={rx_bw/1e6:g}M span={span/1e6:g}M "
          f"olap={overlap_pct}% -> {len(hops)} hops, n_keep={n_keep}")


if __name__ == "__main__":
    check(10_000_000, 4_000_000, 2_400_000_000, 20_000_000, 30)   # app defaults
    check(10_000_000, 10_000_000, 2_400_000_000, 20_000_000, 0)   # bw == sr, no overlap
    check(2_000_000, 4_000_000, 915_000_000, 5_000_000, 50)       # sr-limited, heavy overlap
    check(10_000_000, 4_000_000, 2_400_000_000, 4_000_000, 30)    # single hop
    print("all geometry checks passed")
