"""Self-check for the sweep geometry. Type `python tests/test_geometry.py`.

Two different properties are checked.

  * The map. The program calculates the FFT bin of a tone at the frequency F in each
    hop, then it joins the hops as SweepWorker._sweep_once does. The linear bin-to-Hz
    map of the composite must give F again. The check fails if composite_geometry and
    the slice operation do not agree.
  * The coverage. The band that the composite covers must be the band that the user
    asked for. The map can be correct and still point at the wrong band.
"""

import numpy as np

from _support import Checks, run, stub_hardware

stub_hardware()          # this must run before the import of terminal_v2

from terminal_v2 import compute_hop_freqs, composite_geometry, FFT_BINS

# sample_rate, rx_bw, center, span, overlap_pct
CONFIGS = [
    ("the app defaults",   10_000_000,  4_000_000, 2_400_000_000, 20_000_000, 30),
    ("bw = sr, no overlap", 10_000_000, 10_000_000, 2_400_000_000, 20_000_000, 0),
    ("a large overlap",      2_000_000,  4_000_000,   915_000_000,  5_000_000, 50),
    ("one hop only",        10_000_000,  4_000_000, 2_400_000_000,  4_000_000, 30),
    ("bw = sr, overlap 30", 10_000_000, 10_000_000, 2_400_000_000, 20_000_000, 30),
]


def _geometry(sample_rate, rx_bw, center, span, overlap_pct):
    cfg = {"sample_rate": sample_rate, "rx_bw": rx_bw, "overlap_pct": overlap_pct}
    cfg["hop_freqs"] = compute_hop_freqs(center, span,
                                         min(sample_rate, rx_bw), overlap_pct)
    n_keep, f0, f1 = composite_geometry(cfg)
    return cfg, n_keep, f0, f1


def main():
    c = Checks("Sweep geometry (terminal_v2.py)")

    @c.check("a tone maps back to its own frequency through the composite")
    def _():
        for name, sr, bw, center, span, olap in CONFIGS:
            cfg, n_keep, f0, f1 = _geometry(sr, bw, center, span, olap)
            hops, binw = cfg["hop_freqs"], sr / FFT_BINS
            total, b0 = len(hops) * n_keep, (FFT_BINS - n_keep) // 2
            # The two bins at each end are not probed. f0 and f1 are the outer edges
            # of the band, but the map gives the left edge of each bin, thus the
            # boundary is ambiguous by half a bin. See the defect #15.
            guard = 2.5 * (f1 - f0) / total
            for F in np.linspace(f0 + guard, f1 - guard, 97):
                errs = []
                for i, hc in enumerate(hops):
                    b = int(round((F - (hc - sr / 2)) / binw))   # the FFT bin of F
                    if b0 <= b < b0 + n_keep:                    # inside the slice?
                        idx = i * n_keep + (b - b0)
                        back = f0 + (idx / total) * (f1 - f0)    # the map of the app
                        errs.append(abs(back - F))
                assert errs, f"{name}: {F/1e6:.3f} MHz falls in no slot"
                assert min(errs) < 3 * binw, \
                    f"{name}: {F/1e6:.3f} MHz maps back {min(errs)/1e3:.1f} kHz off"

    @c.check("n_keep stays inside the FFT and the slots tile the composite")
    def _():
        for name, sr, bw, center, span, olap in CONFIGS:
            cfg, n_keep, f0, f1 = _geometry(sr, bw, center, span, olap)
            hops = cfg["hop_freqs"]
            assert 2 <= n_keep <= FFT_BINS, f"{name}: n_keep {n_keep}"
            assert (FFT_BINS - n_keep) // 2 + n_keep <= FFT_BINS, name
            slot = (f1 - f0) / len(hops)
            if len(hops) > 1:
                step = hops[1] - hops[0]
                assert abs(slot - step) < sr / FFT_BINS, \
                    f"{name}: slot {slot/1e3:.1f} kHz vs hop step {step/1e3:.1f} kHz"

    @c.check("the hop count is enough for the span")
    def _():
        for name, sr, bw, center, span, olap in CONFIGS:
            cfg, n_keep, f0, f1 = _geometry(sr, bw, center, span, olap)
            assert f1 - f0 >= span - 1, \
                f"{name}: the composite is {(f1-f0)/1e6:.3f} MHz of {span/1e6:.3f} MHz"

    @c.check("with no overlap the composite is exactly the requested band")
    def _():
        for name, sr, bw, center, span, olap in CONFIGS:
            if olap != 0:
                continue
            _cfg, _n, f0, f1 = _geometry(sr, bw, center, span, olap)
            lo = center - span // 2
            assert abs(f0 - lo) < 1, f"{name}: starts at {f0/1e6:.3f}, want {lo/1e6:.3f}"

    c.note("the band that each configuration really covers:")
    for name, sr, bw, center, span, olap in CONFIGS:
        cfg, n_keep, f0, f1 = _geometry(sr, bw, center, span, olap)
        lo = center - span // 2
        c.note(f"  {name:<20} want {lo/1e6:9.3f}-{(lo+span)/1e6:9.3f}  "
               f"got {f0/1e6:9.3f}-{f1/1e6:9.3f}  blind {max(0.0, f0-lo)/1e3:7.1f} kHz")

    @c.check("the composite covers the band that the user asked for")
    def _():
        # The first hop must sit at start + step // 2. With start + hop_bw // 2 the
        # whole band slides up by overlap / 2, and the low end is never received.
        for name, sr, bw, center, span, olap in CONFIGS:
            _cfg, _n, f0, f1 = _geometry(sr, bw, center, span, olap)
            lo = center - span // 2
            assert f0 <= lo + 1, \
                (f"{name}: blind from {lo/1e6:.3f} to {f0/1e6:.3f} MHz "
                 f"({(f0-lo)/1e3:.1f} kHz of the requested span is never seen)")
            assert f1 >= lo + span - 1, f"{name}: stops short at {f1/1e6:.3f} MHz"

    @c.check("the extra coverage is at the top only, and never a gap at the bottom")
    def _():
        # ceil() on the hop count always gives some extra band. It must go above the
        # requested span, where it costs nothing, and never below it.
        for name, sr, bw, center, span, olap in CONFIGS:
            _cfg, _n, f0, f1 = _geometry(sr, bw, center, span, olap)
            lo = center - span // 2
            over = (f1 - (lo + span)) / 1e6
            assert over >= -1e-6, f"{name}: short by {-over:.3f} MHz"
            assert over < span / 1e6, f"{name}: {over:.3f} MHz of waste"

    # The DC bin stays in the middle of each kept slice. The LO leakage is removed
    # in _peak_hold_psd instead, thus no slot geometry has to change. test_dsp.py
    # holds that behaviour.
    c.note("DC lands mid-slot by design. The DC blocker in _peak_hold_psd handles it.")

    return c.report()


if __name__ == "__main__":
    run(main)
