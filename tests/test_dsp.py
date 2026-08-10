"""Self-check for the DSP and the badge rule. Type `python tests/test_dsp.py`.

The checks cover the functions of terminal that need no radio and no Qt:

  * _peak_hold_psd  - does a short burst stay visible?
  * signal_extent   - are the middle and the edges of a signal correct?
  * _detect_new_peak - does the scanner walk past a frequency that it already caught?
  * badge_for       - does the user read the correct name?

The functions need no radio. The stubs in _support replace adi, pyqtgraph and PyQt5
if they are absent, because terminal imports the three at the top of the file.
"""

import time

import numpy as np

from _support import Checks, run, stub_hardware

stub_hardware()          # this must run before the import of terminal

import terminal
from terminal import (SweepWorker, signal_extent, compute_hop_freqs,
                      composite_geometry, badge_for, peak_hold_bias_db,
                      FFT_BINS, MARK_MIN_SNR_DB, EMPTY_SLOT_DB)

APP_CFG = dict(sample_rate=10_000_000, rx_bw=4_000_000, overlap_pct=30,
               fp_memory_guard_hz=3_000_000)


def _app_cfg():
    cfg = dict(APP_CFG)
    cfg["hop_freqs"] = compute_hop_freqs(2_400_000_000, 20_000_000,
                                         min(cfg["sample_rate"], cfg["rx_bw"]),
                                         cfg["overlap_pct"])
    return cfg


def _worker(cfg, psd_bias_db=0.0):
    """Make a SweepWorker without QThread.__init__, thus without Qt.

    The composites below are made by hand and not by a peak hold, thus the default
    bias is 0 and the dB that comes back is the dB that the array holds."""
    w = SweepWorker.__new__(SweepWorker)
    w.cfg = cfg
    w._caught = []
    w._last_composite = None
    w._psd_bias_db = psd_bias_db
    return w


def _peak_hold(iq):
    """Run _peak_hold_psd on a throwaway instance.

    It must be an instance and not the class. _peak_hold_psd writes _psd_bias_db on
    self, and the class would keep that value for every later check."""
    return SweepWorker._peak_hold_psd(SweepWorker.__new__(SweepWorker), iq)


def _noise_buf(n_windows, sigma=0.01, seed=0):
    r = np.random.RandomState(seed)
    n = n_windows * FFT_BINS
    return ((r.randn(n) + 1j * r.randn(n)) * (sigma / np.sqrt(2))).astype(np.complex64)


def _add_burst(iq, window_i, f_norm=0.1, amp=1.0):
    s = window_i * FFT_BINS
    iq[s:s + FFT_BINS] += (amp * np.exp(2j * np.pi * f_norm
                                        * np.arange(FFT_BINS))).astype(np.complex64)
    return iq


def _snr(psd):
    return float(psd.max()) - float(np.median(psd))


def main():
    c = Checks("DSP and badge rule (terminal.py)")

    # ── Peak hold ─────────────────────────────────────────────────────────────

    @c.check("the peak-hold spectrum has one value for each FFT bin")
    def _():
        psd = _peak_hold(_noise_buf(100))
        assert psd.shape == (FFT_BINS,), psd.shape
        assert psd.dtype == np.float32, psd.dtype

    @c.check("a burst in 1 window of 100 is visible at its true amplitude")
    def _():
        iq = _add_burst(_noise_buf(100), window_i=50)
        got = _snr(_peak_hold(iq))
        assert got > 40, f"the burst is only {got:.0f} dB above the floor"

    @c.check("one FFT window alone would miss that burst")
    def _():
        # This is the reason that _peak_hold_psd exists. A single window covers
        # 102 us of a 50 ms dwell.
        iq = _add_burst(_noise_buf(100), window_i=50)
        one = _snr(_peak_hold(iq[:FFT_BINS]))
        assert one < 20, f"the first window already shows {one:.0f} dB"

    @c.check("the windows touch, thus a burst in the last window is found")
    def _():
        iq = _add_burst(_noise_buf(500), window_i=499)
        got = _snr(_peak_hold(iq))
        assert got > 40, f"the last window gives only {got:.0f} dB"

    @c.check("a buffer shorter than one FFT is padded, not refused")
    def _():
        assert _peak_hold(_noise_buf(1)[:500]).shape == (FFT_BINS,)

    @c.check("the peak hold covers a buffer of more than PSD_CHUNK_WINS windows")
    def _():
        # The old code truncated at 1024 windows, which is 105 ms at 10 Msps, while
        # the Dwell/hop box allows 5000 ms. A burst in the discarded tail was lost
        # and nothing said so. The blocks now run to the end of the buffer.
        for nwins, at in ((1200, 1150), (2100, 2099), (3000, 1500)):
            iq = _add_burst(_noise_buf(nwins), window_i=at)
            got = _snr(_peak_hold(iq))
            assert got > 40, \
                f"a burst at window {at} of {nwins} gives only {got:.0f} dB"

    @c.check("the block size does not change the answer")
    def _():
        # A running maximum over blocks must equal one maximum over everything.
        iq = _add_burst(_noise_buf(2500), window_i=2400)
        got = _peak_hold(iq)
        old = terminal.PSD_CHUNK_WINS
        terminal.PSD_CHUNK_WINS = 97          # an awkward size on purpose
        try:
            other = _peak_hold(iq)
        finally:
            terminal.PSD_CHUNK_WINS = old
        worst = float(np.abs(got - other).max())
        assert worst < 1e-4, f"the blocks disagree by {worst:.6f} dB"

    c.note("the cost of full coverage, against the dwell at 10 Msps:")
    for _dwell in (50, 200, 1000, 5000):
        _n = int(10e6 * _dwell / 1000)
        _buf = _noise_buf(_n // FFT_BINS, sigma=1.0, seed=3)
        _t0 = time.perf_counter()
        _peak_hold(_buf)
        _el = (time.perf_counter() - _t0) * 1000
        c.note(f"  dwell {_dwell:>4} ms -> {_n//FFT_BINS:>5} windows -> "
               f"{_el:6.0f} ms of CPU ({_el/_dwell:4.1%} of the dwell)")

    @c.check("the LO leakage does not make a peak at the middle of a hop slot")
    def _():
        # The DC bin is kept in the middle of every slice, thus a constant offset
        # would give one false peak in each hop slot, and the scanner would lock on
        # it. The DC blocker in _peak_hold_psd removes the offset before the FFT.
        iq = (_noise_buf(100, sigma=0.01) + 0.5).astype(np.complex64)
        psd = _peak_hold(iq)
        excess = float(psd[FFT_BINS // 2] - np.median(psd))
        c.note(f"a DC offset 50x the noise leaves {excess:+.1f} dB on the DC bin")
        assert excess < 6.0, f"the DC bin is {excess:.0f} dB above the floor"

    @c.check("the DC blocker keeps a real signal that sits next to the hop centre")
    def _():
        iq = _add_burst(_noise_buf(100, sigma=0.01), window_i=50,
                        f_norm=0.01, amp=1.0)          # 10 bins from DC
        psd = _peak_hold((iq + 0.5).astype(np.complex64))
        assert abs(int(psd.argmax()) - (FFT_BINS // 2 + 10)) <= 1, int(psd.argmax())

    # Defect #13: the floor is the median of a per-bin maximum, thus it is biased
    # high, and the bias grows with the dwell. peak_hold_bias_db removes it, and
    # _detect_new_peak subtracts it. Thus the SNR is a true SNR.
    c.note("the peak-hold floor of the same noise, raw and corrected:")
    _corrected = []
    for _dwell in (5, 25, 50, 100, 200, 1000):
        _n = int(10e6 * _dwell / 1000) // FFT_BINS
        _raw = float(np.median(_peak_hold(_noise_buf(_n, sigma=1.0, seed=7))))
        _cor = _raw - peak_hold_bias_db(_n)
        _corrected.append(_cor)
        c.note(f"  dwell {_dwell:>4} ms -> {_n:>5} windows -> raw {_raw:+.2f} dB, "
               f"corrected {_cor:+.2f} dB")

    @c.check("the corrected noise floor does not move with the dwell")
    def _():
        # Without the correction, an SNR that the program reports moves by more than
        # 3 dB when you change the Dwell/hop box only.
        spread = max(_corrected) - min(_corrected)
        c.note(f"the corrected floor moves {spread:.2f} dB across the dwell range")
        assert spread < 0.5, f"the corrected floor still moves {spread:.2f} dB"

    @c.check("peak_hold_bias_db agrees with the measured bias")
    def _():
        ref = float(np.median(_peak_hold(_noise_buf(1, sigma=1.0, seed=11))))
        for n in (48, 244, 488, 9765):
            got = float(np.median(_peak_hold(_noise_buf(n, sigma=1.0, seed=11))))
            want = peak_hold_bias_db(n) - peak_hold_bias_db(1)
            assert abs((got - ref) - want) < 0.7, \
                f"n={n}: measured {got - ref:.2f} dB, formula {want:.2f} dB"

    @c.check("the reported SNR of a real signal does not move with the dwell")
    def _():
        # This is what #13 costs a report: the same tone read a different SNR at a
        # different dwell, and the value was never a true SNR.
        snrs = []
        for nwin in (48, 488, 4000):
            iq = _add_burst(_noise_buf(nwin, sigma=0.02, seed=5),
                            window_i=nwin // 2, amp=1.0)
            psd = _peak_hold(iq)
            snrs.append(float(psd.max()) - float(np.median(psd))
                        + peak_hold_bias_db(nwin))
        spread = max(snrs) - min(snrs)
        c.note(f"tone SNR {min(snrs):.1f} to {max(snrs):.1f} dB, spread {spread:.2f} dB")
        assert spread < 1.0, f"the SNR still moves {spread:.2f} dB with the dwell"

    # ── The middle and the edges of a signal ──────────────────────────────────

    def _rect_spectrum(centre_hz=2_401_000_000, half_bw=1_000_000):
        freqs = np.linspace(2.395e9, 2.405e9, FFT_BINS)
        psd = np.full(FFT_BINS, -80.0)
        psd[np.abs(freqs - centre_hz) <= half_bw] = -40.0
        return freqs, psd

    @c.check("signal_extent measures the middle and the two edges of a signal")
    def _():
        freqs, psd = _rect_spectrum()
        got = signal_extent(freqs, psd)
        assert got is not None, "a 40 dB signal was not found"
        f_l, f_c, f_r = got
        assert abs(f_c - 2_401_000_000) < 50e3, f"middle off by {f_c-2.401e9:.0f} Hz"
        assert abs(f_l - 2_400_000_000) < 50e3, f"left off by {f_l-2.400e9:.0f} Hz"
        assert abs(f_r - 2_402_000_000) < 50e3, f"right off by {f_r-2.402e9:.0f} Hz"

    @c.check(f"an empty channel gives no markers (the limit is {MARK_MIN_SNR_DB:.0f} dB)")
    def _():
        r = np.random.RandomState(3)
        freqs = np.linspace(2.395e9, 2.405e9, FFT_BINS)
        psd = -80.0 + 0.5 * r.randn(FFT_BINS)
        assert signal_extent(freqs, psd) is None

    @c.check("the smooth operation pads with the edge value, thus the ends stay flat")
    def _():
        # A zero pad would lift the first bins from -80 dB toward 0 dB and make a
        # false peak at the end of the array.
        freqs = np.linspace(2.395e9, 2.405e9, FFT_BINS)
        psd = np.full(FFT_BINS, -80.0)
        psd[500:520] = -50.0
        f_l, f_c, f_r = signal_extent(freqs, psd)
        assert freqs[495] <= f_c <= freqs[525], \
            f"the middle landed at {f_c/1e6:.3f} MHz, not on the signal"
        assert f_l > freqs[10], "the left edge ran to the start of the array"

    @c.check("the centroid is steadier than the highest bin")
    def _():
        # The reason that the middle marker is a power centroid and not an argmax.
        freqs = np.linspace(2.395e9, 2.405e9, FFT_BINS)
        base = np.full(FFT_BINS, -80.0)
        base[462:562] = -50.0                     # a flat top, thus argmax wanders
        r = np.random.RandomState(11)
        cen, top = [], []
        for _ in range(200):
            psd = base + 2.0 * r.randn(FFT_BINS)
            got = signal_extent(freqs, psd)
            if got is None:
                continue
            cen.append(got[1])
            top.append(freqs[int(np.argmax(psd))])
        s_cen, s_top = float(np.std(cen)), float(np.std(top))
        c.note(f"centroid std {s_cen/1e3:.1f} kHz vs argmax std {s_top/1e3:.1f} kHz "
               f"over {len(cen)} frames")
        assert s_cen < s_top / 3.0, f"centroid {s_cen:.0f} vs argmax {s_top:.0f}"

    @c.check("signal_extent refuses a spectrum whose axes do not agree")
    def _():
        assert signal_extent(np.arange(10), np.zeros(5)) is None
        assert signal_extent(np.array([]), np.array([])) is None

    # ── Peak detection and the caught memory ──────────────────────────────────

    def _composite(cfg, fill=-70.0):
        n_keep, _f0, _f1 = composite_geometry(cfg)
        return np.full(len(cfg["hop_freqs"]) * n_keep, fill, dtype=np.float32)

    @c.check("the scanner finds the strongest peak and reports its height")
    def _():
        cfg = _app_cfg()
        comp = _composite(cfg)
        comp[500] = -50.0
        f, db = _worker(cfg)._detect_new_peak(comp)
        assert abs(db - 20.0) < 0.5, f"the peak reads {db:.1f} dB, expected 20"
        n_keep, f0, f1 = composite_geometry(cfg)
        want = f0 + (500 / len(comp)) * (f1 - f0)
        assert abs(f - want) < 20e3, f"the peak is at {f/1e6:.3f}, expected {want/1e6:.3f}"

    @c.check("a caught frequency is masked, thus the scanner walks to the next signal")
    def _():
        cfg = _app_cfg()
        comp = _composite(cfg)
        comp[500] = -50.0                 # the strongest
        comp[1500] = -55.0                # the next one
        w = _worker(cfg)
        first, _db = w._detect_new_peak(comp)
        w._caught = [(first, 0.0)]
        second, _db2 = w._detect_new_peak(comp)
        assert abs(second - first) > 2e6, \
            f"the scanner returned to {second/1e6:.3f} MHz"
        n_keep, f0, f1 = composite_geometry(cfg)
        want = f0 + (1500 / len(comp)) * (f1 - f0)
        assert abs(second - want) < 20e3

    @c.check("a hop that failed does not poison the noise floor")
    def _():
        # _sweep_once leaves EMPTY_SLOT_DB in the slot of a hop whose tune or rx
        # raised. Those slots must stay out of the median, or the floor falls and
        # every real bin looks like a large peak.
        cfg = _app_cfg()
        comp = _composite(cfg)
        n_keep, _f0, _f1 = composite_geometry(cfg)
        comp[:5 * n_keep] = EMPTY_SLOT_DB   # 5 of the 8 hops failed
        comp[1500] = -55.0                    # a real signal, 15 dB above the floor
        _f, db = _worker(cfg)._detect_new_peak(comp)
        c.note(f"with 5 of 8 hops empty the signal still reads {db:.0f} dB")
        assert abs(db - 15.0) < 0.5, f"{db:.0f} dB, expected 15 dB"

    @c.check("a sweep where every hop failed gives no peak")
    def _():
        cfg = _app_cfg()
        comp = _composite(cfg, fill=EMPTY_SLOT_DB)
        f, _db = _worker(cfg)._detect_new_peak(comp)
        assert f is None, f"it locked on {f}"

    @c.check("a band that is completely caught gives no peak")
    def _():
        cfg = _app_cfg()
        comp = _composite(cfg)
        _n, f0, f1 = composite_geometry(cfg)
        w = _worker(cfg)
        w._caught = [(f, 0.0) for f in np.arange(f0, f1, 2e6)]
        f, db = w._detect_new_peak(comp)
        assert f is None, f"it locked on {f}"

    @c.check("the held spectrum goes into the composite at the correct frequency")
    def _():
        cfg = _app_cfg()
        w = _worker(cfg)
        w._last_composite = _composite(cfg)
        held, sr = 2_400_000_000.0, cfg["sample_rate"]
        comp = w._composite_with_hold(held, np.full(FFT_BINS, -50.0, np.float32))
        moved = np.flatnonzero(comp != -70.0)
        _n, f0, f1 = composite_geometry(cfg)
        hz = lambda i: f0 + (i / len(comp)) * (f1 - f0)
        assert abs(hz(moved[0]) - (held - sr / 2)) < 30e3, hz(moved[0])
        assert abs(hz(moved[-1]) - (held + sr / 2)) < 30e3, hz(moved[-1])

    # ── The badge rule ────────────────────────────────────────────────────────

    def _res(probs, dets=()):
        best = max(probs, key=probs.get)
        return {"label": best, "confidence": probs[best], "probs": probs,
                "detections": [{"label": l, "share": s, "confidence": 0.9}
                               for l, s in dets]}

    @c.check("a quiet channel reads 'clear' and not 'unknown device'")
    def _():
        # 67% noise is a clear channel. The strongest class alone would call it a
        # device, which is why presence uses 1 - P(noise).
        text, kind = badge_for(_res({"droneA": 0.20, "droneB": 0.13, "noise": 0.67}))
        assert text == "clear (67% noise)", text
        assert kind == "none", kind

    @c.check("a confident device is named")
    def _():
        text, kind = badge_for(_res({"droneA": 0.93, "droneB": 0.02, "noise": 0.05}))
        assert text == "droneA (93%)" and kind == "device", (text, kind)

    @c.check("a device with no clear name reads 'unknown device'")
    def _():
        # A device is present (29% noise) but neither drone holds enough.
        text, kind = badge_for(_res({"droneA": 0.38, "droneB": 0.33, "noise": 0.29}))
        assert text.startswith("unknown device") and kind == "other", (text, kind)

    @c.check("two votes have precedence over the mean")
    def _():
        # The mean of two transmitters is 'unknown'. The votes give both names.
        text, kind = badge_for(_res({"droneA": 0.45, "droneB": 0.40, "noise": 0.15},
                                    dets=[("droneA", 0.5), ("droneB", 0.4)]))
        assert text == "droneA 50% + droneB 40%", text
        assert kind == "device", kind

    @c.check("one vote alone does not take the two-name path")
    def _():
        text, _k = badge_for(_res({"droneA": 0.93, "droneB": 0.02, "noise": 0.05},
                                  dets=[("droneA", 0.8)]))
        assert text == "droneA (93%)", text

    @c.check("a noise vote never becomes one of the two names")
    def _():
        text, _k = badge_for(_res({"droneA": 0.93, "droneB": 0.02, "noise": 0.05},
                                  dets=[("droneA", 0.6), ("noise", 0.3)]))
        assert "noise" not in text, text

    @c.check("a model with no noise class can never read 'clear'")
    def _():
        # p_dev is 1 - P(noise), and P(noise) is 0 when the class does not exist,
        # thus presence is always 100% whatever the radio hears. The panel and the
        # badge both warn about it. See the defect #12.
        for probs in ({"droneA": 0.51, "droneB": 0.49},
                      {"droneA": 0.95, "droneB": 0.05},
                      {"droneA": 0.34, "droneB": 0.33, "droneC": 0.33}):
            text, kind = badge_for(_res(probs))
            assert "clear" not in text, (probs, text)
            assert kind in ("device", "other"), (probs, kind)

    return c.report()


if __name__ == "__main__":
    run(main)
