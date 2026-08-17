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
from terminal import (SweepWorker, signal_extent, signal_clipped, compute_hop_freqs,
                      composite_geometry, badge_for, device_votes, device_share,
                      peak_hold_bias_db,
                      band_floor_db, window_filled, window_state, lock_snr_db,
                      band_plan_name, occupied_span, near_known_device,
                      FFT_BINS, MARK_MIN_SNR_DB, EMPTY_SLOT_DB)

# peak_hits 1 keeps these checks on the question they ask, which is where the peak is
# and which frequencies are masked. The rule of the defect #31 has its own checks.
APP_CFG = dict(sample_rate=10_000_000, rx_bw=8_000_000, overlap_pct=30,
               fp_memory_guard_hz=3_000_000, peak_hits=1)


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
    # _detect_new_peak writes the candidate here. A QObject that never ran its
    # __init__ refuses a new attribute, thus every attribute it writes must exist.
    w._cand = None
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
        sig    = 1500                       # the bin of the signal
        failed = sig // n_keep              # every hop before that one failed
        n_hops = len(cfg["hop_freqs"])
        comp[:failed * n_keep] = EMPTY_SLOT_DB
        comp[sig] = -55.0                   # a real signal, 15 dB above the floor
        _f, db = _worker(cfg)._detect_new_peak(comp)
        c.note(f"with {failed} of {n_hops} hops empty the signal still reads {db:.0f} dB")
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
        assert text == "Clear (67% Noise)", text
        assert kind == "none", kind

    @c.check("a confident device is named")
    def _():
        text, kind = badge_for(_res({"droneA": 0.93, "droneB": 0.02, "noise": 0.05}))
        assert text == "droneA (93%)" and kind == "device", (text, kind)

    @c.check("a device with no clear name reads 'unknown device'")
    def _():
        # A device is present (29% noise) but neither drone holds enough.
        text, kind = badge_for(_res({"droneA": 0.38, "droneB": 0.33, "noise": 0.29}))
        assert text.startswith("Unknown Device") and kind == "other", (text, kind)

    # ── A named background class is not a device, §9.2 job 1. The four sites read one
    # AMBIENT_LABELS set, thus a wifi lock is background as a noise lock is. ────────
    @c.check("a wifi-dominant channel reads 'clear', not a device")
    def _():
        # 'wifi' is background, thus p_dev = 1 - P(noise|wifi|bluetooth) is 0.05 here
        # and neither the mean nor a vote names a device.
        text, kind = badge_for(_res({"droneA": 0.03, "wifi": 0.90, "noise": 0.07}))
        assert kind == "none" and text.startswith("Clear"), (text, kind)

    @c.check("a drone beside wifi traffic is still named")
    def _():
        # The mean is split across two background classes; the drone still holds it.
        text, kind = badge_for(_res({"droneA": 0.62, "wifi": 0.30, "noise": 0.08}))
        assert text == "droneA (62%)" and kind == "device", (text, kind)

    @c.check("a wifi vote is not a device vote")
    def _():
        res = _res({"wifi": 0.8, "noise": 0.2}, dets=(("wifi", 0.7),))
        assert device_votes(res) == [], device_votes(res)
        assert device_share(res) == 0.0, device_share(res)

    @c.check("a signal inside the window is not called wider than it")
    def _():
        freqs = np.linspace(-5e6, 5e6, 1024)
        psd = np.full(1024, -80.0)
        psd[400:600] = -40.0
        assert signal_clipped(freqs, psd) == (False, False)

    @c.check("a signal against the window edge is reported, Phase 4b task 4")
    def _():
        # The lock sits low and the signal runs off the top of the window. The plot
        # shows a part of it, thus the title must say so.
        freqs = np.linspace(-5e6, 5e6, 1024)
        psd = np.full(1024, -80.0)
        psd[700:] = -40.0
        low, high = signal_clipped(freqs, psd)
        assert high and not low, (low, high)

    @c.check("an empty window reports no clipped signal")
    def _():
        freqs = np.linspace(-5e6, 5e6, 1024)
        rng = np.random.RandomState(3)
        assert signal_clipped(freqs, -80.0 + rng.randn(1024) * 0.5) == (False, False)

    # ── The window that is full of signal. Phase 4b task 4, the second half. ──
    # A 20 MHz WiFi channel in a 10 MHz receiver window. Every bin holds signal and
    # the outer tenth is the skirt of the analog filter. The numbers follow 120 real
    # captures of WiFi channel 11, where the window sat 14 dB above the band floor at
    # the mean and 22 dB at its best.

    def _full_window(floor=-80.0, above=20.0, n=1024):
        psd = np.full(n, floor + above)
        k = n // 10
        psd[:k] = np.linspace(floor + above - 15.0, floor + above, k)
        psd[-k:] = np.linspace(floor + above, floor + above - 15.0, k)
        return psd

    @c.check("a window full of signal is reported, Phase 4b task 4")
    def _():
        assert window_filled(_full_window(), -80.0)

    @c.check("an empty window is not called full")
    def _():
        rng = np.random.RandomState(11)
        assert not window_filled(-80.0 + rng.randn(1024) * 1.5, -80.0)

    @c.check("a signal that covers a third of the window is not called full")
    def _():
        psd = np.full(1024, -80.0)
        psd[350:680] = -55.0
        assert not window_filled(psd, -80.0)

    @c.check("no band floor means no report of a full window")
    def _():
        # The Narrowband mode never sweeps, thus nothing measured the band around
        # the frequency and the program must not guess.
        assert not window_filled(_full_window(), None)

    @c.check("the Window state gives one short word for each case that can happen")
    def _():
        freqs = np.linspace(-5e6, 5e6, 1024)
        inside = np.full(1024, -80.0)
        inside[400:600] = -40.0                      # a signal inside the window
        wide = np.full(1024, -40.0)                  # it reaches past both sides
        low = np.full(1024, -80.0); low[:300] = -40.0
        high = np.full(1024, -80.0); high[700:] = -40.0
        assert window_state(True, freqs, wide) == "Full window"
        assert window_state(False, freqs, low) == "Low edge"
        assert window_state(False, freqs, high) == "High edge"
        assert window_state(False, freqs, inside) == "—"
        # "Both edges" can not happen and this holds the reason. signal_extent takes
        # its floor from the median of the window, thus the threshold is the median
        # plus 3 dB and half of the bins are below the median by definition. A run
        # that reaches both boundaries would need every bin above it. A signal that
        # runs off both sides fills the window, thus window_filled names it from a
        # floor that the sweep measured outside. That is the defect #38.
        assert signal_clipped(freqs, wide) == (False, False)
        assert window_state(False, freqs, wide) == "—"

    @c.check("the floor inside the window can not see a full window, which is why "
             "band_floor_db exists")
    def _():
        # This is the defect that the two functions above answer. signal_extent takes
        # its floor from the median of the window, thus a full window puts the median
        # inside the signal: it reported 3.24 MHz at the median on 120 real captures
        # of a 20 MHz channel, and signal_clipped saw no edge at the boundary on any.
        freqs = np.linspace(-5e6, 5e6, 1024)
        psd = _full_window()
        # Any structure inside the signal becomes "the signal" once the median sits
        # inside it. A real channel has more than this.
        psd[500:530] += 8.0
        ext = signal_extent(freqs, psd)
        assert ext is not None and (ext[2] - ext[0]) < 2e6, ext
        assert signal_clipped(freqs, psd) == (False, False)
        assert window_filled(psd, -80.0)        # the floor from outside answers it

    @c.check("band_floor_db leaves out the lock and the band next to it")
    def _():
        # 100 MHz of band at -80 dB, and a signal of 12 MHz around the lock. A floor
        # that holds the signal is not a floor.
        comp = np.full(2000, -80.0)
        f0, f1 = 2_390_000_000, 2_490_000_000
        comp[880:1120] = -30.0                  # 2434 to 2446 MHz
        got = band_floor_db(comp, f0, f1, 2_440_000_000, 10_000_000)
        assert abs(got - (-80.0)) < 0.5, got

    @c.check("band_floor_db does not count a hop that gave no data")
    def _():
        comp = np.full(2000, -80.0)
        comp[:900] = EMPTY_SLOT_DB
        got = band_floor_db(comp, 2_390_000_000, 2_490_000_000)
        assert abs(got - (-80.0)) < 0.5, got

    @c.check("band_floor_db gives None when the guard leaves too little band")
    def _():
        comp = np.full(100, -80.0)
        assert band_floor_db(comp, 2_435_000_000, 2_445_000_000,
                             2_440_000_000, 10_000_000) is None

    @c.check("a full window is still present, the defect #38")
    def _():
        # The release test asks "is the signal still there". A window full of signal
        # gave 12.4 dB against its own median and 32 dB against the band floor, on
        # the DJI video link, at a release threshold of 18 dB.
        psd = _full_window(floor=-80.0, above=20.0)
        assert lock_snr_db(psd, 8.2) < 18.0                  # what it did
        assert lock_snr_db(psd, 8.2, -80.0) >= 18.0          # what it does now

    @c.check("an empty window is still absent when the band floor is used")
    def _():
        rng = np.random.RandomState(5)
        psd = -80.0 + rng.randn(1024) * 1.5
        assert lock_snr_db(psd, 8.2, -80.0) < 18.0

    @c.check("a narrow signal is present either way")
    def _():
        psd = np.full(1024, -80.0)
        psd[500:530] = -45.0
        assert lock_snr_db(psd, 8.2) >= 18.0
        assert lock_snr_db(psd, 8.2, -80.0) >= 18.0

    # ── The band plan. §9 Phase 5b item 5, and the answer to §8 #37. ─────────
    # Every width below is a measurement of 2026-08-14 and not a guess.

    @c.check("a 20 MHz channel at a WiFi centre is named WiFi")
    def _():
        assert band_plan_name(2_462_000_000, 20_000_000) == "WiFi ch 11"
        # The narrowest reading of channel 11 over a hold of four sweeps, measured
        # on 2026-08-14 across three sessions: 12.95, 13.04, 13.13 and 15.31 MHz.
        assert band_plan_name(2_460_900_000, 12_950_000) == "WiFi ch 11"

    @c.check("the replayed drones are NOT named WiFi, though they sit on the raster")
    def _():
        # The measurement that decides the whole design. The AT9S was at 2438.15 MHz
        # and the DJI at 2441.48 MHz, which are 1.15 MHz from channel 6 and 0.52 MHz
        # from channel 7. Only the width keeps them out of the WiFi bucket.
        assert band_plan_name(2_438_150_000, 9_120_000) is None
        assert band_plan_name(2_441_480_000, 9_020_000) is None

    @c.check("a narrow signal at a WiFi centre is not WiFi")
    def _():
        assert band_plan_name(2_437_000_000, 1_000_000) != "WiFi ch 6"

    @c.check("Bluetooth is named by its width, and its advertising channels by name")
    def _():
        assert band_plan_name(2_402_100_000, 1_170_000) == "Bluetooth LE advertising"
        assert band_plan_name(2_426_000_000, 930_000) == "Bluetooth LE advertising"
        assert band_plan_name(2_450_000_000, 1_000_000) == "Bluetooth"

    @c.check("the band plan says nothing about a width it does not know")
    def _():
        assert band_plan_name(2_440_000_000, 5_000_000) is None
        assert band_plan_name(2_440_000_000, 9_000_000) is None

    # ── A frequency the model was trained at is never walked past. ───────────
    # The width test alone is not safe at 2440 MHz: the replay sits between WiFi
    # channels 6 and 7, and a drone beside the traffic of the room measured 10.68 to
    # 11.25 MHz against an 11 MHz limit, thus it crossed on some sweeps and not on
    # others. Measured 2026-08-14 in a real composite.

    @c.check("a candidate at a trained frequency is protected")
    def _():
        assert near_known_device(2_438_150_000, [2_440_000_000])   # the AT9S
        assert near_known_device(2_441_480_000, [2_440_000_000])   # the DJI

    @c.check("every trained frequency protects its own place, not only the first")
    def _():
        # A second drone recorded elsewhere in the band must be protected there too.
        known = [2_440_000_000, 2_470_000_000]
        assert near_known_device(2_469_000_000, known)
        assert near_known_device(2_441_000_000, known)
        assert not near_known_device(2_455_000_000, known)

    @c.check("a frequency far from any trained one is not protected")
    def _():
        assert not near_known_device(2_462_000_000, [2_440_000_000])
        assert not near_known_device(2_412_000_000, [2_440_000_000])

    @c.check("no trained frequency means no protection, and no crash")
    def _():
        assert not near_known_device(2_440_000_000, [])
        assert not near_known_device(2_440_000_000, None)

    @c.check("the guard covers where both drones were really found")
    def _():
        # The AT9S was 1.85 MHz below the record frequency and the DJI 1.48 MHz above.
        assert terminal.FP_KNOWN_GUARD_HZ >= 2_000_000

    @c.check("the limit sits between the drones and WiFi with margin on both sides")
    def _():
        # 9.12 MHz is the widest drone measured and 12.95 MHz the narrowest WiFi.
        # If a later session moves WIFI_MIN_WIDTH_HZ, this check says what it costs.
        assert 9_120_000 < terminal.WIFI_MIN_WIDTH_HZ < 12_950_000
        assert terminal.WIFI_MIN_WIDTH_HZ - 9_120_000 >= 1_500_000
        assert 12_950_000 - terminal.WIFI_MIN_WIDTH_HZ >= 1_500_000

    @c.check("occupied_span measures a signal wider than the receiver window")
    def _():
        # 100 MHz of band in 2000 bins. A 20 MHz channel around 2462 MHz, which no
        # 10 MHz window could measure from the inside.
        f0, f1 = 2_390_000_000, 2_490_000_000
        comp = np.full(2000, -80.0)
        lo = int((2_452_000_000 - f0) / (f1 - f0) * 2000)
        hi = int((2_472_000_000 - f0) / (f1 - f0) * 2000)
        comp[lo:hi] = -50.0
        span = occupied_span(comp, f0, f1, 2_462_000_000, -80.0)
        assert span is not None
        width = span[1] - span[0]
        assert 19e6 <= width <= 21e6, width / 1e6
        assert band_plan_name((span[0] + span[1]) / 2, width) == "WiFi ch 11"

    @c.check("occupied_span gives None where the band holds no signal")
    def _():
        comp = np.full(2000, -80.0)
        assert occupied_span(comp, 2_390_000_000, 2_490_000_000,
                             2_440_000_000, -80.0) is None

    @c.check("occupied_span stops at the edge of the signal, not at the band edge")
    def _():
        f0, f1 = 2_390_000_000, 2_490_000_000
        comp = np.full(2000, -80.0)
        comp[900:1000] = -50.0          # one block
        comp[1200:1300] = -50.0         # another, with a gap between them
        span = occupied_span(comp, f0, f1, 2_437_500_000, -80.0)
        assert span is not None and (span[1] - span[0]) < 6e6, span

    @c.check("the band floor is the quiet part of the band, not its middle")
    def _():
        # Half the band is busy. The percentile must stay on the quiet half, because
        # the room is not empty and the reference may not follow the traffic.
        comp = np.full(2000, -80.0)
        comp[1000:] = -40.0
        got = band_floor_db(comp, 2_390_000_000, 2_490_000_000)
        assert abs(got - (-80.0)) < 0.5, got

    @c.check("a vote names a device that the mean alone calls 'unknown'")
    def _():
        # The defect #35. A device is present (49% noise), no class holds enough of
        # the mean, and one vote names droneA. "unknown device" is the answer of last
        # resort, thus the vote wins. 82 of 500 real DJI captures read "unknown
        # device" while this test ran after the mean.
        text, kind = badge_for(_res({"droneA": 0.38, "droneB": 0.13, "noise": 0.49},
                                    dets=[("droneA", 0.35)]))
        assert text == "droneA (35%)" and kind == "device", (text, kind)

    @c.check("two votes have precedence over the mean")
    def _():
        # The mean of two transmitters is 'unknown'. The votes give both names.
        text, kind = badge_for(_res({"droneA": 0.45, "droneB": 0.40, "noise": 0.15},
                                    dets=[("droneA", 0.5), ("droneB", 0.4)]))
        assert text == "droneA (50%) + droneB (40%)", text
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

    @c.check("a peak of one sweep does not cause a lock, the defect #31")
    def _():
        cfg = _app_cfg(); cfg["peak_hits"] = 2
        w = _worker(cfg)
        n, _f0, _f1 = composite_geometry(cfg)
        comp = np.full(n, -80.0, dtype=np.float32)
        comp[n // 2] = -20.0
        f, _db = w._detect_new_peak(comp)
        assert f is None, f          # one sweep is not enough
        f, db = w._detect_new_peak(comp)
        assert f is not None and db > 50, (f, db)   # the second sweep agrees

    @c.check("a burst that moves to another frequency never causes a lock")
    def _():
        # The report of Nojus: the lock jumps to the side because another emitter is
        # loud for one sweep. Two sweeps that disagree must give nothing.
        cfg = _app_cfg(); cfg["peak_hits"] = 2
        w = _worker(cfg)
        n, _f0, _f1 = composite_geometry(cfg)
        for pos in (n // 4, 3 * n // 4, n // 4, 3 * n // 4):
            comp = np.full(n, -80.0, dtype=np.float32)
            comp[pos] = -20.0
            f, _db = w._detect_new_peak(comp)
            assert f is None, (pos, f)

    @c.check("the candidate keeps its strongest reading across the sweeps")
    def _():
        # A burst is not at its peak in every sweep. The lock must report the height
        # that was really seen, and not the height of the last sweep.
        cfg = _app_cfg(); cfg["peak_hits"] = 2
        w = _worker(cfg)
        n, _f0, _f1 = composite_geometry(cfg)
        comp = np.full(n, -80.0, dtype=np.float32)
        comp[n // 2] = -20.0
        w._detect_new_peak(comp)
        weaker = np.full(n, -80.0, dtype=np.float32)
        weaker[n // 2] = -50.0
        _f, db = w._detect_new_peak(weaker)
        assert db > 50, db           # -20 against the floor, not -50

    @c.check("a bursty drone is named although the mean of the buffer reads noise")
    def _():
        # The defect #29. A real DJI MINI 3 capture is 26% signal and 74% silence,
        # thus the mean reads 71% noise and the old rule answered 'clear' while a
        # quarter of the segments named the drone. The numbers are measured, from
        # 10 captures of session 1 on 2026-08-13.
        text, kind = badge_for(_res({"DJI-MINI-3": 0.29, "Radiolink": 0.00,
                                     "noise": 0.71},
                                    dets=[("DJI-MINI-3", 0.29)]))
        assert text == "DJI-MINI-3 (29%)", text
        assert kind == "device", kind

    @c.check("a quiet channel with no vote still reads 'clear'")
    def _():
        # The other half of #29: the votes give presence, thus they must not invent
        # it. An empty channel has no vote above min_share and reads clear.
        text, kind = badge_for(_res({"droneA": 0.05, "droneB": 0.03, "noise": 0.92}))
        assert text == "Clear (92% Noise)", text
        assert kind == "none", kind

    @c.check("a model with no noise class can never read 'clear'")
    def _():
        # p_dev is 1 - P(noise), and P(noise) is 0 when the class does not exist,
        # thus presence is always 100% whatever the radio hears. The panel and the
        # badge both warn about it. See the defect #12.
        for probs in ({"droneA": 0.51, "droneB": 0.49},
                      {"droneA": 0.95, "droneB": 0.05},
                      {"droneA": 0.34, "droneB": 0.33, "droneC": 0.33}):
            text, kind = badge_for(_res(probs))
            assert "Clear" not in text, (probs, text)
            assert kind in ("device", "other"), (probs, kind)

    return c.report()


if __name__ == "__main__":
    run(main)
