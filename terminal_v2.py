"""
terminal_v2.py  –  PlutoSDR monitor + RF-fingerprint detection.

v2 reworks the v1 drone/noise monitor into a fingerprinting tool:
  * ML is the spectrogram FingerprintModel (fp_spectrogram.py);
  * the worker runs a SCAN -> HOLD state machine: hop+scan, and when a peak crosses
    FP_PEAK_THRESH_DB park the LO on a fixed frequency, zoom in, and classify the
    held signal until the user Skips or the signal stops;
  * recording (raw fixed-LO IQ) has three kinds — a device fingerprint, band-swept
    noise, or narrowband noise at one frequency — all under fingerprint_data/;
  * a third "zoom" plot shows the currently-held narrowband signal.
"""

import os
import sys
import time
import json
import math
import re
import collections

import numpy as np
import adi
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

try:
    import torch  # presence check only — actual use is lazy-imported via fp_spectrogram
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ==========================================
# CONFIGURATION
# ==========================================

SDR_URI         = "ip:192.168.2.1"
SAMPLE_RATE     = 10_000_000
RX_BW_HZ        = 4_000_000
GAIN            = 10

CENTER_FREQ     = 2_400_000_000
TOTAL_SPAN_HZ   = 20_000_000
HOP_DWELL_MS    = 50
HOP_SETTLE_MS   = 50
HOP_OVERLAP_PCT = 30

FFT_BINS           = 1024
WATERFALL_ROWS     = 200
WF_SCALE_MIN_DBFS  = -10.0
WF_SCALE_MAX_DBFS  = 10.0

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "trained_model.pt")

# ── Fingerprint Locking state-machine params (tune against live signals) ──────
FP_PEAK_THRESH_DB    = 10.0      # peak-above-floor (dB) for a signal to count / trigger a lock
FP_HOLD_SETTLE_MS    = 20        # LO settle before grabbing a held capture
FP_GONE_S            = 2.5       # Locking: drop the lock if the signal's been gone this long
FP_MEMORY_TTL_S      = 30.0      # remembered catches are skipped for this long, then revisitable
FP_MEMORY_GUARD_HZ   = 3_000_000 # peaks within this of a remembered freq count as the same signal
FP_AUTO_DWELL_MS     = 5000      # Auto: dwell on a lock this long before judging it noise
FP_AUTO_NOISE_PCT    = 90        # Auto: auto-skip the lock when noise prob reaches this %
ML_INTERVAL_S        = 0.75      # min seconds between classifier runs (throttle; a signal's
                                 # identity doesn't change 20x/s, so per-loop inference is wasted)
NOISE_REC_EVERY_N    = 5         # wideband noise rec: save every Nth sweep. Undecimated, the
                                 # max-files ring refills in ~100 s, so long recordings keep only
                                 # the tail; every Nth spreads the same ring over N× the time
                                 # (more diverse noise) and cuts disk churn to match.

# ── Narrowband signal markers (live middle/edge overlay on the zoom plot) ─────
# Measured straight off the displayed PSD — no center freq, no bandwidth param —
# so the red lines track the true signal and drift with it.
MARK_MIN_SNR_DB     = 6.0   # peak must beat the noise floor by this before any line is drawn
MARK_EDGE_MARGIN_DB = 3.0   # an edge is where the signal crosses floor + this (occupied BW)
MARK_SMOOTH_BINS    = 5     # light smoothing so one noisy bin can't make the edges jump

# ── Stylesheet applied to every input widget so text is always white on dark ──
_INPUT_SS = (
    "QLineEdit, QSpinBox, QDoubleSpinBox {"
    "  color: #ffffff;"
    "  background-color: #2a2a2a;"
    "  border: 1px solid #555555;"
    "  border-radius: 3px;"
    "  padding: 1px 3px;"
    "}"
    "QLineEdit:read-only {"
    "  background-color: #1e1e1e;"
    "  color: #cccccc;"
    "}"
    "QSpinBox::up-button, QSpinBox::down-button,"
    "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
    "  background-color: #3a3a3a;"
    "  border: 1px solid #555555;"
    "}"
    "QPushButton {"
    "  color: #ffffff;"
    "  background-color: #3a3a3a;"
    "  border: 1px solid #666666;"
    "  border-radius: 3px;"
    "  padding: 3px 6px;"
    "}"
    "QPushButton:hover  { background-color: #4a4a4a; }"
    "QPushButton:pressed{ background-color: #222222; }"
    "QPushButton:checked{ background-color: #0064b4; border: 1px solid #3399dd; }"
    "QComboBox {"
    "  color: #ffffff;"
    "  background-color: #2a2a2a;"
    "  border: 1px solid #555555;"
    "  border-radius: 3px;"
    "  padding: 1px 3px;"
    "}"
    "QComboBox QAbstractItemView {"
    "  color: #ffffff;"
    "  background-color: #2a2a2a;"
    "  selection-background-color: #0064b4;"
    "}"
    "QCheckBox { color: #dddddd; }"
    "QCheckBox::indicator { border: 1px solid #666666; background: #2a2a2a; }"
    "QCheckBox::indicator:checked { background: #0064b4; }"
    "QProgressBar {"
    "  border: 1px solid #555555;"
    "  border-radius: 3px;"
    "  background: #1e1e1e;"
    "}"
    "QProgressBar::chunk { background: #0064b4; }"
)

# ==========================================
# HELPERS
# ==========================================

_VAL_COLOR = "#66ccff"   # accent for variable values in the status readouts
_VAL_RE = re.compile(r'([+\-]?\d+\.?\d*\s?(?:GHz|MHz|kHz|dB|ms|hops|files|sweeps?))')


def _hl(x):
    """Wrap a value in the status accent colour (rich text)."""
    return f"<span style='color:{_VAL_COLOR}'>{x}</span>"


def _hl_values(text):
    """Colour every numeric value-with-unit in a free-form status string."""
    return _VAL_RE.sub(lambda mt: _hl(mt.group(1)), text)


def compute_hop_freqs(center_freq, total_span, hop_bw, overlap_pct=0):
    overlap_hz = int(hop_bw * overlap_pct / 100.0)
    step       = hop_bw - overlap_hz
    n_hops     = math.ceil(total_span / step)
    start      = center_freq - total_span // 2
    return [int(start + hop_bw // 2 + i * step) for i in range(n_hops)]


def composite_geometry(cfg):
    """Geometry of the stitched sweep composite -> (n_keep, f_start, f_stop).

    Each hop's FFT spans sample_rate Hz, but hops advance by only `step` Hz
    (hop_bw minus overlap).  Stitching all FFT_BINS bins per hop would compress
    the frequency axis by ~sample_rate/step and show the same signal in several
    slots, so only the central n_keep bins (~step Hz) of every hop are kept —
    slots then tile contiguously and bin -> Hz is one linear map, shared by the
    worker (peak detect / hold overlay) and the GUI (axis, waterfall, hop lines).
    """
    sr   = float(cfg["sample_rate"])
    hops = cfg["hop_freqs"]
    if len(hops) > 1:
        step = float(hops[1] - hops[0])
    else:
        step = min(sr, float(cfg["rx_bw"])) * (1.0 - cfg["overlap_pct"] / 100.0)
    n_keep = max(2, min(FFT_BINS, int(round(FFT_BINS * step / sr))))
    half   = n_keep * (sr / FFT_BINS) / 2.0
    return n_keep, hops[0] - half, hops[-1] + half


def signal_extent(freqs, psd, min_snr_db=MARK_MIN_SNR_DB,
                  edge_margin_db=MARK_EDGE_MARGIN_DB, smooth_bins=MARK_SMOOTH_BINS):
    """True (LO-independent) middle and edges of the dominant signal in a spectrum.

    Read straight off the live PSD — no center frequency, no bandwidth assumption —
    so the result tracks the real signal and moves with it. Returns
    (f_left, f_center, f_right) in the same units as `freqs`, or None when nothing
    clears the noise floor.

      * edges  : walk out from the peak while the (smoothed) PSD stays above
                 floor + edge_margin_db — the band the signal actually occupies.
      * middle : power-weighted centroid across that band — the energy centre of
                 mass, which is steadier and truer than just the single peak bin.
    """
    n = len(psd)
    if n == 0 or len(freqs) != n:
        return None
    sm = np.asarray(psd, dtype=np.float64)
    if smooth_bins > 1 and n >= smooth_bins:
        L   = int(smooth_bins)
        k   = np.ones(L) / float(L)
        pad = L // 2
        # Edge-pad (NOT zero-pad): dBFS sits near -80, so a zero-padded convolution
        # would pull the end bins up toward 0 and fake a peak at the window edges.
        sm  = np.convolve(np.pad(sm, pad, mode="edge"), k, mode="valid")[:n]
    floor = float(np.median(sm))
    pk    = int(np.argmax(sm))
    if sm[pk] - floor < min_snr_db:            # no signal worth marking
        return None
    thr = floor + edge_margin_db
    l = pk
    while l > 0 and sm[l - 1] >= thr:
        l -= 1
    r = pk
    while r < n - 1 and sm[r + 1] >= thr:
        r += 1
    lin   = np.power(10.0, np.asarray(psd[l:r + 1], dtype=np.float64) / 10.0)  # dB power -> linear
    fseg  = np.asarray(freqs[l:r + 1], dtype=np.float64)
    denom = float(lin.sum())
    f_center = float((fseg * lin).sum() / denom) if denom > 0 else float(freqs[pk])
    return float(freqs[l]), f_center, float(freqs[r])


# ==========================================
# ML  (fingerprinting)
# ==========================================
# The spectrogram FingerprintModel lives in fp_spectrogram.py — imported lazily in
# PlutoApp._load_model so the GUI still starts if torch is missing.


def _write_iq_sidecar(iq_path, device, session, freq, cfg, n_samples, ts):
    """JSON metadata next to a recorded .iq (carries the recorded_device label)."""
    meta = {
        "recorded_device": device,
        "session"        : str(session),
        "center_freq"    : float(freq),
        "sample_rate"    : float(cfg["sample_rate"]),
        "bandwidth"      : float(cfg["rx_bw"]),
        "gain_db"        : int(cfg["gain"]),
        "n_samples"      : int(n_samples),
        "dtype"          : "complex64",
        "timestamp_ms"   : ts,
    }
    with open(os.path.splitext(iq_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


# ==========================================
# SWEEP WORKER
# ==========================================

class SweepWorker(QtCore.QThread):
    # display + control signals
    sweep_ready       = QtCore.pyqtSignal(object, object)        # composite, hop_bufs
    zoom_ready        = QtCore.pyqtSignal(object, object, float)  # freqs, psd, held_freq
    fingerprint_ready = QtCore.pyqtSignal(object)               # result dict
    mode_changed      = QtCore.pyqtSignal(str, float)           # mode label, held_freq
    caught_changed    = QtCore.pyqtSignal(object)               # list of remembered freqs
    hop_progress      = QtCore.pyqtSignal(int, int)
    status_msg        = QtCore.pyqtSignal(str)
    files_changed     = QtCore.pyqtSignal(int)

    _BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)

    def __init__(self, sdr, cfg, engine=None):
        super().__init__()
        self.sdr        = sdr
        self.cfg        = cfg
        self.engine     = engine          # fp_spectrogram.FingerprintModel (or None)
        self._stop      = False
        self._fq        = collections.deque()
        self._held_freq = None
        self._last_composite = None        # last full sweep, kept alive during a lock
        self._last_present_t = 0.0         # Locking: last time the locked signal was present
        self._last_infer_t   = 0.0         # throttle: last time the classifier ran
        self._caught         = []          # [(freq_hz, t_caught)] — memory of catches
        self._lock_t         = 0.0         # Auto: when the current lock began (dwell timer)
        self._last_class     = None        # Auto: most recent classifier result for this lock

    def stop(self):
        self._stop = True
        if not self.wait(4000):
            self.status_msg.emit(
                "Warning: sweep thread blocked on sdr.rx() — forcing terminate."
            )
            self.terminate()
            self.wait(1000)

    # ── geometry / DSP helpers ────────────────────────────────────────────────

    def _band_edges(self):
        _n, f0, f1 = composite_geometry(self.cfg)
        return f0, f1

    def _sweep_once(self):
        """One wideband hop sweep -> (composite, hop_bufs). Mirrors v1 SCAN."""
        hop_freqs  = self.cfg["hop_freqs"]
        n_hops     = len(hop_freqs)
        n_keep, _f0, _f1 = composite_geometry(self.cfg)
        b0         = (FFT_BINS - n_keep) // 2       # central slice of each hop's PSD
        composite  = np.full(n_hops * n_keep, -100.0, dtype=np.float32)
        hop_bufs   = {}
        for i, freq in enumerate(hop_freqs):
            if self._stop:
                return None, None
            self.hop_progress.emit(i, n_hops)
            try:
                self.sdr.rx_lo = int(freq)
            except Exception as e:
                self.status_msg.emit(f"Tune error hop {i}: {e}")
                continue
            time.sleep(self.cfg["settle_ms"] / 1000.0)
            try:
                raw = np.array(self.sdr.rx(), dtype=np.complex64)
            except Exception as e:
                self.status_msg.emit(f"RX error hop {i}: {e}")
                continue
            if raw is None or len(raw) == 0:
                continue
            hop_bufs[i] = raw
            # Peak-hold over the WHOLE dwell buffer (not just its first 102 µs):
            # the radio already paid for these samples, and bursty emitters are
            # invisible to a single window. Adds ~ms per hop vs the ~100 ms dwell.
            psd = self._peak_hold_psd(raw)
            composite[i * n_keep:(i + 1) * n_keep] = psd[b0:b0 + n_keep]
        return composite, hop_bufs

    def _peak_hold_psd(self, iq):
        """1024-bin dBFS PSD, peak-held across contiguous gap-free windows spanning
        the buffer, so a burst anywhere in the dwell shows at full amplitude (one
        window would miss it ~99.8% of the time at the default 50 ms dwell).
        Contiguous (not sparse) windows so a burst can't fall between them; batched
        FFT; the window cap bounds CPU/RAM however long the dwell is."""
        n = len(iq)
        if n < FFT_BINS:
            iq = np.pad(iq, (0, FFT_BINS - n)); n = FFT_BINS
        nwin = min(1024, n // FFT_BINS)
        seg  = iq[:nwin * FFT_BINS].reshape(nwin, FFT_BINS) * self._BLACKMAN
        mag  = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) / FFT_BINS
        return (20.0 * np.log10(mag.max(axis=0) + 1e-10)).astype(np.float32)

    def _narrowband_psd(self, iq, center):
        """Peak-hold dBFS spectrum of a held capture, mapped to absolute Hz."""
        psd = self._peak_hold_psd(iq)
        sr = self.cfg["sample_rate"]
        freqs = np.linspace(center - sr / 2, center + sr / 2, FFT_BINS)
        return freqs, psd

    def _composite_with_hold(self, held_freq, psd):
        """Overlay the live held-band spectrum onto the last full sweep, so the
        wideband view keeps showing the whole band (with the locked spot live)
        while parked in HOLD."""
        base = self._last_composite
        if base is None:                       # no prior sweep — synthesise a floor
            n_keep, _f0, _f1 = composite_geometry(self.cfg)
            base = np.full(len(self.cfg["hop_freqs"]) * n_keep, -100.0, dtype=np.float32)
        comp = base.copy()
        f_min, f_max = self._band_edges()
        span = f_max - f_min
        if span <= 0:
            return comp
        total = len(comp)
        sr = self.cfg["sample_rate"]
        s = int(round((held_freq - sr / 2 - f_min) / span * total))
        e = int(round((held_freq + sr / 2 - f_min) / span * total))
        s = max(0, min(total, s)); e = max(0, min(total, e))
        if e - s >= 2:
            comp[s:e] = np.interp(np.linspace(0.0, 1.0, e - s),
                                  np.linspace(0.0, 1.0, len(psd)),
                                  psd).astype(np.float32)
        return comp

    def _detect_new_peak(self, composite):
        """Strongest peak-above-floor whose frequency isn't already in memory."""
        f_min, f_max = self._band_edges()
        span  = f_max - f_min
        total = len(composite)
        med   = float(np.median(composite))
        masked = composite.copy()
        guard = float(self.cfg.get("fp_memory_guard_hz", FP_MEMORY_GUARD_HZ))
        for cf, _t in self._caught:
            s = int(round((cf - guard - f_min) / span * total))
            e = int(round((cf + guard - f_min) / span * total))
            s = max(0, min(total, s)); e = max(0, min(total, e))
            if e > s:
                masked[s:e] = -1e9
        idx = int(np.argmax(masked))
        if masked[idx] <= -1e8:                     # whole band already in memory
            return None, -999.0
        return f_min + (idx / total) * span, float(composite[idx]) - med

    def _remember(self, freq):
        """Add a caught frequency to memory (drop near-duplicates first)."""
        guard = float(self.cfg.get("fp_memory_guard_hz", FP_MEMORY_GUARD_HZ))
        self._caught = [(f, t) for (f, t) in self._caught if abs(f - freq) > guard]
        self._caught.append((float(freq), time.time()))
        self.caught_changed.emit([f for f, _t in self._caught])

    def _release_lock(self, msg):
        """Forget the current lock, remember it as caught, and return to SCAN.
        Caller still sets its local `mode = "SCAN"` and `continue`s."""
        self._remember(self._held_freq)
        self._held_freq = None
        self.mode_changed.emit("SCAN", 0.0)
        self.status_msg.emit(msg)

    def _prune_memory(self):
        """Forget catches older than the TTL so they can be revisited."""
        ttl = float(self.cfg.get("fp_memory_ttl_s", FP_MEMORY_TTL_S))
        now = time.time()
        kept = [(f, t) for (f, t) in self._caught if now - t < ttl]
        if len(kept) != len(self._caught):
            self._caught = kept
            self.caught_changed.emit([f for f, _t in self._caught])

    def _save_iq(self, iq, device, session, freq):
        d = os.path.join("./fingerprint_data", device, f"session_{session}")
        os.makedirs(d, exist_ok=True)
        ts    = int(time.time() * 1000)
        fpath = os.path.join(d, f"{device}_s{session}_{ts}.iq")
        iq.astype(np.complex64).tofile(fpath)
        _write_iq_sidecar(fpath, device, session, freq, self.cfg, len(iq), ts)
        self._fq.append(fpath)
        max_files = int(self.cfg.get("record_max_files", 1000))
        while len(self._fq) > max_files:
            old = self._fq.popleft()
            for p in (old, os.path.splitext(old)[0] + ".json"):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        self.status_msg.emit(f"Warning: could not remove {p}: {e}")
        self.files_changed.emit(len(self._fq))

    def _maybe_classify(self, iq):
        """Run the classifier, but at most once every ml_interval_s. Gated by the
        live ML on/off flag. Throttling keeps the per-frame CNN forward off the
        hot loop so the PSD/waterfall/markers stay responsive."""
        if self.engine is None or not self.cfg.get("ml_enabled", True):
            return
        now = time.time()
        if now - self._last_infer_t < float(self.cfg.get("ml_interval_s", ML_INTERVAL_S)):
            return
        self._last_infer_t = now
        try:
            res = self.engine.classify_iq(iq)
            self._last_class = res          # Auto reads this to judge the lock
            self.fingerprint_ready.emit(res)
        except Exception as e:
            self.status_msg.emit(f"Inference error: {e}")

    @staticmethod
    def _noise_prob(res):
        """Probability the classifier assigned to the 'noise' class (0.0 if the
        model has no noise class or hasn't classified this lock yet)."""
        return float((res or {}).get("probs", {}).get("noise", 0.0))

    # ── top-level dispatch ────────────────────────────────────────────────────

    def run(self):
        op = self.cfg.get("op_mode", "locking")
        if op == "wideband":
            self._run_wideband()
        elif op == "focus":
            self._run_focus()
        else:                       # locking + auto share the lock state-machine
            self._run_locking()

    def _run_locking(self):
        """Scan -> lock the strongest not-yet-caught signal -> HOLD it on a FIXED
        frequency (no nudge, no rescan, so the narrowband stays put) until the user
        hits Skip -> remember it and step to the next. The Caught list walks through
        every signal; entries expire after the TTL so they become revisitable."""
        mode = "SCAN"
        self.cfg["skip_lock"] = False
        self.cfg["jump_to"] = None
        self.mode_changed.emit("SCAN", 0.0)
        while not self._stop:
            jt = self.cfg.get("jump_to")
            if jt:                              # jump straight onto a chosen freq
                self.cfg["jump_to"] = None
                self._held_freq, mode = int(jt), "LOCK"
                self._last_present_t = self._lock_t = time.time()
                self._last_infer_t = 0.0        # classify the new lock immediately
                self._last_class = None         # Auto: judge this lock on fresh results
                self.mode_changed.emit("LOCK", float(jt))
                self.status_msg.emit(f"Jumped to {jt/1e6:.3f} MHz — Skip to advance")
            if mode == "SCAN":
                t0 = time.perf_counter()
                composite, hop_bufs = self._sweep_once()
                if composite is None:
                    return
                self._last_composite = composite
                self.sweep_ready.emit(composite, hop_bufs)
                self.status_msg.emit(
                    f"Scan: {(time.perf_counter()-t0)*1000:.0f} ms  |  "
                    f"{len(self.cfg['hop_freqs'])} hops")
                self._prune_memory()
                f, peak_db = self._detect_new_peak(composite)
                if f is not None and peak_db >= float(
                        self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB)):
                    self._held_freq, mode = f, "LOCK"
                    self._last_present_t = self._lock_t = time.time()
                    self._last_infer_t = 0.0    # classify the new lock immediately
                    self._last_class = None     # Auto: judge this lock on fresh results
                    self.mode_changed.emit("LOCK", f)
                    self.status_msg.emit(
                        f"Locked +{peak_db:.0f} dB @ {f/1e6:.3f} MHz — Skip to advance")
            else:  # LOCK — fixed-freq hold until skipped or the signal stops
                if self._stop:
                    break
                if self.cfg.get("skip_lock"):
                    self.cfg["skip_lock"] = False
                    self._release_lock("Skipped — scanning for the next signal")
                    mode = "SCAN"
                    continue
                try:
                    self.sdr.rx_lo = int(self._held_freq)
                except Exception as e:
                    self.status_msg.emit(f"Lock tune error: {e}")
                    mode = "SCAN"; self.mode_changed.emit("SCAN", 0.0); continue
                time.sleep(self.cfg.get("fp_hold_settle_ms", FP_HOLD_SETTLE_MS) / 1000.0)
                try:
                    iq = np.array(self.sdr.rx(), dtype=np.complex64)
                except Exception as e:
                    self.status_msg.emit(f"Lock RX error: {e}"); continue
                if iq is None or len(iq) == 0:
                    continue
                freqs, psd = self._narrowband_psd(iq, self._held_freq)
                self.zoom_ready.emit(freqs, psd, float(self._held_freq))
                comp = self._composite_with_hold(self._held_freq, psd)
                if comp is not None:
                    self.sweep_ready.emit(comp, {})
                self._maybe_classify(iq)
                # Auto mode: once we've dwelled long enough on the lock, hand it to
                # the classifier — if it reads as noise, skip on automatically so the
                # user never has to Skip past a noise lock by hand. A real device
                # (noise below threshold) is left held, exactly like plain Locking.
                if self.cfg.get("op_mode") == "auto":
                    dwell = float(self.cfg.get("auto_dwell_ms", FP_AUTO_DWELL_MS)) / 1000.0
                    thr   = float(self.cfg.get("auto_noise_pct", FP_AUTO_NOISE_PCT)) / 100.0
                    noise_p = self._noise_prob(self._last_class)
                    if (self._last_class is not None
                            and time.time() - self._lock_t >= dwell
                            and noise_p >= thr):
                        self._release_lock(
                            f"Auto-skip: noise {noise_p:.0%} ≥ {thr:.0%} "
                            f"after {dwell*1000:.0f} ms")
                        mode = "SCAN"
                        continue
                # Auto-advance if the signal's been gone a while (e.g. the noise it
                # locked onto stopped). No nudge — the LO stays fixed.
                thresh = float(self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB))
                if (float(psd.max()) - float(np.median(psd))) >= thresh:
                    self._last_present_t = time.time()
                elif time.time() - self._last_present_t >= FP_GONE_S:
                    self._release_lock("Signal gone — moving on")
                    mode = "SCAN"
                    continue
                self.status_msg.emit(
                    f"Locked @ {self._held_freq/1e6:.3f} MHz — Skip to advance")

    def _run_wideband(self):
        """Continuous full-band scan, no focus/hold — just the wideband view.
        If 'record' is on, save every NOISE_REC_EVERY_N-th sweep's raw IQ as the
        noise class (training caps noise at ~83 files, so sparser saves spanning
        more wall-clock time beat contiguous ones)."""
        self.mode_changed.emit("WIDEBAND", 0.0)
        sweep_i = 0
        while not self._stop:
            t0 = time.perf_counter()
            composite, hop_bufs = self._sweep_once()
            if composite is None:
                return
            self._last_composite = composite
            self.sweep_ready.emit(composite, hop_bufs)
            rec = (bool(self.cfg.get("record"))
                   and self.cfg.get("record_kind") == "noise_band")
            if rec and sweep_i % NOISE_REC_EVERY_N == 0:
                session = self.cfg.get("record_session", "1")
                for i, raw in hop_bufs.items():
                    if len(raw):
                        self._save_iq(raw, "noise", session,
                                      int(self.cfg["hop_freqs"][i]))
            sweep_i += 1
            el  = (time.perf_counter() - t0) * 1000
            tag = (f"  |  REC noise: {len(self._fq)} files "
                   f"(1/{NOISE_REC_EVERY_N} sweeps)" if rec else "")
            self.status_msg.emit(
                f"Wideband: {el:.0f} ms  |  {len(self.cfg['hop_freqs'])} hops{tag}")

    def _run_focus(self):
        """Park on one frequency (focus_freq), show the zoom only (no band sweep),
        classify if a model is loaded, and — if 'record' is on — save the held IQ
        as the device's fingerprints. Used for fingerprint recording."""
        freq = int(self.cfg.get("focus_freq", self.cfg["center_freq"]))
        try:
            self.sdr.rx_lo = freq
        except Exception as e:
            self.status_msg.emit(f"Focus tune error: {e}"); return
        self.mode_changed.emit("FOCUS", float(freq))
        time.sleep(self.cfg.get("fp_hold_settle_ms", FP_HOLD_SETTLE_MS) / 1000.0)
        self._last_infer_t = 0.0        # classify the parked signal immediately
        while not self._stop:
            nf = int(self.cfg.get("focus_freq", freq))      # live retune if grabbed
            if nf != freq:
                try:
                    self.sdr.rx_lo = nf
                except Exception as e:
                    # keep the old freq: adopting nf here would label captures
                    # with a frequency we never actually tuned to
                    self.status_msg.emit(f"Focus tune error: {e}")
                    time.sleep(0.5); continue
                freq = nf
                self.mode_changed.emit("FOCUS", float(freq))
                time.sleep(self.cfg.get("fp_hold_settle_ms", FP_HOLD_SETTLE_MS) / 1000.0)
            try:
                iq = np.array(self.sdr.rx(), dtype=np.complex64)
            except Exception as e:
                self.status_msg.emit(f"Focus RX error: {e}"); continue
            if iq is None or len(iq) == 0:
                continue
            freqs, psd = self._narrowband_psd(iq, freq)
            self.zoom_ready.emit(freqs, psd, float(freq))
            self._maybe_classify(iq)
            kind = self.cfg.get("record_kind", "device")
            if self.cfg.get("record") and kind in ("device", "noise_freq"):
                # noise_freq writes a frequency-matched negative to the noise class.
                label = "noise" if kind == "noise_freq" else \
                    self.cfg.get("record_device", "deviceA")
                self._save_iq(iq, label, self.cfg.get("record_session", "1"), freq)
                self.status_msg.emit(
                    f"{'NOISE' if kind == 'noise_freq' else 'DEVICE'} REC "
                    f"{label}/s{self.cfg.get('record_session','1')} @ "
                    f"{freq/1e6:.3f} MHz: {len(self._fq)} files")
            else:
                self.status_msg.emit(f"Focus @ {freq/1e6:.3f} MHz")


# ==========================================
# MAIN APPLICATION
# ==========================================

class PlutoApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR Monitor  +  RF-Fingerprint detection (v2)")
        self.resize(1600, 950)

        self.cfg = {
            "sample_rate"     : SAMPLE_RATE,
            "rx_bw"           : RX_BW_HZ,
            "gain"            : GAIN,
            "center_freq"     : CENTER_FREQ,
            "total_span"      : TOTAL_SPAN_HZ,
            "dwell_ms"        : HOP_DWELL_MS,
            "settle_ms"       : HOP_SETTLE_MS,
            "overlap_pct"     : HOP_OVERLAP_PCT,
            "hop_freqs"       : [],
            # ── v2 fingerprinting ──
            "op_mode"             : "auto",        # auto | locking | wideband | focus
            "ml_enabled"          : True,         # run the classifier (live toggle)
            "ml_interval_s"       : ML_INTERVAL_S,# throttle: min seconds between inferences
            "record"              : False,        # save data appropriate to the mode
            "record_kind"         : "device",     # Record toggle saves: device | noise
            "skip_lock"           : False,        # Locking/Auto mode: advance to next signal
            "jump_to"             : None,         # Locking/Auto mode: lock onto this freq now
            "auto_dwell_ms"       : FP_AUTO_DWELL_MS,   # Auto: dwell before judging a lock
            "auto_noise_pct"      : FP_AUTO_NOISE_PCT,  # Auto: noise % that triggers a skip
            "record_device"       : "deviceA",
            "record_session"      : "1",
            "focus_freq"          : CENTER_FREQ,
            "record_max_files"    : 1000,
            "fp_peak_thresh_db"   : FP_PEAK_THRESH_DB,
            "fp_hold_settle_ms"   : FP_HOLD_SETTLE_MS,
            "fp_memory_ttl_s"     : FP_MEMORY_TTL_S,
            "fp_memory_guard_hz"  : FP_MEMORY_GUARD_HZ,
        }
        self._recompute_hops()

        self._engine      = None

        self._build_ui()

        if os.path.exists(DEFAULT_MODEL_PATH):
            self.w_model_path.setText(DEFAULT_MODEL_PATH)
            self._load_model(DEFAULT_MODEL_PATH)

        self.sdr = None
        try:
            self.sdr = adi.Pluto(SDR_URI)
            self._push_sdr_settings()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "SDR Connection Error",
                f"Could not connect to PlutoSDR at {SDR_URI}:\n\n{e}\n\n"
                "Check the URI, USB/network connection, and PlutoSDR firmware."
            )
            self.status_lbl.setText(f"SDR offline: {e}")
            return

        self.wf_data = self._make_waterfall_buf()
        self.img.setImage(self.wf_data, autoLevels=False)
        self._update_waterfall_rect()
        self._start_worker()

    # ── SDR / hop helpers ─────────────────────────────────────────────────────

    def _recompute_hops(self):
        effective_bw = min(self.cfg["sample_rate"], self.cfg["rx_bw"])
        self.cfg["hop_freqs"] = compute_hop_freqs(
            self.cfg["center_freq"], self.cfg["total_span"],
            effective_bw, self.cfg["overlap_pct"])
        self.n_hops        = len(self.cfg["hop_freqs"])
        # Axis/waterfall extent = what the stitched composite actually covers
        # (must match the worker's mapping bin-for-bin).
        n_keep, f0, f1     = composite_geometry(self.cfg)
        self.total_bins    = self.n_hops * n_keep
        self.f_global_min  = f0
        self.f_global_max  = f1
        self._slot_hz      = (f1 - f0) / self.n_hops
        self._effective_bw = effective_bw

    def _push_sdr_settings(self):
        self.sdr.sample_rate           = int(self.cfg["sample_rate"])
        self.sdr.rx_rf_bandwidth       = int(self.cfg["rx_bw"])
        self.sdr.rx_lo                 = int(self.cfg["hop_freqs"][0])
        # Fixed manual gain (AGC off) — consistent level matters for fingerprints.
        try:
            self.sdr.gain_control_mode_chan0 = "manual"
        except Exception:
            pass
        self.sdr.rx_hardwaregain_chan0 = int(self.cfg["gain"])
        self.sdr.rx_buffer_size        = max(
            1024, int(self.cfg["sample_rate"] * self.cfg["dwell_ms"] / 1000.0))

    def _make_waterfall_buf(self):
        return np.full((self.total_bins, WATERFALL_ROWS),
                       WF_SCALE_MIN_DBFS, dtype=np.float32)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self.win = pg.GraphicsLayoutWidget()
        root.addWidget(self.win, stretch=5)
        _axis_w = 64   # shared left-axis width so the four plots line up in columns

        # ── wideband spectrum  (row 0, col 0) ───────────────────────────────────
        self.p1 = self.win.addPlot(row=0, col=0, title="Wideband Spectrum")
        self.p1.setLabel("bottom", "Frequency", units="Hz")
        self.p1.setLabel("left",   "Power",     units="dBFS")
        # Disable auto-range so per-sweep setData() can't re-fit the view.
        self.p1.enableAutoRange(x=False, y=False)
        self.p1.setXRange(self.f_global_min, self.f_global_max, padding=0)
        self.p1.setYRange(WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS, padding=0)
        self.p1.setMouseEnabled(x=True, y=True)
        self.p1.setMenuEnabled(False)
        self.p1.showGrid(x=True, y=True, alpha=0.25)
        self.p1.getAxis("left").setWidth(_axis_w)
        self.hop_lines = []
        self._rebuild_hop_lines()
        self.curve = self.p1.plot(pen=pg.mkPen("y", width=1))

        # ── wideband waterfall  (row 1, col 0) ──────────────────────────────────
        self.p2 = self.win.addPlot(row=1, col=0, title="Wideband Waterfall")
        self.p2.setLabel("bottom", "Frequency", units="Hz")
        self.p2.setLabel("left",   "Time",      units="sweeps")
        self.p2.setXLink(self.p1)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setMenuEnabled(False)
        self.p2.getViewBox().setAutoVisible(x=False, y=False)
        self.p2.enableAutoRange(x=False, y=False)
        self.p2.getAxis("left").setWidth(_axis_w)
        self.img = pg.ImageItem(axisOrder="col-major")
        self.p2.addItem(self.img)
        cmap = pg.colormap.get("viridis")
        self.img.setLookupTable(cmap.getLookupTable())
        self.img.setLevels([WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS])

        # ── zoom spectrum  (row 0, col 1) — the currently-held signal ───────────
        self.p_zoom = self.win.addPlot(row=0, col=1, title="Narrowband Spectrum")
        self.p_zoom.setLabel("bottom", "Frequency", units="Hz")
        self.p_zoom.setLabel("left",   "Power",     units="dBFS")
        self.p_zoom.setMouseEnabled(x=True, y=True)   # pan/zoom the held view
        self.p_zoom.setMenuEnabled(True)
        self.p_zoom.showGrid(x=True, y=True, alpha=0.25)
        self.p_zoom.setYRange(WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS, padding=0)
        self.p_zoom.getAxis("left").setWidth(_axis_w)
        self.zoom_curve = self.p_zoom.plot(pen=pg.mkPen("c", width=1))

        # Live signal markers (toggled from the panel): a solid red line at the
        # signal's true middle and two dashed red lines at its edges. Positions
        # come from signal_extent() each frame, so they drift with the signal.
        _mid_pen  = pg.mkPen(color=(255, 40, 40), width=2)
        _edge_pen = pg.mkPen(color=(255, 40, 40), width=1, style=QtCore.Qt.DashLine)
        self.mid_line   = pg.InfiniteLine(angle=90, movable=False, pen=_mid_pen)
        self.edge_lines = [pg.InfiniteLine(angle=90, movable=False, pen=_edge_pen)
                           for _ in range(2)]
        for _ln in (self.mid_line, *self.edge_lines):
            _ln.setVisible(False)
            _ln.setZValue(10)                 # keep markers above the curve
            self.p_zoom.addItem(_ln, ignoreBounds=True)
        self._last_zoom = None                # (freqs, psd) cache for instant retoggle

        # ── zoom waterfall  (row 1, col 1) ──────────────────────────────────────
        self.p_zoom_wf = self.win.addPlot(row=1, col=1, title="Narrowband Waterfall")
        self.p_zoom_wf.setLabel("bottom", "Frequency", units="Hz")
        self.p_zoom_wf.setLabel("left",   "Time",      units="holds")
        self.p_zoom_wf.setXLink(self.p_zoom)
        self.p_zoom_wf.setMouseEnabled(x=True, y=False)   # x-pan follows p_zoom
        self.p_zoom_wf.setMenuEnabled(True)
        self.p_zoom_wf.getViewBox().setAutoVisible(x=False, y=False)
        self.p_zoom_wf.enableAutoRange(x=False, y=False)
        self.p_zoom_wf.getAxis("left").setWidth(_axis_w)
        self.zoom_wf_img = pg.ImageItem(axisOrder="col-major")
        self.p_zoom_wf.addItem(self.zoom_wf_img)
        self.zoom_wf_img.setLookupTable(cmap.getLookupTable())
        self.zoom_wf_img.setLevels([WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS])
        self.zoom_wf_data = np.full((FFT_BINS, WATERFALL_ROWS),
                                    WF_SCALE_MIN_DBFS, dtype=np.float32)
        self.zoom_wf_img.setImage(self.zoom_wf_data, autoLevels=False)
        _zc, _zsr = self.cfg["center_freq"], self.cfg["sample_rate"]
        self.zoom_wf_img.setRect(QtCore.QRectF(_zc - _zsr / 2, 0, _zsr, WATERFALL_ROWS))
        self.p_zoom.setXRange(_zc - _zsr / 2, _zc + _zsr / 2, padding=0)
        self.p_zoom_wf.setYRange(0, WATERFALL_ROWS, padding=0)

        # Wideband column wider than the zoom column.
        try:
            self.win.ci.layout.setColumnStretchFactor(0, 2)
            self.win.ci.layout.setColumnStretchFactor(1, 1)
        except Exception:
            pass

        # ── right side panel ───────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        # Apply the global input stylesheet to the whole panel so every
        # QLineEdit / QSpinBox / QDoubleSpinBox / QPushButton inside it
        # inherits white-on-dark styling without individual setStyleSheet calls.
        panel.setStyleSheet(_INPUT_SS)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setAlignment(QtCore.Qt.AlignTop)
        vbox.setSpacing(3)

        def section(title):
            vbox.addSpacing(8)
            lbl = QtWidgets.QLabel(f"<b>{title}</b>")
            lbl.setStyleSheet("color: #ffffff; font-size: 12px;")
            vbox.addWidget(lbl)

        def labeled(label_text, widget):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
            vbox.addWidget(lbl)
            vbox.addWidget(widget)
            return widget

        # ══════════════════════════════════════════════════════
        # ML INFERENCE SECTION
        # ══════════════════════════════════════════════════════
        section("ML Inference")

        self.det_badge = QtWidgets.QLabel("NO MODEL LOADED")
        self.det_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.det_badge.setFixedHeight(40)
        self.det_badge.setFont(QtGui.QFont("Monospace", 11, QtGui.QFont.Bold))
        self._set_badge_style("none")
        vbox.addWidget(self.det_badge)

        self._conf_bars   = {}
        self._conf_labels = {}
        self._conf_container = QtWidgets.QWidget()
        conf_grid = QtWidgets.QGridLayout(self._conf_container)
        conf_grid.setContentsMargins(0, 2, 0, 2)
        conf_grid.setVerticalSpacing(3)
        vbox.addWidget(self._conf_container)

        vbox.addSpacing(4)
        model_row = QtWidgets.QHBoxLayout()
        self.w_model_path = QtWidgets.QLineEdit()
        self.w_model_path.setPlaceholderText("trained_model.pt")
        self.w_model_path.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.w_model_path)
        model_row.addWidget(browse_btn)
        vbox.addLayout(model_row)

        load_row = QtWidgets.QHBoxLayout()
        load_btn = QtWidgets.QPushButton("⟳  Load / Reload Model")
        load_btn.setFixedHeight(28)
        load_btn.clicked.connect(self._on_load_model_btn)
        # Live ML on/off — gates the classifier in the worker. OFF makes the loop
        # just rx -> PSD -> display (no per-frame CNN), the big speed win.
        self.w_ml_toggle = QtWidgets.QPushButton("ML Inference: ON")
        self.w_ml_toggle.setCheckable(True)
        self.w_ml_toggle.setChecked(self.cfg.get("ml_enabled", True))
        self.w_ml_toggle.setFixedHeight(28)
        self.w_ml_toggle.toggled.connect(self._on_ml_toggle)
        load_row.addWidget(load_btn, 1)
        load_row.addWidget(self.w_ml_toggle)
        vbox.addLayout(load_row)

        # ══════════════════════════════════════════════════════
        # MODE SECTION  (Locking / Auto / Wideband / Narrowband)
        # ══════════════════════════════════════════════════════
        section("Mode")
        mode_row = QtWidgets.QHBoxLayout()
        self._mode_btns  = {}
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for key, label in (("auto", "Auto"), ("locking", "Locking"),
                           ("wideband", "Wideband"), ("focus", "Narrowband")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(26)
            b.clicked.connect(lambda _=False, k=key: self._on_mode_btn(k))
            self._mode_group.addButton(b)
            self._mode_btns[key] = b
            mode_row.addWidget(b)
        self._mode_btns["auto"].setChecked(True)
        vbox.addLayout(mode_row)

        # Second row (Auto/Locking only): Skip the lock + Jump straight to a caught freq.
        skip_row = QtWidgets.QHBoxLayout()
        self.w_skip_btn = QtWidgets.QPushButton("Skip lock")
        self.w_skip_btn.setFixedHeight(26)
        self.w_skip_btn.setEnabled(True)         # Auto is the default mode (lock-based)
        self.w_skip_btn.clicked.connect(self._on_skip_lock)
        self.w_jump_btn = QtWidgets.QPushButton("Jump to:")
        self.w_jump_btn.setFixedHeight(26)
        self.w_jump_btn.clicked.connect(self._on_jump_to)
        self.w_jump_combo = QtWidgets.QComboBox()
        self.w_jump_combo.setFixedHeight(26)
        skip_row.addWidget(self.w_skip_btn)
        skip_row.addWidget(self.w_jump_btn)
        skip_row.addWidget(self.w_jump_combo, 1)
        vbox.addLayout(skip_row)

        # Auto-mode tuning: dwell on each lock this long, then auto-skip it if the
        # classifier calls it noise at or above the threshold. Enabled in Auto only.
        auto_row = QtWidgets.QHBoxLayout()
        auto_cap = QtWidgets.QLabel("Auto skip:")
        auto_cap.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_auto_dwell = QtWidgets.QSpinBox()
        self.w_auto_dwell.setRange(100, 60000)
        self.w_auto_dwell.setSingleStep(100)
        self.w_auto_dwell.setSuffix(" ms")
        self.w_auto_dwell.setValue(FP_AUTO_DWELL_MS)
        self.w_auto_dwell.setToolTip("Dwell on each lock this long before judging it")
        self.w_auto_noise = QtWidgets.QSpinBox()
        self.w_auto_noise.setRange(1, 100)
        self.w_auto_noise.setSuffix(" % noise")
        self.w_auto_noise.setValue(FP_AUTO_NOISE_PCT)
        self.w_auto_noise.setToolTip("Skip the lock when noise probability reaches this")
        for w in (self.w_auto_dwell, self.w_auto_noise):
            w.setFixedHeight(26)
            w.setEnabled(True)              # Auto is the default mode
            w.valueChanged.connect(self._on_auto_params)
        auto_row.addWidget(auto_cap)
        auto_row.addWidget(self.w_auto_dwell, 1)
        auto_row.addWidget(self.w_auto_noise, 1)
        vbox.addLayout(auto_row)

        # ══════════════════════════════════════════════════════
        # SDR SECTION
        # ══════════════════════════════════════════════════════
        def hz_combo(presets, current):
            cb = QtWidgets.QComboBox()
            cb.setEditable(True)
            cb.addItems([str(int(p)) for p in presets])
            cb.setCurrentText(str(int(current)))
            return cb

        section("SDR  &  Frequency")
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        for _c in range(3):
            grid.setColumnStretch(_c, 1)

        def cell(r, c, text, widget):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color:#cccccc; font-size:10px;")
            grid.addWidget(lbl, r, c)
            grid.addWidget(widget, r + 1, c)
            return widget

        # row 0/1: sample rate · bandwidth · gain
        self.w_sr   = cell(0, 0, "Sample Rate (Hz)",
                           hz_combo([2e6, 4e6, 5e6, 10e6, 15e6, 20e6, 30e6, 56e6], SAMPLE_RATE))
        self.w_bw   = cell(0, 1, "Bandwidth (Hz)",
                           hz_combo([1e6, 2e6, 4e6, 5e6, 10e6, 20e6, 40e6], RX_BW_HZ))
        self.w_gain = cell(0, 2, "RX Gain (dB)", QtWidgets.QSpinBox())
        self.w_gain.setRange(-3, 71)
        self.w_gain.setValue(GAIN)
        # row 2/3: center freq · total span · overlap
        self.w_center = cell(2, 0, "Center Freq (Hz)", QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_span   = cell(2, 1, "Total Span (Hz)",
                             hz_combo([5e6, 10e6, 20e6, 40e6, 80e6], TOTAL_SPAN_HZ))
        self.w_olap_pct = cell(2, 2, "Overlap (%)", QtWidgets.QComboBox())
        self.w_olap_pct.addItems(["0", "10", "20", "30", "40", "50", "60", "75"])
        self.w_olap_pct.setCurrentText(str(HOP_OVERLAP_PCT))
        # row 4/5: per-hop timing (rarely changed; dwell = sample time per hop,
        # settle = PLL settle wait after retune)
        self.w_dwell = cell(4, 0, "Dwell/hop (ms)", QtWidgets.QSpinBox())
        self.w_dwell.setRange(1, 5000)
        self.w_dwell.setValue(HOP_DWELL_MS)
        self.w_settle = cell(4, 1, "Settle (ms)", QtWidgets.QSpinBox())
        self.w_settle.setRange(0, 500)
        self.w_settle.setValue(HOP_SETTLE_MS)
        vbox.addLayout(grid)

        section("Waterfall Scale  (dBFS)")
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(4)
        min_lbl = QtWidgets.QLabel("min:")
        min_lbl.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_wf_min = hz_combo([-120, -110, -100, -90, -80, -70, -60, -50, -40,
                                  -30, -20, -10, 0], int(WF_SCALE_MIN_DBFS))
        self.w_wf_min.setFixedWidth(64)
        max_lbl = QtWidgets.QLabel("max:")
        max_lbl.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_wf_max = hz_combo([-20, -10, 0, 10, 20, 30, 40], int(WF_SCALE_MAX_DBFS))
        self.w_wf_max.setFixedWidth(64)
        scale_row.addWidget(min_lbl)
        scale_row.addWidget(self.w_wf_min)
        scale_row.addSpacing(10)
        scale_row.addWidget(max_lbl)
        scale_row.addWidget(self.w_wf_max)
        scale_row.addStretch(1)
        vbox.addLayout(scale_row)
        self.w_wf_min.currentTextChanged.connect(self._apply_wf_scale)
        self.w_wf_max.currentTextChanged.connect(self._apply_wf_scale)

        # ══════════════════════════════════════════════════════
        # NARROWBAND MARKERS SECTION  (live red overlays on the zoom plot)
        # ══════════════════════════════════════════════════════
        section("Narrowband Markers")
        marker_row = QtWidgets.QHBoxLayout()
        self.w_show_mid = QtWidgets.QCheckBox("show signal middle")
        self.w_show_mid.toggled.connect(self._on_marker_toggle)
        self.w_show_borders = QtWidgets.QCheckBox("show signal borders")
        self.w_show_borders.toggled.connect(self._on_marker_toggle)
        marker_row.addWidget(self.w_show_mid)
        marker_row.addWidget(self.w_show_borders)
        marker_row.addStretch(1)
        vbox.addLayout(marker_row)


        section("Recording")

        # What to record: a parked DEVICE fingerprint (Focus capture) or band-swept
        # NOISE (Wideband capture). The Record toggle routes to the right acquisition.
        kind_row = QtWidgets.QHBoxLayout()
        self._rec_kind_btns  = {}
        self._rec_kind_group = QtWidgets.QButtonGroup(self)
        self._rec_kind_group.setExclusive(True)
        for key, label in (("device", "Device"),
                           ("noise_band", "Noise (band)"),
                           ("noise_freq", "Noise (freq)")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(24)
            b.clicked.connect(lambda _=False, k=key: self._on_record_kind(k))
            self._rec_kind_group.addButton(b)
            self._rec_kind_btns[key] = b
            kind_row.addWidget(b)
        self._rec_kind_btns["device"].setChecked(True)
        vbox.addLayout(kind_row)

        rec_grid = QtWidgets.QGridLayout()
        rec_grid.setHorizontalSpacing(6)
        rec_grid.setVerticalSpacing(2)
        rec_grid.setColumnStretch(0, 1)
        rec_grid.setColumnStretch(1, 1)

        def rec_cell(r, c, text, widget):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color:#cccccc; font-size:10px;")
            rec_grid.addWidget(lbl, r, c)
            rec_grid.addWidget(widget, r + 1, c)
            return widget

        # Device label + Focus freq apply to a Device recording; Noise ignores them
        # (it sweeps the band into noise/). Session + Max files apply to both.
        self.w_rec_device  = rec_cell(0, 0, "Device label",    QtWidgets.QLineEdit("deviceA"))
        self.w_rec_session = rec_cell(0, 1, "Session",         QtWidgets.QLineEdit("1"))
        self.w_rec_freq    = rec_cell(2, 0, "Focus freq (Hz)", QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_rec_max     = rec_cell(2, 1, "Max files",       QtWidgets.QSpinBox())
        self.w_rec_max.setRange(1, 200000)
        self.w_rec_max.setValue(1000)
        vbox.addLayout(rec_grid)

        rec_btn_row = QtWidgets.QHBoxLayout()
        self.w_rec_btn = QtWidgets.QPushButton("●  Record")
        self.w_rec_btn.setFixedHeight(28)
        self.w_rec_btn.setCheckable(True)
        self.w_rec_btn.clicked.connect(self._on_record_toggle)
        grab_btn = QtWidgets.QPushButton("Lock freq")
        grab_btn.setFixedWidth(80)
        grab_btn.setFixedHeight(28)
        grab_btn.clicked.connect(self._on_grab_lock)
        rec_btn_row.addWidget(self.w_rec_btn)
        rec_btn_row.addWidget(grab_btn)
        vbox.addLayout(rec_btn_row)
        self._update_record_btn_style(False)
        self._on_record_kind("device")     # sync field enable-state + hint text

        vbox.addSpacing(8)
        apply_btn = QtWidgets.QPushButton("⟳  Apply Settings")
        apply_btn.setFixedHeight(34)
        apply_btn.clicked.connect(self._apply_settings)
        vbox.addWidget(apply_btn)

        section("Status")
        _ss = "color: #cccccc; font-size: 11px;"   # one unified style for all rows
        self.model_info_lbl = QtWidgets.QLabel("")
        self.model_info_lbl.setWordWrap(True)
        self.model_info_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.model_info_lbl)
        self.infer_stat_lbl = QtWidgets.QLabel("Inference: —")
        self.infer_stat_lbl.setWordWrap(True)
        self.infer_stat_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.infer_stat_lbl)
        self.mode_lbl = QtWidgets.QLabel("Mode: —")
        self.mode_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.mode_lbl)
        self.caught_lbl = QtWidgets.QLabel("Caught: —")
        self.caught_lbl.setWordWrap(True)
        self.caught_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.caught_lbl)
        self.hop_info_lbl = QtWidgets.QLabel("")
        self.hop_info_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.hop_info_lbl)
        self._update_hop_info_label()
        self.status_lbl = QtWidgets.QLabel("Starting…")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(_ss)
        self.file_lbl = QtWidgets.QLabel("Files on disk: 0")
        self.file_lbl.setStyleSheet(_ss)
        self.prog_bar = QtWidgets.QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setFixedHeight(8)
        vbox.addWidget(self.status_lbl)
        vbox.addWidget(self.file_lbl)
        vbox.addWidget(self.prog_bar)
        vbox.addStretch()

        # Drop the tiny up/down arrows on every spin box — type or scroll instead.
        for _sb in panel.findChildren(QtWidgets.QAbstractSpinBox):
            _sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(340)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel_scroll.setWidget(panel)
        root.addWidget(panel_scroll, stretch=0)

    # ── ML: badge styling ─────────────────────────────────────────────────────

    _BADGE_STYLES = {
        "none"  : ("background: #3a3a3a; color: #ffffff;"
                   " border: 1px solid #555555; border-radius: 6px;"),
        "device": ("background: #14507a; color: #d6ecff;"
                   " border: 1px solid #2a86c8; border-radius: 6px;"),
        "other" : ("background: #4a4a4a; color: #ffffff;"
                   " border: 1px solid #777777; border-radius: 6px;"),
        "error" : ("background: #7a4c00; color: #ffe0a0;"
                   " border: 1px solid #cc8800; border-radius: 6px;"),
    }

    def _set_badge_style(self, kind: str):
        self.det_badge.setStyleSheet(
            self._BADGE_STYLES.get(kind, self._BADGE_STYLES["other"]))

    # ── ML: confidence bar grid rebuild ──────────────────────────────────────

    def _rebuild_conf_bars(self, classes: list):
        layout = self._conf_container.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._conf_bars.clear()
        self._conf_labels.clear()

        for row_i, cls in enumerate(classes):
            name_lbl = QtWidgets.QLabel(cls)
            name_lbl.setFixedWidth(60)
            name_lbl.setStyleSheet("color: #ffffff; font-size: 11px;")

            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(14)
            bar.setStyleSheet(
                "QProgressBar::chunk { background: #336633; }"
                "QProgressBar { border: 1px solid #666666; border-radius: 3px;"
                " background: #1e1e1e; }")

            pct_lbl = QtWidgets.QLabel("0%")
            pct_lbl.setFixedWidth(38)
            pct_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            pct_lbl.setStyleSheet("color: #ffffff; font-size: 11px;")

            layout.addWidget(name_lbl, row_i, 0)
            layout.addWidget(bar,      row_i, 1)
            layout.addWidget(pct_lbl,  row_i, 2)
            self._conf_bars[cls]   = bar
            self._conf_labels[cls] = pct_lbl

    # ── ML: model load ────────────────────────────────────────────────────────

    def _browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model (.pt)", _SCRIPT_DIR,
            "PyTorch model (*.pt);;All files (*)"
        )
        if path:
            self.w_model_path.setText(path)

    def _on_load_model_btn(self):
        path = self.w_model_path.text().strip()
        if not path:
            path = DEFAULT_MODEL_PATH
            self.w_model_path.setText(path)
        self._load_model(path)

    def _load_model(self, path: str):
        if not _TORCH_OK:
            self.model_info_lbl.setText("⚠ PyTorch not installed.")
            self.det_badge.setText("TORCH MISSING")
            self._set_badge_style("error")
            return
        if not os.path.exists(path):
            self.model_info_lbl.setText(f"⚠ File not found:\n{path}")
            self.det_badge.setText("MODEL NOT FOUND")
            self._set_badge_style("error")
            return
        try:
            from fp_spectrogram import FingerprintModel
            engine = FingerprintModel(path)
            self._engine = engine
            if hasattr(self, "worker"):
                self.worker.engine = engine
            self._rebuild_conf_bars(engine.classes)
            self.det_badge.setText("MODEL LOADED — SCANNING")
            self._set_badge_style("none")
            short = os.path.basename(path)
            warn = ("" if any(c.lower() == "noise" for c in engine.classes) else
                    "<br>⚠ No 'noise' class — Auto mode's auto-skip will never trigger")
            self.model_info_lbl.setText(
                f"{_hl(short)} loaded successfully<br>"
                f"Model classes: {_hl(', '.join(engine.classes))}{warn}"
            )
        except Exception as e:
            self._engine = None
            self.model_info_lbl.setText(f"⚠ Load error:\n{e}")
            self.det_badge.setText("LOAD ERROR")
            self._set_badge_style("error")

    def _on_ml_toggle(self, checked: bool):
        """Flip the worker's live inference gate. Reads through self.cfg (shared with
        the worker), so it takes effect on the next loop without a restart."""
        self.cfg["ml_enabled"] = checked
        self.w_ml_toggle.setText(f"ML Inference: {'ON' if checked else 'OFF'}")
        if not checked:
            # Park the readouts so a stale prediction doesn't look live.
            self.det_badge.setText("ML OFF")
            self._set_badge_style("none")
            for cls, bar in self._conf_bars.items():
                bar.setValue(0)
                self._conf_labels[cls].setText("0%")
            self.infer_stat_lbl.setText("Inference: off")
        elif self._engine is not None:
            self.det_badge.setText("MODEL LOADED — SCANNING")

    # ── ML: result + zoom handlers (inference runs in the worker) ──────────────

    def _on_fingerprint_ready(self, result):
        label = result["label"]
        conf  = result["confidence"]
        probs = result["probs"]
        if result.get("unknown"):
            self.det_badge.setText(f"unknown ({conf:.0%})")
            self._set_badge_style("other")
        else:
            self.det_badge.setText(f"{label} ({conf:.0%})")
            self._set_badge_style("device")
        for cls, bar in self._conf_bars.items():
            p = probs.get(cls, 0.0)
            bar.setValue(int(p * 100))
            self._conf_labels[cls].setText(f"{p:.0%}")
        self.infer_stat_lbl.setText(f"Inference: {_hl(f'{label} @ {conf:.1%}')}")

    def _on_zoom_ready(self, freqs, psd, held_freq):
        self.zoom_curve.setData(freqs, psd)
        sr = self.cfg["sample_rate"]
        # Recenter the view only when the hold jumps to a new frequency, so manual
        # pan/zoom on the zoom plots isn't overridden every frame.
        if (getattr(self, "_zoom_center", None) is None
                or abs(held_freq - self._zoom_center) > sr / 4):
            self.p_zoom.setXRange(held_freq - sr / 2, held_freq + sr / 2, padding=0)
            self._zoom_center = held_freq
            self.zoom_wf_data[:] = -200.0      # wipe stale history on a NEW lock
        self.p_zoom.setTitle(f"Narrowband Spectrum ({held_freq / 1e6:.3f} MHz)")
        # scroll the held-signal spectrum into the zoom waterfall
        self.zoom_wf_data = np.roll(self.zoom_wf_data, -1, axis=1)
        self.zoom_wf_data[:, -1] = psd
        self.zoom_wf_img.setImage(self.zoom_wf_data, autoLevels=False)
        self.zoom_wf_img.setRect(QtCore.QRectF(held_freq - sr / 2, 0, sr, WATERFALL_ROWS))
        self._last_zoom = (freqs, psd)
        self._update_signal_markers(freqs, psd)

    # ── narrowband signal markers ─────────────────────────────────────────────

    def _on_marker_toggle(self, _checked=False):
        """A marker checkbox flipped — hide what's now off, then refresh against
        the last spectrum so turning one on shows it immediately (even in SCAN,
        where no fresh zoom frame is arriving)."""
        if not self.w_show_mid.isChecked():
            self.mid_line.setVisible(False)
        if not self.w_show_borders.isChecked():
            for ln in self.edge_lines:
                ln.setVisible(False)
        if self._last_zoom is not None:
            self._update_signal_markers(*self._last_zoom)

    def _update_signal_markers(self, freqs, psd):
        """Place the red middle/edge lines from the live PSD (see signal_extent).
        Hidden checkboxes draw nothing; a signal below the floor hides them too."""
        show_mid  = self.w_show_mid.isChecked()
        show_edge = self.w_show_borders.isChecked()
        if not (show_mid or show_edge):
            return
        extent = signal_extent(freqs, psd)
        if extent is None:                         # nothing above the noise floor
            self.mid_line.setVisible(False)
            for ln in self.edge_lines:
                ln.setVisible(False)
            return
        f_left, f_center, f_right = extent
        self.mid_line.setPos(f_center)
        self.mid_line.setVisible(show_mid)
        self.edge_lines[0].setPos(f_left)
        self.edge_lines[1].setPos(f_right)
        for ln in self.edge_lines:
            ln.setVisible(show_edge)

    def _on_mode_changed(self, mode, held_freq):
        if held_freq:
            self._last_held_freq = held_freq
            self.mode_lbl.setText(
                f"Mode: {_hl(mode)}  @  {_hl(f'{held_freq/1e6:.3f} MHz')}")
        else:
            self.mode_lbl.setText(f"Mode: {_hl(mode)}")

    def _on_caught_changed(self, freqs):
        if freqs:
            vals = ", ".join(f"{f / 1e6:.2f}" for f in freqs)
            self.caught_lbl.setText(f"Caught: {_hl(vals + ' MHz')}")
        else:
            self.caught_lbl.setText("Caught: —")
        # mirror the caught list into the Jump-to dropdown
        self.w_jump_combo.clear()
        for f in freqs:
            self.w_jump_combo.addItem(f"{f / 1e6:.3f} MHz", int(f))

    # ── waterfall helpers ─────────────────────────────────────────────────────

    def _update_waterfall_rect(self):
        span = self.f_global_max - self.f_global_min
        self.img.setRect(QtCore.QRectF(self.f_global_min, 0, span, WATERFALL_ROWS))
        # Pin p2's view to the data so the image can't render wider/taller than
        # the axis. x is XLinked to p1; y is fixed to the sweep-row count.
        self.p2.setYRange(0, WATERFALL_ROWS, padding=0)

    def _apply_wf_scale(self):
        try:
            v_min = float(self.w_wf_min.currentText())
            v_max = float(self.w_wf_max.currentText())
        except (ValueError, AttributeError):
            return                       # mid-edit / non-numeric text — ignore
        if v_max <= v_min:
            return
        self.img.setLevels([v_min, v_max])
        if hasattr(self, "zoom_wf_img"):
            self.zoom_wf_img.setLevels([v_min, v_max])

    def _rebuild_hop_lines(self):
        for ln in self.hop_lines:
            self.p1.removeItem(ln)
        self.hop_lines.clear()
        for i in range(self.n_hops + 1):
            x  = self.f_global_min + i * self._slot_hz
            ln = pg.InfiniteLine(
                pos=x, angle=90,
                pen=pg.mkPen(color=(70, 70, 70),
                             style=QtCore.Qt.DashLine, width=1))
            self.p1.addItem(ln)
            self.hop_lines.append(ln)

    def _update_hop_info_label(self):
        sweep_ms = self.n_hops * (self.cfg["dwell_ms"] + self.cfg["settle_ms"])
        span_mhz = self.cfg["total_span"] / 1e6
        self.hop_info_lbl.setText(
            f"{_hl(self.n_hops)} hops · {_hl(f'{span_mhz:.1f} MHz')} total · "
            f"~{_hl(f'{sweep_ms} ms')}/sweep")

    # ── recording controls ────────────────────────────────────────────────────

    def _switch_mode(self, key: str):
        """Apply op_mode `key` and restart the worker. Leaves recording state alone."""
        self.cfg["op_mode"] = key
        is_lock = key in ("locking", "auto")    # both run the lock state-machine
        self.w_skip_btn.setEnabled(is_lock)
        self.w_jump_btn.setEnabled(is_lock)
        self.w_jump_combo.setEnabled(is_lock)
        is_auto = (key == "auto")
        self.w_auto_dwell.setEnabled(is_auto)
        self.w_auto_noise.setEnabled(is_auto)
        if key in self._mode_btns:
            self._mode_btns[key].setChecked(True)
        self.worker.stop()
        self._start_worker()

    def _on_mode_btn(self, key: str):
        if self.sdr is None:
            self.status_lbl.setText("SDR not connected.")
            return
        if self.cfg.get("op_mode") == key:
            return
        # Switching the view mode by hand stops recording (what gets saved differs).
        self.cfg["record"] = False
        self.w_rec_btn.setChecked(False)
        self._update_record_btn_style(False)
        self._sync_record_cfg()
        self._switch_mode(key)

    def _on_skip_lock(self):
        self.cfg["skip_lock"] = True        # Locking/Auto mode reads this live

    def _on_jump_to(self):
        f = self.w_jump_combo.currentData()
        if f:
            self.cfg["jump_to"] = int(f)    # Locking/Auto mode reads this live

    def _on_auto_params(self, _v=0):
        self.cfg["auto_dwell_ms"]  = self.w_auto_dwell.value()   # Auto reads these live
        self.cfg["auto_noise_pct"] = self.w_auto_noise.value()

    def _on_record_toggle(self, checked: bool):
        if self.sdr is None:                 # no worker exists — nothing to record
            self.w_rec_btn.setChecked(False)
            self.status_lbl.setText("SDR not connected.")
            return
        self._sync_record_cfg()
        if not checked:
            self.cfg["record"] = False
            self._update_record_btn_style(False)
            return
        # Route to the acquisition that matches what we're recording:
        #   device -> parked Focus capture; noise -> band-swept Wideband capture.
        kind   = self.cfg.get("record_kind", "device")
        target = "wideband" if kind == "noise_band" else "focus"
        self.cfg["record"] = True             # read live by the worker loops
        self.w_rec_btn.setChecked(True)
        self._update_record_btn_style(True)
        if self.cfg.get("op_mode") != target:
            self._switch_mode(target)

    def _on_record_kind(self, kind: str):
        """Choose what Record saves: a Device fingerprint, band-swept Noise, or
        narrowband Noise parked at the Focus freq (a frequency-matched negative)."""
        self.cfg["record_kind"] = kind
        parked = kind in ("device", "noise_freq")     # parked capture vs band sweep
        self.w_rec_device.setEnabled(kind == "device")
        self.w_rec_freq.setEnabled(parked)
        if kind in self._rec_kind_btns:
            self._rec_kind_btns[kind].setChecked(True)
        # If already recording, re-route into the matching acquisition mode now.
        if self.cfg.get("record") and hasattr(self, "worker"):
            self._on_record_toggle(True)

    def _sync_record_cfg(self):
        self.cfg["record_device"]    = self.w_rec_device.text().strip() or "deviceA"
        self.cfg["record_session"]   = self.w_rec_session.text().strip() or "1"
        self.cfg["record_max_files"] = self.w_rec_max.value()
        try:
            self.cfg["focus_freq"] = int(float(self.w_rec_freq.text()))
        except ValueError:
            self.cfg["focus_freq"] = self.cfg["center_freq"]

    def _update_record_btn_style(self, recording: bool):
        if recording:
            self.w_rec_btn.setText("■  Recording")
            self.w_rec_btn.setStyleSheet(
                "QPushButton { color: #ffffff; background-color: #882200;"
                " border: 1px solid #cc4422; border-radius: 3px; padding: 3px 6px; }"
                "QPushButton:hover { background-color: #aa2200; }"
            )
        else:
            self.w_rec_btn.setText("●  Record")
            self.w_rec_btn.setStyleSheet("")  # inherit panel default

    def _on_grab_lock(self):
        f = getattr(self, "_last_held_freq", None)
        if f:
            self.w_rec_freq.setText(str(int(f)))
            self.cfg["focus_freq"] = int(f)

    # ── settings apply ────────────────────────────────────────────────────────

    def _apply_settings(self):
        if self.sdr is None:
            self.status_lbl.setText("SDR not connected — cannot apply settings.")
            return
        self.worker.stop()
        try:
            self.cfg["sample_rate"]        = int(float(self.w_sr.currentText()))
            self.cfg["rx_bw"]              = int(float(self.w_bw.currentText()))
            self.cfg["gain"]               = self.w_gain.value()
            self.cfg["center_freq"]        = int(float(self.w_center.text()))
            self.cfg["total_span"]         = int(float(self.w_span.currentText()))
            self.cfg["dwell_ms"]           = self.w_dwell.value()
            self.cfg["settle_ms"]          = self.w_settle.value()
            self.cfg["overlap_pct"]        = int(self.w_olap_pct.currentText())
            self._sync_record_cfg()        # apply Focus/Narrowband freq + device/session
            self.cfg["record"]             = False     # stop recording on settings change
            self.w_rec_btn.setChecked(False)
            self._update_record_btn_style(False)
            self._recompute_hops()
            self._push_sdr_settings()
            self.wf_data = self._make_waterfall_buf()
            self.img.setImage(self.wf_data, autoLevels=False)
            self._update_waterfall_rect()
            self._rebuild_hop_lines()
            self._update_hop_info_label()
            self.p1.setXRange(self.f_global_min, self.f_global_max, padding=0)
            self.p1.setYRange(WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS, padding=0)
            self._apply_wf_scale()
            self.status_lbl.setText(
                f"Applied. {self.n_hops} hops · "
                f"{self.cfg['center_freq'] / 1e6:.1f} MHz center")
        except Exception as e:
            self.status_lbl.setText(f"Apply error: {e}")
        self._start_worker()

    def _start_worker(self):
        self._zoom_center = None        # let the zoom view recenter on the next hold
        self.cfg["skip_lock"] = False
        self.caught_lbl.setText("Caught: —")     # fresh memory each (re)start
        self.w_jump_combo.clear()
        self.worker = SweepWorker(self.sdr, self.cfg, engine=self._engine)
        self.worker.sweep_ready.connect(self._on_sweep_ready)
        self.worker.zoom_ready.connect(self._on_zoom_ready)
        self.worker.fingerprint_ready.connect(self._on_fingerprint_ready)
        self.worker.mode_changed.connect(self._on_mode_changed)
        self.worker.caught_changed.connect(self._on_caught_changed)
        self.worker.hop_progress.connect(self._on_hop_progress)
        self.worker.status_msg.connect(self._on_status)
        self.worker.files_changed.connect(self._on_files_changed)
        self.worker.start()

    # ── signal handlers ───────────────────────────────────────────────────────

    def _on_sweep_ready(self, composite: np.ndarray, hop_bufs: dict):
        if len(composite) != self.total_bins:
            return
        freqs = np.linspace(self.f_global_min, self.f_global_max, self.total_bins)
        self.curve.setData(freqs, composite)

        if (self.wf_data.shape[0] != self.total_bins or
                self.wf_data.shape[1] != WATERFALL_ROWS):
            self.wf_data = self._make_waterfall_buf()

        self.wf_data = np.roll(self.wf_data, -1, axis=1)
        self.wf_data[:, -1] = composite
        self.img.setImage(self.wf_data, autoLevels=False)
        self._update_waterfall_rect()

    def _on_hop_progress(self, hop_idx, total):
        self.prog_bar.setValue(int(100 * hop_idx / max(total, 1)))

    def _on_status(self, msg):
        self.status_lbl.setText(_hl_values(msg))
        self.prog_bar.setValue(100)

    def _on_files_changed(self, count):
        self.file_lbl.setText(f"Files on disk: {_hl(count)}")

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
        if self.sdr is not None:
            del self.sdr
        event.accept()


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    # High-DPI / multi-monitor: must be set BEFORE the QApplication exists.
    # PassThrough rounding keeps fractional scale factors (125%, 150%) exact so
    # axis-label widths and the waterfall ImageItem stay pixel-aligned when the
    # window is moved to a monitor with different scaling.
    try:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except AttributeError:
        pass  # Qt < 5.14
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window,          QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.WindowText,      QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base,            QtGui.QColor(20, 20, 20))
    palette.setColor(QtGui.QPalette.AlternateBase,   QtGui.QColor(40, 40, 40))
    palette.setColor(QtGui.QPalette.Button,          QtGui.QColor(50, 50, 50))
    palette.setColor(QtGui.QPalette.ButtonText,      QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Text,            QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(120, 120, 120))
    palette.setColor(QtGui.QPalette.Highlight,       QtGui.QColor(0, 100, 180))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
    app.setPalette(palette)
    window = PlutoApp()
    window.show()
    app.exec_()