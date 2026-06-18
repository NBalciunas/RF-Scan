"""
terminal-ml-v2.py  –  PlutoSDR Monitor + RF-Fingerprint detection (cascade).

v2 reworks the v1 drone/noise monitor into a fingerprinting tool:
  * the drone/noise CNN is gone; ML is now the spectrogram FingerprintModel
    (fp_spectrogram.py), loaded the same way but used differently;
  * the worker runs a SCAN -> HOLD state machine: hop+scan as before, and when a
    peak crosses FP_PEAK_THRESH_DB, park the LO, zoom in, and classify the held
    signal; nudge the LO to recentre, and resume scanning if the fingerprint is
    lost or unknown;
  * recording is raw fixed-LO IQ: stationary per-device fingerprints, or a
    band-swept "noise" class, both written to fingerprint_data/<device>/session_*/;
  * a third "zoom" plot shows the currently-held narrowband signal.
"""

import os
import sys
import time
import json
import math
import re
import collections
import xml.etree.ElementTree as ET

import numpy as np
import adi
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

torch     = None
nn        = None
_TORCH_OK = False
try:
    import torch as _torch
    import torch.nn as _nn
    torch     = _torch
    nn        = _nn
    _TORCH_OK = True
except ImportError:
    pass

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

DRONE_NAME      = "Drone"
SERIAL_NUM      = "00001"
REFERENCE_SNR   = 40

WIDEBAND_RECORD_ENABLED = False
WIDEBAND_SECS           = 2.0
WIDEBAND_MAX_FILES      = 500

WATERFALL_JPG_ENABLED   = False
WATERFALL_JPG_INTERVAL  = 5.0
WATERFALL_JPG_DIR       = "./output/waterfall_snaps"
WATERFALL_JPG_QUALITY   = 90

FFT_BINS           = 1024
WATERFALL_ROWS     = 200
WF_SCALE_MIN_DBFS  = -10.0
WF_SCALE_MAX_DBFS  = 10.0

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "fingerprint_spec_model.pt")

# ── Fingerprint SCAN -> HOLD state-machine params (tune against live signals) ──
FP_PEAK_THRESH_DB    = 10.0      # peak-above-floor (dB) on a sweep that triggers a hold
FP_UNKNOWN_THRESH    = 0.80      # below this top-class prob a hold is "no fingerprint"
FP_MAX_MISSES        = 3         # consecutive unknown holds before resuming the scan
FP_NUDGE_STEP_HZ     = 500_000   # max LO nudge per hold to recentre the signal
FP_NUDGE_DEADZONE_HZ = 200_000   # don't nudge if the signal is already this centred
FP_HOLD_SETTLE_MS    = 20        # LO settle before grabbing a held capture
FP_HOLD_RESCAN_S     = 4.0       # while holding, do a full band sweep this often (keeps scanning)
FP_HOLD_MAX_S        = 5.0       # focus a caught signal this long, then memorise it and move on
FP_MEMORY_TTL_S      = 30.0      # remembered catches are skipped for this long, then revisitable
FP_MEMORY_GUARD_HZ   = 3_000_000 # peaks within this of a remembered freq count as the same signal

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


def _xml_path(iq_path: str) -> str:
    base, _ = os.path.splitext(iq_path)
    return base + ".xml"


def write_xml(xml_path, iq_filename, sample_count, center_freq, sample_rate,
              extra_tags=None):
    root = ET.Element("SignalHoundIQFile")
    ET.SubElement(root, "DeviceType").text        = "PlutoSDR"
    ET.SubElement(root, "Drone").text             = DRONE_NAME
    ET.SubElement(root, "SerialNumber").text      = SERIAL_NUM
    ET.SubElement(root, "DataType").text          = "Complex Float"
    ET.SubElement(root, "ReferenceSNRLevel").text = str(REFERENCE_SNR)
    ET.SubElement(root, "CenterFrequency").text   = str(center_freq)
    ET.SubElement(root, "SampleRate").text        = str(sample_rate)
    ET.SubElement(root, "IFBandwidth").text       = str(sample_rate)
    ET.SubElement(root, "ScaleFactor").text       = "1.0"
    ET.SubElement(root, "IQFileName").text        = iq_filename
    ET.SubElement(root, "SampleCount").text       = str(sample_count)
    if extra_tags:
        for k, v in extra_tags.items():
            ET.SubElement(root, k).text = str(v)
    ET.ElementTree(root).write(xml_path)


# ==========================================
# WIDEBAND STITCHER
# ==========================================

class WidebandStitcher:
    def __init__(self, hop_freqs, hop_bw, total_span, center_freq,
                 fft_bins=FFT_BINS):
        self.hop_freqs      = hop_freqs
        self.hop_bw         = hop_bw
        self.total_span     = total_span
        self.center_freq    = center_freq
        self.fft_bins       = fft_bins
        self.n_hops         = len(hop_freqs)
        self.composite_bins = self.n_hops * fft_bins
        self.synthetic_rate = self.n_hops * hop_bw
        self.window         = np.hanning(fft_bins).astype(np.complex64)

        f_min  = center_freq - total_span / 2
        bin_hz = hop_bw / fft_bins
        self._hop_slices = []
        for freq in hop_freqs:
            hop_f_min = freq - hop_bw / 2
            start_bin = int(round((hop_f_min - f_min) / bin_hz))
            end_bin   = start_bin + fft_bins
            self._hop_slices.append((
                max(start_bin, 0),
                min(end_bin, self.composite_bins)
            ))

    def stitch(self, hop_iq_buffers):
        composite_spectrum = np.zeros(self.composite_bins, dtype=np.complex128)
        weight_map         = np.zeros(self.composite_bins, dtype=np.float64)

        for i in range(self.n_hops):
            buf = hop_iq_buffers.get(i)
            if buf is None or len(buf) < self.fft_bins:
                continue
            chunk    = buf[-self.fft_bins:].astype(np.complex128)
            spectrum = np.fft.fftshift(np.fft.fft(chunk * self.window))
            s_start, s_end = self._hop_slices[i]
            bins_to_place  = s_end - s_start
            if bins_to_place <= 0:
                continue
            edge_taper = np.ones(bins_to_place, dtype=np.float64)
            taper_len  = min(self.fft_bins // 8, bins_to_place // 2)
            if taper_len > 0:
                ramp = np.hanning(taper_len * 2)[:taper_len]
                edge_taper[:taper_len]  = ramp
                edge_taper[-taper_len:] = ramp[::-1]
            composite_spectrum[s_start:s_end] += spectrum[:bins_to_place] * edge_taper
            weight_map[s_start:s_end]         += edge_taper

        nonzero = weight_map > 0
        composite_spectrum[nonzero] /= weight_map[nonzero]
        if not np.any(nonzero):
            return None
        return np.fft.ifft(np.fft.ifftshift(composite_spectrum)).astype(np.complex64)


# ==========================================
# ML  (fingerprinting)
# ==========================================
# The v1 drone/noise CNN (SpectralCNNSlim + InferenceEngine) is removed.  v2 loads
# the spectrogram FingerprintModel from fp_spectrogram.py — imported lazily in
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
        self._misses    = 0
        self._last_composite = None        # last full sweep, kept alive during HOLD
        self._last_scan_t    = 0.0         # time of last full sweep (periodic rescan)
        self._caught         = []          # [(freq_hz, t_caught)] — memory of catches
        self._hold_t0        = 0.0         # when the current HOLD started

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
        c, s = self.cfg["center_freq"], self.cfg["total_span"]
        return c - s // 2, c + s // 2

    def _sweep_once(self):
        """One wideband hop sweep -> (composite, hop_bufs). Mirrors v1 SCAN."""
        hop_freqs  = self.cfg["hop_freqs"]
        n_hops     = len(hop_freqs)
        composite  = np.full(n_hops * FFT_BINS, -100.0, dtype=np.float32)
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
            chunk = raw[:FFT_BINS]
            if len(chunk) < FFT_BINS:
                chunk = np.pad(chunk, (0, FFT_BINS - len(chunk)))
            psd = 20.0 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(chunk * self._BLACKMAN)))
                / FFT_BINS + 1e-10)
            composite[i * FFT_BINS:(i + 1) * FFT_BINS] = psd.astype(np.float32)
        return composite, hop_bufs

    def _narrowband_psd(self, iq, center):
        """Peak-hold dBFS spectrum of a held capture, mapped to absolute Hz.

        Peak-hold across the whole buffer (not just the first window) so a burst
        landing anywhere in the ~50 ms capture is shown — otherwise the display
        flickers and mostly misses the bursty emission."""
        n = len(iq)
        if n < FFT_BINS:
            iq = np.pad(iq, (0, FFT_BINS - n)); n = FFT_BINS
        # Contiguous, gap-free windows (a sparse step lets a burst fall between
        # windows and vanish).  Batched FFT, peak-hold across windows.
        nwin = min(1024, n // FFT_BINS)
        seg  = iq[:nwin * FFT_BINS].reshape(nwin, FFT_BINS) * self._BLACKMAN
        mag  = np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) / FFT_BINS
        psd  = 20.0 * np.log10(mag.max(axis=0) + 1e-10)     # peak-hold
        sr = self.cfg["sample_rate"]
        freqs = np.linspace(center - sr / 2, center + sr / 2, FFT_BINS)
        return freqs, psd.astype(np.float32)

    def _center_offset_hz(self, iq):
        """Hz offset of the strongest bin from the capture centre (for nudging)."""
        chunk = iq[:FFT_BINS]
        if len(chunk) < FFT_BINS:
            chunk = np.pad(chunk, (0, FFT_BINS - len(chunk)))
        mag  = np.abs(np.fft.fftshift(np.fft.fft(chunk * self._BLACKMAN)))
        peak = int(np.argmax(mag))
        return (peak - FFT_BINS / 2) / FFT_BINS * self.cfg["sample_rate"]

    def _composite_with_hold(self, held_freq, psd):
        """Overlay the live held-band spectrum onto the last full sweep, so the
        wideband view keeps showing the whole band (with the locked spot live)
        while parked in HOLD."""
        base = self._last_composite
        if base is None:                       # no prior sweep — synthesise a floor
            base = np.full(len(self.cfg["hop_freqs"]) * FFT_BINS, -100.0, dtype=np.float32)
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

    def _detect_peak(self, composite):
        """Strongest peak-above-floor on the sweep -> (freq_hz, peak_db)."""
        med = float(np.median(composite))
        idx = int(np.argmax(composite))
        f_min, f_max = self._band_edges()
        f = f_min + (idx / len(composite)) * (f_max - f_min)
        return f, float(composite[idx]) - med

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

    # ── top-level dispatch ────────────────────────────────────────────────────

    def run(self):
        op = self.cfg.get("op_mode", "normal")
        if op == "wideband":
            self._run_wideband()
        elif op == "focus":
            self._run_focus()
        else:
            self._run_normal()

    def _run_normal(self):
        """Auto loop: SCAN until a new peak crosses threshold, then HOLD/focus it,
        classify + nudge, memorise the frequency, and move on to the next signal."""
        mode = "SCAN"
        self.mode_changed.emit("SCAN", 0.0)
        while not self._stop:
            if mode == "SCAN":
                t0 = time.perf_counter()
                composite, hop_bufs = self._sweep_once()
                if composite is None:
                    return
                self._last_composite = composite
                self._last_scan_t = time.time()
                self.sweep_ready.emit(composite, hop_bufs)
                el = (time.perf_counter() - t0) * 1000
                self.status_msg.emit(
                    f"Scan: {el:.0f} ms  |  {len(self.cfg['hop_freqs'])} hops")
                self._prune_memory()
                # Lock onto the strongest peak that ISN'T already in memory, so the
                # loop moves on to new signals instead of re-catching the same one.
                f, peak_db = self._detect_new_peak(composite)
                if f is not None and peak_db >= float(
                        self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB)):
                    self._held_freq, self._misses, mode = f, 0, "HOLD"
                    self._hold_t0 = time.time()
                    self.mode_changed.emit("HOLD", f)
                    self.status_msg.emit(
                        f"Peak +{peak_db:.0f} dB @ {f/1e6:.3f} MHz — holding")
            else:  # HOLD
                if self._stop:
                    break
                # Periodic full sweep so it keeps scanning at times instead of
                # parking forever; refreshes the whole wideband, then resumes hold.
                if time.time() - self._last_scan_t >= float(
                        self.cfg.get("fp_hold_rescan_s", FP_HOLD_RESCAN_S)):
                    composite, hop_bufs = self._sweep_once()
                    if composite is None:
                        return
                    self._last_composite = composite
                    self._last_scan_t = time.time()
                    self.sweep_ready.emit(composite, hop_bufs)
                    self.status_msg.emit(
                        f"Hold @ {self._held_freq / 1e6:.3f} MHz — periodic rescan")
                try:
                    self.sdr.rx_lo = int(self._held_freq)
                except Exception as e:
                    self.status_msg.emit(f"Hold tune error: {e}")
                    mode = "SCAN"; self.mode_changed.emit("SCAN", 0.0); continue
                time.sleep(self.cfg.get("fp_hold_settle_ms", FP_HOLD_SETTLE_MS) / 1000.0)
                try:
                    iq = np.array(self.sdr.rx(), dtype=np.complex64)
                except Exception as e:
                    self.status_msg.emit(f"Hold RX error: {e}"); continue
                if iq is None or len(iq) == 0:
                    continue

                freqs, psd = self._narrowband_psd(iq, self._held_freq)
                self.zoom_ready.emit(freqs, psd, float(self._held_freq))
                # keep the wideband alive: overlay the held band on the last sweep
                comp = self._composite_with_hold(self._held_freq, psd)
                if comp is not None:
                    self.sweep_ready.emit(comp, {})

                # Classify if a model is loaded; with no model just hold on the
                # signal so the zoom still engages and you can watch park + nudge.
                thresh = float(self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB))
                if self.engine is not None:
                    try:
                        result = self.engine.classify_iq(iq)
                    except Exception as e:
                        self.status_msg.emit(f"Inference error: {e}"); continue
                    self.fingerprint_ready.emit(result)
                    present = not result.get("unknown", True)
                else:
                    present = (float(psd.max()) - float(np.median(psd))) >= thresh

                # recentre nudge — move the LO toward the strongest bin
                off  = self._center_offset_hz(iq)
                dead = float(self.cfg.get("fp_nudge_deadzone_hz", FP_NUDGE_DEADZONE_HZ))
                if abs(off) > dead:
                    step = float(self.cfg.get("fp_nudge_step_hz", FP_NUDGE_STEP_HZ))
                    self._held_freq += max(-step, min(step, off))
                    self.mode_changed.emit("HOLD", float(self._held_freq))

                # Leave HOLD when the signal is gone OR we've focused long enough;
                # either way memorise the frequency so the scan moves to a new one.
                leave, reason = False, ""
                if present:
                    self._misses = 0
                else:
                    self._misses += 1
                    if self._misses >= int(self.cfg.get("fp_max_misses", FP_MAX_MISSES)):
                        leave, reason = True, "signal gone"
                if time.time() - self._hold_t0 >= float(
                        self.cfg.get("fp_hold_max_s", FP_HOLD_MAX_S)):
                    leave, reason = True, "focused long enough"
                if leave:
                    self._remember(self._held_freq)
                    mode, self._held_freq = "SCAN", None
                    self.mode_changed.emit("SCAN", 0.0)
                    self.status_msg.emit(f"{reason} — memorised, resuming scan")

    def _run_wideband(self):
        """Continuous full-band scan, no focus/hold — just the wideband view.
        If 'record' is on, save each hop's raw IQ as the noise class."""
        self.mode_changed.emit("WIDEBAND", 0.0)
        while not self._stop:
            t0 = time.perf_counter()
            composite, hop_bufs = self._sweep_once()
            if composite is None:
                return
            self._last_composite = composite
            self.sweep_ready.emit(composite, hop_bufs)
            rec = bool(self.cfg.get("record"))
            if rec:
                session = self.cfg.get("record_session", "1")
                for i, raw in hop_bufs.items():
                    if len(raw):
                        self._save_iq(raw, "noise", session,
                                      int(self.cfg["hop_freqs"][i]))
            el  = (time.perf_counter() - t0) * 1000
            tag = f"  |  REC noise: {len(self._fq)} files" if rec else ""
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
        while not self._stop:
            nf = int(self.cfg.get("focus_freq", freq))      # live retune if grabbed
            if nf != freq:
                freq = nf
                try:
                    self.sdr.rx_lo = freq
                except Exception as e:
                    self.status_msg.emit(f"Focus tune error: {e}"); continue
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
            if self.engine is not None:
                try:
                    self.fingerprint_ready.emit(self.engine.classify_iq(iq))
                except Exception as e:
                    self.status_msg.emit(f"Inference error: {e}")
            if self.cfg.get("record"):
                self._save_iq(iq, self.cfg.get("record_device", "deviceA"),
                              self.cfg.get("record_session", "1"), freq)
                self.status_msg.emit(
                    f"FOCUS REC {self.cfg.get('record_device','deviceA')}"
                    f"/s{self.cfg.get('record_session','1')} @ {freq/1e6:.3f} MHz: "
                    f"{len(self._fq)} files")
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
            "wf_jpg_enabled"  : WATERFALL_JPG_ENABLED,
            "wf_jpg_interval" : WATERFALL_JPG_INTERVAL,
            "wf_jpg_dir"      : WATERFALL_JPG_DIR,
            "wf_jpg_quality"  : WATERFALL_JPG_QUALITY,
            # ── v2 fingerprinting ──
            "op_mode"             : "normal",     # normal | wideband | focus
            "record"              : False,        # save data appropriate to the mode
            "record_device"       : "deviceA",
            "record_session"      : "1",
            "focus_freq"          : CENTER_FREQ,
            "record_max_files"    : 1000,
            "fp_peak_thresh_db"   : FP_PEAK_THRESH_DB,
            "fp_unknown_thresh"   : FP_UNKNOWN_THRESH,
            "fp_max_misses"       : FP_MAX_MISSES,
            "fp_nudge_step_hz"    : FP_NUDGE_STEP_HZ,
            "fp_nudge_deadzone_hz": FP_NUDGE_DEADZONE_HZ,
            "fp_hold_settle_ms"   : FP_HOLD_SETTLE_MS,
            "fp_hold_rescan_s"    : FP_HOLD_RESCAN_S,
            "fp_hold_max_s"       : FP_HOLD_MAX_S,
            "fp_memory_ttl_s"     : FP_MEMORY_TTL_S,
            "fp_memory_guard_hz"  : FP_MEMORY_GUARD_HZ,
        }
        self._recompute_hops()

        self._engine      = None
        self._last_result = None

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
        self._last_wf_jpg_t = time.time()
        self._start_worker()

    # ── SDR / hop helpers ─────────────────────────────────────────────────────

    def _recompute_hops(self):
        effective_bw = min(self.cfg["sample_rate"], self.cfg["rx_bw"])
        self.cfg["hop_freqs"] = compute_hop_freqs(
            self.cfg["center_freq"], self.cfg["total_span"],
            effective_bw, self.cfg["overlap_pct"])
        self.n_hops        = len(self.cfg["hop_freqs"])
        self.total_bins    = self.n_hops * FFT_BINS
        self.f_global_min  = self.cfg["center_freq"] - self.cfg["total_span"] // 2
        self.f_global_max  = self.cfg["center_freq"] + self.cfg["total_span"] // 2
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
        self.p2 = self.win.addPlot(row=1, col=0, title="Waterfall  (newest → top)")
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
        self.p_zoom = self.win.addPlot(row=0, col=1, title="Zoom — held signal")
        self.p_zoom.setLabel("bottom", "Frequency", units="Hz")
        self.p_zoom.setLabel("left",   "Power",     units="dBFS")
        self.p_zoom.setMouseEnabled(x=True, y=True)   # pan/zoom the held view
        self.p_zoom.setMenuEnabled(True)
        self.p_zoom.showGrid(x=True, y=True, alpha=0.25)
        self.p_zoom.setYRange(WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS, padding=0)
        self.p_zoom.getAxis("left").setWidth(_axis_w)
        self.zoom_curve = self.p_zoom.plot(pen=pg.mkPen("c", width=1))

        # ── zoom waterfall  (row 1, col 1) ──────────────────────────────────────
        self.p_zoom_wf = self.win.addPlot(row=1, col=1, title="Zoom Waterfall")
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

        self.infer_stat_lbl = QtWidgets.QLabel("Inference: —")
        self.infer_stat_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        vbox.addWidget(self.infer_stat_lbl)

        vbox.addSpacing(4)
        model_row = QtWidgets.QHBoxLayout()
        self.w_model_path = QtWidgets.QLineEdit()
        self.w_model_path.setPlaceholderText("fingerprint_spec_model.pt")
        self.w_model_path.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.w_model_path)
        model_row.addWidget(browse_btn)
        vbox.addLayout(model_row)

        load_btn = QtWidgets.QPushButton("⟳  Load / Reload Model")
        load_btn.setFixedHeight(28)
        load_btn.clicked.connect(self._on_load_model_btn)
        vbox.addWidget(load_btn)

        self.model_info_lbl = QtWidgets.QLabel("")
        self.model_info_lbl.setWordWrap(True)
        self.model_info_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        vbox.addWidget(self.model_info_lbl)

        # ══════════════════════════════════════════════════════
        # MODE SECTION  (Normal / Wideband / Focus)
        # ══════════════════════════════════════════════════════
        section("Mode")
        mode_row = QtWidgets.QHBoxLayout()
        self._mode_btns  = {}
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for key, label in (("normal", "Normal"), ("wideband", "Wideband"), ("focus", "Focus")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setFixedHeight(26)
            b.clicked.connect(lambda _=False, k=key: self._on_mode_btn(k))
            self._mode_group.addButton(b)
            self._mode_btns[key] = b
            mode_row.addWidget(b)
        self._mode_btns["normal"].setChecked(True)
        vbox.addLayout(mode_row)

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
        self.w_sr   = cell(0, 0, "Sample Rate",
                           hz_combo([2e6, 4e6, 5e6, 10e6, 15e6, 20e6, 30e6, 56e6], SAMPLE_RATE))
        self.w_bw   = cell(0, 1, "Bandwidth",
                           hz_combo([1e6, 2e6, 4e6, 5e6, 10e6, 20e6, 40e6], RX_BW_HZ))
        self.w_gain = cell(0, 2, "RX Gain (dB)", QtWidgets.QSpinBox())
        self.w_gain.setRange(-3, 71)
        self.w_gain.setValue(GAIN)
        # row 2/3: center freq · total span · overlap
        self.w_center = cell(2, 0, "Center Freq", QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_span   = cell(2, 1, "Total Span",
                             hz_combo([5e6, 10e6, 20e6, 40e6, 80e6], TOTAL_SPAN_HZ))
        self.w_olap_pct = cell(2, 2, "Overlap %", QtWidgets.QComboBox())
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


        section("Recording  ★")

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

        # Focus mode records device fingerprints at 'Focus freq'; Wideband mode
        # records the band-swept noise class. The Record toggle saves per-mode.
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
        grab_btn = QtWidgets.QPushButton("Grab lock")
        grab_btn.setFixedWidth(80)
        grab_btn.clicked.connect(self._on_grab_lock)
        rec_btn_row.addWidget(self.w_rec_btn)
        rec_btn_row.addWidget(grab_btn)
        vbox.addLayout(rec_btn_row)
        self._update_record_btn_style(False)

        vbox.addSpacing(8)
        apply_btn = QtWidgets.QPushButton("⟳  Apply Settings")
        apply_btn.setFixedHeight(34)
        apply_btn.clicked.connect(self._apply_settings)
        vbox.addWidget(apply_btn)

        section("Status")
        _ss = "color: #cccccc; font-size: 11px;"   # one unified style for all rows
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
        self.wf_jpg_lbl = QtWidgets.QLabel("")
        self.wf_jpg_lbl.setStyleSheet(_ss)
        self.prog_bar = QtWidgets.QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setValue(0)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setFixedHeight(8)
        vbox.addWidget(self.status_lbl)
        vbox.addWidget(self.file_lbl)
        vbox.addWidget(self.wf_jpg_lbl)
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
        "drone" : ("background: #cc2200; color: #ffffff;"
                   " border: 1px solid #ff4422; border-radius: 6px;"),
        "noise" : ("background: #1a5c1a; color: #ccffcc;"
                   " border: 1px solid #33aa33; border-radius: 6px;"),
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
            if "drone" in cls.lower():
                bar.setStyleSheet(
                    "QProgressBar::chunk { background: #cc4422; }"
                    "QProgressBar { border: 1px solid #666666; border-radius: 3px;"
                    " background: #1e1e1e; }")
            else:
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
            self.model_info_lbl.setText(
                f"✓ {short}\n"
                f"Classes: {', '.join(engine.classes)}\n"
                f"seg {engine.seg_len} · unknown < {engine.unknown_thresh:.2f}"
            )
        except Exception as e:
            self._engine = None
            self.model_info_lbl.setText(f"⚠ Load error:\n{e}")
            self.det_badge.setText("LOAD ERROR")
            self._set_badge_style("error")

    # ── ML: result + zoom handlers (inference runs in the worker) ──────────────

    def _on_fingerprint_ready(self, result):
        self._last_result = result
        label = result["label"]
        conf  = result["confidence"]
        probs = result["probs"]
        if result.get("unknown"):
            self.det_badge.setText(f"…  UNKNOWN  {conf:.0%}")
            self._set_badge_style("other")
        else:
            self.det_badge.setText(f"✓  {label.upper()}  {conf:.0%}")
            self._set_badge_style("device")
        for cls, bar in self._conf_bars.items():
            p = probs.get(cls, 0.0)
            bar.setValue(int(p * 100))
            self._conf_labels[cls].setText(f"{p:.0%}")
        self.infer_stat_lbl.setText(f"{label} @ {conf:.1%}")

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
        self.p_zoom.setTitle(f"Zoom — {held_freq / 1e6:.3f} MHz")
        # scroll the held-signal spectrum into the zoom waterfall
        self.zoom_wf_data = np.roll(self.zoom_wf_data, -1, axis=1)
        self.zoom_wf_data[:, -1] = psd
        self.zoom_wf_img.setImage(self.zoom_wf_data, autoLevels=False)
        self.zoom_wf_img.setRect(QtCore.QRectF(held_freq - sr / 2, 0, sr, WATERFALL_ROWS))

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
        effective_bw = getattr(self, "_effective_bw",
                               min(self.cfg["sample_rate"], self.cfg["rx_bw"]))
        overlap_hz = int(effective_bw * self.cfg["overlap_pct"] / 100.0)
        step = effective_bw - overlap_hz
        for i in range(self.n_hops + 1):
            x  = self.f_global_min + i * step
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

    def _on_mode_btn(self, key: str):
        if self.sdr is None:
            self.status_lbl.setText("SDR not connected.")
            return
        if self.cfg.get("op_mode") == key:
            return
        self.cfg["op_mode"] = key
        self._sync_record_cfg()
        self.worker.stop()
        self._start_worker()

    def _on_record_toggle(self, checked: bool):
        self._sync_record_cfg()
        self.cfg["record"] = checked          # read live by the worker loops
        self._update_record_btn_style(checked)

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