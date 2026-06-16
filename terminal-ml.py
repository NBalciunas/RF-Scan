"""
pluto_monitor_ml.py  –  PlutoSDR Wideband Monitor  +  Live ML Inference
"""

import os
import sys
import time
import math
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
TOTAL_SPAN_HZ   = 10_000_000
HOP_DWELL_MS    = 50
HOP_SETTLE_MS   = 50
HOP_OVERLAP_PCT = 30

DRONE_NAME      = "Noise"
SERIAL_NUM      = "00001"
REFERENCE_SNR   = 40

WIDEBAND_RECORD_ENABLED = False
WIDEBAND_SECS           = 2.0
WIDEBAND_MAX_FILES      = 50

WATERFALL_JPG_ENABLED   = False
WATERFALL_JPG_INTERVAL  = 5.0
WATERFALL_JPG_DIR       = "./output/waterfall_snaps"
WATERFALL_JPG_QUALITY   = 90

FFT_BINS           = 1024
WATERFALL_ROWS     = 200
WF_SCALE_MIN_DBFS  = -10.0
WF_SCALE_MAX_DBFS  = 10.0

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "detector_quick.pt")

INFER_SWEEP_HISTORY = 3

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
# ML MODEL WRAPPER
# ==========================================

class SpectralCNNSlim(object):

    @staticmethod
    def _build(n_classes, n_bins=FFT_BINS, base_channels=16):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")

        # Must mirror ml-train.py exactly (including SEBlock) so state_dict loads cleanly.
        class _SEBlock(nn.Module):
            def __init__(self, channels, reduction=4):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(channels, channels // reduction, bias=False),
                    nn.GELU(),
                    nn.Linear(channels // reduction, channels, bias=False),
                    nn.Sigmoid(),
                )
            def forward(self, x):
                return x * self.fc(x).unsqueeze(-1)

        def block(ic, oc, k, pool):
            return nn.Sequential(
                nn.Conv1d(ic, oc, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(oc),
                nn.GELU(),
                _SEBlock(oc),
                nn.MaxPool1d(pool),
            )

        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    block(1,  c1, 15, 4),
                    block(c1, c2, 7,  4),
                    block(c2, c3, 5,  4),
                    block(c3, c4, 3,  2),
                )
                cnn_flat = c4 * (n_bins // (4 * 4 * 4 * 2))
                self.freq_branch = nn.Sequential(
                    nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16),
                )
                self.head = nn.Sequential(
                    nn.Linear(cnn_flat + 16, 128),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, n_classes),
                )

            def forward(self, spectrum, meta):
                cnn_out  = self.cnn(spectrum).flatten(1)
                freq_out = self.freq_branch(meta)
                return self.head(torch.cat([cnn_out, freq_out], dim=1))

        return _Net()


class InferenceEngine:
    def __init__(self, model_path: str):
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed. Run: pip install torch")

        self.model_path = model_path
        meta_path = os.path.splitext(model_path)[0] + ".meta.json"

        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self.classes   = meta.get("classes",    ["noise", "drone"])
            self.psd_mean  = float(meta.get("psd_mean",  0.0))
            self.psd_std   = float(meta.get("psd_std",   1.0))
            self.freq_mean = float(meta.get("freq_mean", 0.0))
            self.freq_std  = float(meta.get("freq_std",  1.0))
            self.span_mean = float(meta.get("span_mean", 0.0))
            self.span_std  = float(meta.get("span_std",  1.0))
            # Per-bin normalisation + floor clip must match training (see
            # ml-precomp.py / ml-train.py).  None ⇒ old scalar-only model.
            bm = meta.get("psd_bin_mean")
            bs = meta.get("psd_bin_std")
            self.psd_bin_mean = None if bm is None else np.asarray(bm, dtype=np.float32)
            self.psd_bin_std  = None if bs is None else np.asarray(bs, dtype=np.float32)
            self.floor_db     = meta.get("floor_db")
            self.base_channels = int(meta.get("base_channels", 16))
        else:
            self.classes   = ["noise", "drone"]
            self.psd_mean  = 0.0
            self.psd_std   = 1.0
            self.freq_mean = 0.0
            self.freq_std  = 1.0
            self.span_mean = 0.0
            self.span_std  = 1.0
            self.psd_bin_mean = None
            self.psd_bin_std  = None
            self.floor_db     = None
            self.base_channels = 16

        n_classes = len(self.classes)
        self.net  = SpectralCNNSlim._build(n_classes, base_channels=self.base_channels)
        state     = torch.load(model_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state)
        self.net.eval()

        self._blackman = np.blackman(FFT_BINS).astype(np.float32)

    def _iq_to_psd_windows(self, iq_chunk: np.ndarray) -> np.ndarray:
        """
        Compute Blackman-windowed PSDs for every FFT_BINS-sample window in the
        buffer — (n_windows, FFT_BINS), in dBFS.

        The drone emission is bursty, so most windows in a buffer are silent
        gaps that look like noise.  Averaging them together (the old behaviour)
        diluted any burst straight back into the noise floor — the same label
        noise that burst-gating removes at training time.  We therefore keep the
        windows separate and let the caller pick the most burst-like one, which
        mirrors how train_drone.py gates the training set.

        For wideband stitched IQ the buffer is n_hops × FFT_BINS samples long, so
        many windows are produced; for narrowband IQ (≈FFT_BINS samples) a single
        window comes back and behaviour is unchanged.
        """
        n = len(iq_chunk)
        if n < FFT_BINS:
            iq_chunk = np.pad(iq_chunk.astype(np.complex64), (0, FFT_BINS - n))
            n = FFT_BINS

        # Overlapping half-windows across the whole buffer (cap so a very long
        # stitched buffer can't blow up the per-sweep forward pass).
        hop       = FFT_BINS // 2
        n_windows = min(32, (n - FFT_BINS) // hop + 1)
        step      = (n - FFT_BINS) // (n_windows - 1) if n_windows > 1 else 0
        offsets   = [i * step for i in range(n_windows)]

        psds = np.empty((n_windows, FFT_BINS), dtype=np.float32)
        for i, off in enumerate(offsets):
            chunk = iq_chunk[off:off + FFT_BINS].astype(np.complex64)
            psds[i] = 20.0 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(chunk * self._blackman)))
                / FFT_BINS + 1e-10
            )
        if self.floor_db is not None:          # match training floor clip
            np.maximum(psds, np.float32(self.floor_db), out=psds)
        return psds

    def infer_from_iq(self, iq: np.ndarray,
                      center_freq_hz: float,
                      freq_span_hz: float = 0.0) -> dict:
        psds = self._iq_to_psd_windows(iq)     # (n_windows, FFT_BINS) raw dBFS

        if self.psd_bin_mean is not None:      # per-bin normalisation (preferred)
            x = (psds - self.psd_bin_mean) / self.psd_bin_std
        else:
            x = (psds - self.psd_mean) / self.psd_std

        spec_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)   # (N, 1, BINS)

        freq_n = (center_freq_hz - self.freq_mean) / self.freq_std
        span_n = (freq_span_hz   - self.span_mean) / self.span_std
        meta_t = torch.tensor([[freq_n, span_n]], dtype=torch.float32).repeat(
            len(psds), 1)                                            # (N, 2)

        with torch.no_grad():
            logits   = self.net(spec_t, meta_t)
            all_probs = torch.softmax(logits, dim=1).numpy()         # (N, n_classes)

        # "Drone present if ANY window is a burst": report the single window most
        # confident about drone, instead of averaging windows into the floor.
        # Falls back to the most confident window overall when there is no
        # explicit drone class.
        drone_idx = next((i for i, c in enumerate(self.classes)
                          if "drone" in c.lower()), None)
        if drone_idx is not None:
            win = int(all_probs[:, drone_idx].argmax())
        else:
            win = int(all_probs.max(axis=1).argmax())
        probs = all_probs[win]

        idx   = int(probs.argmax())
        label = self.classes[idx] if idx < len(self.classes) else str(idx)
        return {
            "label"     : label,
            "confidence": float(probs[idx]),
            "probs"     : {cls: float(p) for cls, p in zip(self.classes, probs)},
        }


# ==========================================
# SWEEP WORKER
# ==========================================

class SweepWorker(QtCore.QThread):
    sweep_ready   = QtCore.pyqtSignal(object, object)
    hop_progress  = QtCore.pyqtSignal(int, int)
    status_msg    = QtCore.pyqtSignal(str)
    files_changed = QtCore.pyqtSignal(int)

    _BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)

    def __init__(self, sdr, cfg):
        super().__init__()
        self.sdr           = sdr
        self.cfg           = cfg
        self._stop         = False
        self._wb_stitcher  = None
        self._wb_accum     = None
        self._wb_flush_t   = time.time()
        self._wb_fq        = collections.deque()
        self._rebuild_stitcher()

    def stop(self):
        self._stop = True
        if not self.wait(4000):
            self.status_msg.emit(
                "Warning: sweep thread blocked on sdr.rx() — forcing terminate."
            )
            self.terminate()
            self.wait(1000)

    def _rebuild_stitcher(self):
        effective_bw = min(self.cfg["sample_rate"], self.cfg["rx_bw"])
        self._wb_stitcher = WidebandStitcher(
            hop_freqs   = self.cfg["hop_freqs"],
            hop_bw      = effective_bw,
            total_span  = self.cfg["total_span"],
            center_freq = self.cfg["center_freq"],
        )
        self._wb_accum   = None
        self._wb_flush_t = time.time()

    def run(self):
        self._rebuild_stitcher()
        while not self._stop:
            t0         = time.perf_counter()
            hop_freqs  = self.cfg["hop_freqs"]
            n_hops     = len(hop_freqs)
            total_bins = n_hops * FFT_BINS
            composite  = np.full(total_bins, -100.0, dtype=np.float32)
            sweep_hop_bufs = {}

            for i, freq in enumerate(hop_freqs):
                if self._stop:
                    return
                self.hop_progress.emit(i, n_hops)
                try:
                    self.sdr.rx_lo = freq
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
                sweep_hop_bufs[i] = raw
                chunk = raw[:FFT_BINS]
                if len(chunk) < FFT_BINS:
                    chunk = np.pad(chunk, (0, FFT_BINS - len(chunk)))
                psd = 20.0 * np.log10(
                    np.abs(np.fft.fftshift(
                        np.fft.fft(chunk * self._BLACKMAN))) / FFT_BINS + 1e-10
                )
                composite[i * FFT_BINS:(i + 1) * FFT_BINS] = psd.astype(np.float32)

            if self.cfg.get("wideband_record_enabled", True) and sweep_hop_bufs:
                wb_chunk = self._wb_stitcher.stitch(sweep_hop_bufs)
                if wb_chunk is not None:
                    self._wb_accum = wb_chunk if self._wb_accum is None \
                                     else np.concatenate((self._wb_accum, wb_chunk))
                now = time.time()
                if (now - self._wb_flush_t) >= self.cfg.get("wideband_secs", WIDEBAND_SECS):
                    if self._wb_accum is not None and len(self._wb_accum) > 0:
                        self._save_wideband_file(self._wb_accum)
                    self._wb_accum   = None
                    self._wb_flush_t = now
            else:
                self._wb_accum = None

            if not self._stop:
                elapsed = time.perf_counter() - t0
                wb_tag  = "  |  wb ⏺ OFF" if not self.cfg.get("wideband_record_enabled", True) else ""
                self.status_msg.emit(
                    f"Sweep: {elapsed * 1000:.0f} ms  |  "
                    f"{1.0 / elapsed:.2f} sweeps/s  |  "
                    f"{n_hops} hops{wb_tag}"
                )
                self.sweep_ready.emit(composite, sweep_hop_bufs)

    def _save_wideband_file(self, data):
        record_class = self.cfg.get("record_class", "noise")
        wb_dir = os.path.join("./training_data", record_class)
        os.makedirs(wb_dir, exist_ok=True)
        ts    = int(time.time() * 1000)
        fname = f"capture_{self.cfg['center_freq'] // 1_000_000}MHz_{ts}.iq"
        fpath = os.path.join(wb_dir, fname)
        data.astype(np.complex64).tofile(fpath)
        f_min = self.cfg["center_freq"] - self.cfg["total_span"] // 2
        f_max = self.cfg["center_freq"] + self.cfg["total_span"] // 2
        write_xml(_xml_path(fpath), fname, len(data),
                  self.cfg["center_freq"], self._wb_stitcher.synthetic_rate,
                  extra_tags={
                      "FreqMin"      : str(f_min),
                      "FreqMax"      : str(f_max),
                      "HopCount"     : str(len(self.cfg["hop_freqs"])),
                      "HopBandwidth" : str(min(self.cfg["sample_rate"], self.cfg["rx_bw"])),
                      "StitchMethod" : "FFT-stitch-IFFT-Hann-cosine-taper",
                      "SyntheticRate": "true",
                  })
        max_wb = self.cfg.get("wideband_max_files", WIDEBAND_MAX_FILES)
        self._wb_fq.append(fpath)
        while len(self._wb_fq) > max_wb:
            old = self._wb_fq.popleft()
            for p in [old, _xml_path(old)]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        self.status_msg.emit(f"Warning: could not remove {p}: {e}")
        self.files_changed.emit(len(self._wb_fq))


# ==========================================
# MAIN APPLICATION
# ==========================================

class PlutoApp(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR Wideband Monitor  +  ML Inference")
        self.resize(1600, 950)

        self.cfg = {
            "sample_rate"            : SAMPLE_RATE,
            "rx_bw"                  : RX_BW_HZ,
            "gain"                   : GAIN,
            "center_freq"            : CENTER_FREQ,
            "total_span"             : TOTAL_SPAN_HZ,
            "dwell_ms"               : HOP_DWELL_MS,
            "settle_ms"              : HOP_SETTLE_MS,
            "overlap_pct"            : HOP_OVERLAP_PCT,
            "hop_freqs"              : [],
            "wideband_record_enabled": WIDEBAND_RECORD_ENABLED,
            "wideband_secs"          : WIDEBAND_SECS,
            "wideband_max_files"     : WIDEBAND_MAX_FILES,
            "wf_jpg_enabled"         : WATERFALL_JPG_ENABLED,
            "wf_jpg_interval"        : WATERFALL_JPG_INTERVAL,
            "wf_jpg_dir"             : WATERFALL_JPG_DIR,
            "wf_jpg_quality"         : WATERFALL_JPG_QUALITY,
            "record_class"            : "noise",
        }
        self._recompute_hops()

        self._engine      = None
        self._last_result = None
        self._prob_history = collections.deque(maxlen=INFER_SWEEP_HISTORY)

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

        # ── spectrum plot ──────────────────────────────────────────────────────
        self.p1 = self.win.addPlot(title="Wideband Spectrum")
        self.p1.setLabel("bottom", "Frequency", units="Hz")
        self.p1.setLabel("left",   "Power",     units="dBFS")
        # Disable auto-range so per-sweep setData() can't re-fit the view (which
        # adds default padding, pushing the axes off their limits and — via the
        # X-link — off the waterfall image edges).
        self.p1.enableAutoRange(x=False, y=False)
        self.p1.setXRange(self.f_global_min, self.f_global_max, padding=0)
        self.p1.setYRange(WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS, padding=0)
        self.p1.setMouseEnabled(x=True, y=True)
        self.p1.setMenuEnabled(False)
        self.p1.showGrid(x=True, y=True, alpha=0.25)
        self.hop_lines = []
        self._rebuild_hop_lines()
        self.curve = self.p1.plot(pen=pg.mkPen("y", width=1))

        self.win.nextRow()

        # ── waterfall plot ─────────────────────────────────────────────────────
        self.p2 = self.win.addPlot(title="Waterfall  (newest → top)")
        self.p2.setLabel("bottom", "Frequency", units="Hz")
        self.p2.setLabel("left",   "Time",      units="sweeps")
        self.p2.setXLink(self.p1)
        self.p2.setMouseEnabled(x=False, y=False)
        self.p2.setMenuEnabled(False)
        self.p2.getViewBox().setAutoVisible(x=False, y=False)
        self.p2.enableAutoRange(x=False, y=False)
        # Match the two plots' left-axis widths so their plot areas (and thus
        # the linked x-axis) line up. p2's labels ("200") are wider than p1's,
        # which otherwise shifts the waterfall relative to the spectrum.
        _axis_w = 64
        self.p1.getAxis("left").setWidth(_axis_w)
        self.p2.getAxis("left").setWidth(_axis_w)
        self.img = pg.ImageItem(axisOrder="col-major")
        self.p2.addItem(self.img)
        cmap = pg.colormap.get("viridis")
        self.img.setLookupTable(cmap.getLookupTable())
        self.img.setLevels([WF_SCALE_MIN_DBFS, WF_SCALE_MAX_DBFS])

        # ── right side panel ───────────────────────────────────────────────────
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(300)
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
        self.w_model_path.setPlaceholderText("detector_quick.pt")
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
        # SDR SECTION
        # ══════════════════════════════════════════════════════
        section("SDR")
        self.w_sr   = labeled("Sample Rate (Hz):",  QtWidgets.QLineEdit(str(SAMPLE_RATE)))
        self.w_bw   = labeled("RX Bandwidth (Hz):", QtWidgets.QLineEdit(str(RX_BW_HZ)))
        self.w_gain = labeled("RX Gain (dB):",      QtWidgets.QSpinBox())
        self.w_gain.setRange(-3, 71)
        self.w_gain.setValue(GAIN)

        section("Frequency Hopping")
        self.w_center = labeled("Center Freq (Hz):",   QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_span   = labeled("Total Span (Hz):",    QtWidgets.QLineEdit(str(TOTAL_SPAN_HZ)))
        self.w_dwell  = labeled("Dwell per hop (ms):", QtWidgets.QSpinBox())
        self.w_dwell.setRange(1, 5000)
        self.w_dwell.setValue(HOP_DWELL_MS)
        self.w_settle = labeled("Settle time (ms):",   QtWidgets.QSpinBox())
        self.w_settle.setRange(0, 500)
        self.w_settle.setValue(HOP_SETTLE_MS)
        self.w_olap_pct = labeled("Hop Overlap (%):", QtWidgets.QSpinBox())
        self.w_olap_pct.setRange(0, 75)
        self.w_olap_pct.setSuffix(" %")
        self.w_olap_pct.setValue(HOP_OVERLAP_PCT)
        self.hop_info_lbl = QtWidgets.QLabel("")
        self.hop_info_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        vbox.addWidget(self.hop_info_lbl)
        self._update_hop_info_label()

        section("Waterfall Scale  (dBFS)")
        scale_row = QtWidgets.QHBoxLayout()
        self.w_wf_min = QtWidgets.QSpinBox()
        self.w_wf_min.setRange(-140, 0)
        self.w_wf_min.setValue(int(WF_SCALE_MIN_DBFS))
        self.w_wf_min.setPrefix("min ")
        self.w_wf_max = QtWidgets.QSpinBox()
        self.w_wf_max.setRange(-100, 10)
        self.w_wf_max.setValue(int(WF_SCALE_MAX_DBFS))
        self.w_wf_max.setPrefix("max ")
        scale_row.addWidget(self.w_wf_min)
        scale_row.addWidget(self.w_wf_max)
        vbox.addLayout(scale_row)
        self.w_wf_min.valueChanged.connect(self._apply_wf_scale)
        self.w_wf_max.valueChanged.connect(self._apply_wf_scale)


        section("Wideband Recording  ★")

        cls_row = QtWidgets.QHBoxLayout()
        cls_lbl = QtWidgets.QLabel("Record class:")
        cls_lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.w_record_class = QtWidgets.QComboBox()
        self.w_record_class.addItems(["noise", "drone"])
        self.w_record_class.setStyleSheet(
            "QComboBox { color: #ffffff; background-color: #2a2a2a;"
            " border: 1px solid #555555; border-radius: 3px; padding: 1px 3px; }"
            "QComboBox QAbstractItemView { color: #ffffff; background-color: #2a2a2a; }"
        )
        self.w_record_class.currentTextChanged.connect(self._on_record_class_changed)
        cls_row.addWidget(cls_lbl)
        cls_row.addWidget(self.w_record_class)
        vbox.addLayout(cls_row)

        self.w_wb_btn = QtWidgets.QPushButton("▶  Start Recording")
        self.w_wb_btn.setFixedHeight(30)
        self.w_wb_btn.setCheckable(True)
        self.w_wb_btn.setChecked(False)
        self.w_wb_btn.clicked.connect(self._on_wb_btn)
        vbox.addWidget(self.w_wb_btn)
        self._update_wb_btn_style(False)

        self._wb_widgets = []

        def wb_labeled(label_text, widget):
            lbl = QtWidgets.QLabel(label_text)
            lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
            vbox.addWidget(lbl)
            vbox.addWidget(widget)
            self._wb_widgets.append(widget)
            return widget

        self.w_wb_secs = wb_labeled("Flush interval (s):", QtWidgets.QDoubleSpinBox())
        self.w_wb_secs.setRange(0.5, 300.0)
        self.w_wb_secs.setValue(WIDEBAND_SECS)
        self.w_wb_maxf = wb_labeled("Max wideband files:", QtWidgets.QSpinBox())
        self.w_wb_maxf.setRange(1, 5000)
        self.w_wb_maxf.setValue(WIDEBAND_MAX_FILES)

        vbox.addSpacing(8)
        apply_btn = QtWidgets.QPushButton("⟳  Apply Settings")
        apply_btn.setFixedHeight(34)
        apply_btn.clicked.connect(self._apply_settings)
        vbox.addWidget(apply_btn)

        section("Status")
        self.status_lbl = QtWidgets.QLabel("Starting…")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("color: #dddddd; font-size: 11px;")
        self.file_lbl = QtWidgets.QLabel("Files on disk: 0")
        self.file_lbl.setStyleSheet("color: #dddddd; font-size: 11px;")
        self.wf_jpg_lbl = QtWidgets.QLabel("")
        self.wf_jpg_lbl.setStyleSheet("color: #aaaaaa; font-size: 10px;")
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

        root.addWidget(panel, stretch=0)

    # ── ML: badge styling ─────────────────────────────────────────────────────

    _BADGE_STYLES = {
        "none"  : ("background: #3a3a3a; color: #ffffff;"
                   " border: 1px solid #555555; border-radius: 6px;"),
        "drone" : ("background: #cc2200; color: #ffffff;"
                   " border: 1px solid #ff4422; border-radius: 6px;"),
        "noise" : ("background: #1a5c1a; color: #ccffcc;"
                   " border: 1px solid #33aa33; border-radius: 6px;"),
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
            engine = InferenceEngine(path)
            self._engine = engine
            self._prob_history.clear()
            self._rebuild_conf_bars(engine.classes)
            self.det_badge.setText("MODEL LOADED — WAITING")
            self._set_badge_style("none")
            short = os.path.basename(path)
            self.model_info_lbl.setText(
                f"✓ {short}\n"
                f"Classes: {', '.join(engine.classes)}\n"
                f"PSD mean/std: {engine.psd_mean:.1f} / {engine.psd_std:.1f} dBFS"
            )
        except Exception as e:
            self._engine = None
            self.model_info_lbl.setText(f"⚠ Load error:\n{e}")
            self.det_badge.setText("LOAD ERROR")
            self._set_badge_style("error")

    # ── ML: run inference ─────────────────────────────────────────────────────

    def _run_inference(self, hop_bufs: dict):
        if self._engine is None or not hop_bufs:
            return

        t0 = time.perf_counter()

        # Stitch all hops into a single wideband IQ — this matches what
        # terminal-no-ml records and what the model was trained on.
        iq = None
        if hasattr(self, "worker") and self.worker._wb_stitcher is not None:
            iq = self.worker._wb_stitcher.stitch(hop_bufs)

        # Fallback: single middle-hop IQ (narrowband; suboptimal but won't crash)
        if iq is None or len(iq) < 1024:
            mid = len(self.cfg["hop_freqs"]) // 2
            iq  = hop_bufs.get(mid)
            if iq is None:
                iq = next(iter(hop_bufs.values()))

        cf   = float(self.cfg["center_freq"])
        span = float(self.cfg["total_span"])

        try:
            result = self._engine.infer_from_iq(iq, cf, span)
        except Exception as e:
            self.infer_stat_lbl.setText(f"Inference error: {e}")
            return

        ms = (time.perf_counter() - t0) * 1000

        # Peak-hold over the last INFER_SWEEP_HISTORY sweeps.  A bursty drone is
        # only visible in occasional sweeps, so averaging probabilities across
        # sweeps washes the detection out (same dilution problem as averaging
        # windows).  Instead, report the recent sweep that was most confident
        # about drone — a short hold so a single burst still raises the alarm,
        # and it self-clears once the burst leaves the history window.
        self._prob_history.append(result["probs"])
        drone_key = next((c for c in result["probs"] if "drone" in c.lower()), None)
        if drone_key is not None:
            probs = max(self._prob_history, key=lambda h: h.get(drone_key, 0.0))
        else:
            probs = max(self._prob_history, key=lambda h: max(h.values()))
        best_cls = max(probs, key=probs.get)
        result   = {"label": best_cls, "confidence": probs[best_cls], "probs": probs}

        self._last_result = result
        label = result["label"]
        conf  = result["confidence"]
        probs = result["probs"]

        if "drone" in label.lower():
            self.det_badge.setText(f"⚠  DRONE DETECTED  {conf:.0%}")
            self._set_badge_style("drone")
        elif "noise" in label.lower():
            self.det_badge.setText(f"✓  NOISE  {conf:.0%}")
            self._set_badge_style("noise")
        else:
            self.det_badge.setText(f"{label.upper()}  {conf:.0%}")
            self._set_badge_style("other")

        for cls, bar in self._conf_bars.items():
            p = probs.get(cls, 0.0)
            bar.setValue(int(p * 100))
            self._conf_labels[cls].setText(f"{p:.0%}")

        self.infer_stat_lbl.setText(
            f"Inference: {ms:.1f} ms/sweep  |  {label} @ {conf:.1%}")

    # ── waterfall helpers ─────────────────────────────────────────────────────

    def _update_waterfall_rect(self):
        span = self.f_global_max - self.f_global_min
        self.img.setRect(QtCore.QRectF(self.f_global_min, 0, span, WATERFALL_ROWS))
        # Pin p2's view to the data so the image can't render wider/taller than
        # the axis. x is XLinked to p1; y is fixed to the sweep-row count.
        self.p2.setYRange(0, WATERFALL_ROWS, padding=0)

    def _apply_wf_scale(self):
        v_min = float(self.w_wf_min.value())
        v_max = float(self.w_wf_max.value())
        if v_max <= v_min:
            return
        self.img.setLevels([v_min, v_max])

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
        self.hop_info_lbl.setText(
            f"<i>{self.n_hops} hops · "
            f"{self.cfg['total_span'] / 1e6:.1f} MHz total · "
            f"~{sweep_ms} ms/sweep</i>")

    # ── toggle helpers ────────────────────────────────────────────────────────

    def _on_record_class_changed(self, text: str):
        self.cfg["record_class"] = text
        if self.cfg.get("wideband_record_enabled"):
            self._update_wb_btn_style(True)

    def _on_wb_btn(self, checked: bool):
        self.cfg["wideband_record_enabled"] = checked
        self._update_wb_btn_style(checked)
        if hasattr(self, "worker"):
            self.worker.cfg["wideband_record_enabled"] = checked

    def _update_wb_btn_style(self, recording: bool):
        if recording:
            self.w_wb_btn.setText(
                f"■  Stop Recording  [{self.cfg.get('record_class','noise')}]")
            self.w_wb_btn.setStyleSheet(
                "QPushButton { color: #ffffff; background-color: #882200;"
                " border: 1px solid #cc4422; border-radius: 3px; padding: 3px 6px; }"
                "QPushButton:hover { background-color: #aa2200; }"
            )
        else:
            self.w_wb_btn.setText("▶  Start Recording")
            self.w_wb_btn.setStyleSheet("")  # inherit panel default

    # ── settings apply ────────────────────────────────────────────────────────

    def _apply_settings(self):
        if self.sdr is None:
            self.status_lbl.setText("SDR not connected — cannot apply settings.")
            return
        self.worker.stop()
        try:
            self.cfg["sample_rate"]        = int(float(self.w_sr.text()))
            self.cfg["rx_bw"]              = int(float(self.w_bw.text()))
            self.cfg["gain"]               = self.w_gain.value()
            self.cfg["center_freq"]        = int(float(self.w_center.text()))
            self.cfg["total_span"]         = int(float(self.w_span.text()))
            self.cfg["dwell_ms"]           = self.w_dwell.value()
            self.cfg["settle_ms"]          = self.w_settle.value()
            self.cfg["overlap_pct"]        = self.w_olap_pct.value()
            self.cfg["wideband_secs"]      = self.w_wb_secs.value()
            self.cfg["wideband_max_files"] = self.w_wb_maxf.value()
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
        self.worker = SweepWorker(self.sdr, self.cfg)
        self.worker.sweep_ready.connect(self._on_sweep_ready)
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

        self._run_inference(hop_bufs)

    def _on_hop_progress(self, hop_idx, total):
        self.prog_bar.setValue(int(100 * hop_idx / max(total, 1)))

    def _on_status(self, msg):
        self.status_lbl.setText(msg)
        self.prog_bar.setValue(100)

    def _on_files_changed(self, count):
        self.file_lbl.setText(f"Files on disk: {count}")

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