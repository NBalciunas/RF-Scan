"""The PlutoSDR monitor with the RF-fingerprint detection.

The worker thread has four modes:
  * Auto and Locking sweep the band. When a peak goes above FP_PEAK_THRESH_DB, the
    program holds the radio on that frequency and classifies the signal. The hold
    continues until the user clicks Skip, or until the signal stops. In Auto the
    program also releases a hold that the classifier calls noise.
  * Wideband sweeps the band only.
  * Narrowband holds one frequency only.

The program also records raw IQ data to fingerprint_data/ for train_model.py. There
are three kinds of record: a device, the noise of the full band, and the noise at
one frequency.
"""

import os
import sys
import time
import json
import math
import re
import collections
from pathlib import Path

import numpy as np
import adi

# Import torch before Qt. On Windows the Qt libraries and the torch libraries share
# dependencies, and Qt first makes c10.dll fail with OSError 1114. The program then
# stops at the start. Do not move this block below the two imports that follow it.
# The except catches Exception, because that failure is an OSError and not an
# ImportError, thus a narrow except does not give the fallback either.
try:
    import torch  # a check only — fp_spectrogram does the import of the model
    from fp_spectrogram import AMBIENT_LABELS
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    # torch is missing, thus no model runs and no vote or badge reads these. The literal
    # keeps the badge functions definable; fp_spectrogram holds the one true copy.
    AMBIENT_LABELS = ("noise", "wifi", "bluetooth")

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

# ── Configuration ─────────────────────────────────────────────────────────────

SDR_URI         = os.environ.get("PLUTO_URI", "ip:192.168.2.1")   # the factory address
SAMPLE_RATE     = 10_000_000
RX_BW_HZ        = 8_000_000   # 0.8 of the sample rate. A lower value fills part of each
                              # spectrogram with the skirt of the filter and not with signal.
GAIN            = 10

CENTER_FREQ     = 2_400_000_000   # the low edge of the 2.4 GHz ISM band. The narrowband
                                  # mode and the record path also start here.
TOTAL_SPAN_HZ   = 100_000_000     # 2350 to 2450 MHz at this centre. 15 hops at 10 Msps.
                                  # The ISM band goes to 2483.5 MHz, thus the top of the
                                  # band is outside the default sweep. Set the centre to
                                  # 2440 MHz to hold all of it.
HOP_DWELL_MS    = 50
HOP_SETTLE_MS   = 5           # the wait after a tune, before the program reads. It was
                              # 50 ms until 2026-08-14, where it was believed to keep the
                              # data of the new frequency. It never did that: the queue of
                              # the driver did, by accident. See RX_KERNEL_BUFFERS and the
                              # defect #36. The radio gives the new frequency in the first
                              # buffer at every wait from 0 to 50 ms, measured.
HOP_OVERLAP_PCT = 30
RX_KERNEL_BUFFERS = 1         # libiio keeps four buffers and rx() gives the oldest, thus a
                              # change of rx_lo appeared only in the fourth buffer that the
                              # program read. One buffer costs 51 ms for each read and it
                              # makes the next read hold the frequency that was asked for.
                              # See the defect #36.

FFT_BINS           = 1024
PSD_CHUNK_WINS     = 1024     # FFT windows for each block of the peak hold. Memory only.
EMPTY_SLOT_DB    = -300.0   # a hop that gave no data. _peak_hold_psd stops at -200.
WATERFALL_ROWS     = 200
WF_SCALE_MIN_DB  = -10.0
WF_SCALE_MAX_DB  = 10.0

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "trained_model.pt")

# ── The parameters of the lock. Adjust them against a live signal. ────────────
FP_PEAK_THRESH_DB    = 18.0      # dB of true SNR for a peak to cause a lock. The floor is
                                 # corrected with peak_hold_bias_db, thus this value is a
                                 # real SNR and it does not move with the dwell time.
FP_HOLD_SETTLE_MS    = 20        # wait for the radio before the program reads a capture
FP_GONE_S            = 2.5       # release the lock if the signal stops for this time
FP_MEMORY_TTL_S      = 30.0      # a caught frequency stays in the memory this long
FP_MEMORY_TTL_ROUNDS = 4.0       # the safety valve of the caught memory, in units of
                                 # fp_memory_ttl_s. The memory clears when the scan has
                                 # been round once, thus the wall clock only has to stop
                                 # a round that never ends from blinding the scanner.
                                 # See the defect #40.
FP_MEMORY_GUARD_HZ   = 3_000_000 # a peak nearer than this to a caught frequency is the same
FP_AUTO_DWELL_MS     = 5000      # Auto: hold this long before the program judges the lock
FP_AUTO_NOISE_PCT    = 85        # Auto: release the lock at this probability of noise
FP_DEVICE_HOLD_S     = 5.0       # Auto: a device that was seen this recently holds the
                                 # lock. A hopping link is silent between its bursts, and
                                 # silence is not absence. See the defect #30.
FP_REFINE_LOCK       = True      # centre a new lock on the signal, one time
FP_REFINE_MAX_FRAC   = 0.25      # the refinement may move the lock by this part of the
                                 # sample rate. A larger step is another signal.
FP_PEAK_HITS         = 2         # sweeps that must agree before a peak causes a lock.
                                 # 1 = lock on one sweep, as before. See the defect #31.
FP_DEVICE_THRESH     = 0.4       # the badge shows a device at this probability. Below it, the
                                 # badge shows "clear". The votes use unknown_thresh.
FP_SECOND_NAME_SHARE = 0.25      # a second name on the badge needs this share of the
                                 # segments, where the first name needs MIN_SEG_SHARE of
                                 # 0.10. The catch-all class of a model puts a weak vote
                                 # into a strong single-drone capture, thus the badge
                                 # named a drone that was not on the air. Measured on the
                                 # air 2026-08-18: 12% and 17% against a drone at 33%,
                                 # over 252 live classifications, and never once in 600
                                 # captures of the dataset. See the defect #41.
FP_HOLD_SHARE        = 0.05      # a vote at this share of the segments holds a lock. It is
                                 # below MIN_SEG_SHARE on purpose: to keep a lock costs one
                                 # dwell, and to drop one costs the drone. See the defect #30.
ML_INTERVAL_S        = 0.75      # the minimum time between two runs of the classifier
RECORD_DIR           = "./fingerprint_data"   # relative to the current directory
RECORD_EVERY_N       = 10        # the record keeps every Nth buffer, or every Nth sweep in
                                 # Wideband. Thus the same quantity of files covers more time
                                 # and gives more different data. One buffer at the default
                                 # settings is 4 MB, and there are 20 of them each second.

# ── The markers of the narrowband signal on the zoom plot. ────────────────────
MARK_MIN_SNR_DB     = 6.0   # the peak must be this many dB above the floor
MARK_EDGE_MARGIN_DB = 3.0   # an edge is where the signal goes below floor plus this
MARK_SMOOTH_BINS    = 5     # a smooth operation prevents a movement of the edges
MARK_CLIP_EDGE_FRAC = 0.02  # an edge inside this part of the window is not a real edge.
                            # The signal continues past the receiver. See signal_clipped.
MARK_BAND_FLOOR_PCT = 25    # the percentile of the swept band that is its noise floor.
                            # Measured over three sessions: p25 held -22.4 to -23.9 dB
                            # while p50 moved -14.9 to -23.2 with the traffic of the room.
MARK_BAND_GUARD_FRAC = 1.0  # the guard around the lock when the floor is measured, in
                            # sample rates. The signal that fills the window continues
                            # past it, thus the reference must start further away.
MARK_BAND_FLOOR_MIN = 64    # fewer bins than this outside the guard measure nothing
MARK_FILL_MARGIN_DB = 10.0  # a bin this far above the band floor holds signal
MARK_FILL_SHARE     = 0.5   # this share of the window above the margin makes it full.
                            # Measured on 280 captures of the room: a WiFi channel of
                            # 20 MHz in the 10 MHz window gives 1.00 and the widest empty
                            # window gives 0.16. See window_filled.

# ── The band plan of 2.4 GHz. It is public and fixed, thus it needs no model. ─────
# Channels 1 to 13 are 2412 to 2472 MHz at a step of 5 MHz, and channel 14 is 2484.
WIFI_CH_HZ = {n: int((2407 + 5 * n) * 1e6) for n in range(1, 14)}
WIFI_CH_HZ[14] = 2_484_000_000
WIFI_TOL_HZ       = 3_000_000   # how near the middle must sit to a channel centre
WIFI_MIN_WIDTH_HZ = 11_000_000  # and how wide it must be. The width is what decides:
                                # the channels are 5 MHz apart, thus every frequency in
                                # the band is within 2.5 MHz of one of them. Measured
                                # 2026-08-14 over a hold of BAND_HOLD_SWEEPS: WiFi
                                # channel 11 read 12.95 to 15.31 MHz and the replayed
                                # drones 9.00 to 9.12 MHz, and both drones sit within
                                # 1.2 MHz of a channel centre. This value is the middle
                                # of those two, thus each has about 1.9 MHz of margin.
BAND_HOLD_SWEEPS  = 4           # sweeps that the band plan holds before it measures a
                                # width. One sweep of WiFi is full of gaps: the run of
                                # bins above the floor read 4.6 to 13.0 MHz across ten
                                # sweeps and reached the limit twice. A hold reads 12.95
                                # to 15.31 MHz. Same reason as the peak hold of the PSD.
BLE_ADV_HZ      = (2_402_000_000, 2_426_000_000, 2_480_000_000)
                                # The three advertising channels of Bluetooth LE, 37, 38
                                # and 39. These are fixed by the specification and every
                                # LE device uses these three and no others to advertise.
                                # They sit in the gaps between WiFi channels 1, 6 and 11
                                # on purpose. The 37 data channels hop and only their
                                # grid is fixed, the even megahertz from 2404 to 2478.
BT_LOW_HZ       = 2_400_000_000
BT_HIGH_HZ      = 2_483_500_000
BT_MAX_WIDTH_HZ = 3_000_000     # classic Bluetooth is 1 MHz and LE is 2 MHz for each
                                # channel, thus 3 MHz covers both with margin. Measured
                                # 0.93 to 1.17 MHz in this room. The drones are 9 MHz,
                                # thus this limit has no way to reach them.
BT_TOL_HZ       = 1_000_000
# The frequencies the model was trained at. A candidate near one of these is never
# walked past, whatever the band plan says, because the program knows it has been
# taught to listen there. The model carries them in its .meta.json when the trainer
# wrote them, and this is the fallback for a model from before that field. Every one
# of the 6,000 captures of this dataset is at 2440 MHz. See near_known_device.
FP_KNOWN_DEVICE_HZ = (2_440_000_000,)
FP_KNOWN_GUARD_HZ  = 6_000_000  # half a receiver window plus the drift of a hopping
                                # link. The AT9S was found 1.85 MHz below its record
                                # frequency and the DJI 1.48 MHz above it.
FP_SKIP_BAND_PLAN = True        # Auto walks past a signal that the band plan names.
                                # The scan exists to find what is not on the raster.
                                # See the defect #37.

# ── The stylesheet of the panel: white text on a dark background. ─────────────
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

# ── Helpers ───────────────────────────────────────────────────────────────────

_VAL_COLOR = "#66ccff"   # the accent color of the values in the status text
_VAL_RE = re.compile(r'([+\-]?\d+\.?\d*\s?(?:GHz|MHz|kHz|dB|ms|hops|files|sweeps?))')


def _hl(x):
    """Put the accent color on a value."""
    return f"<span style='color:{_VAL_COLOR}'>{x}</span>"


def _hl_values(text):
    """Put the accent color on each value that has a unit."""
    return _VAL_RE.sub(lambda mt: _hl(mt.group(1)), text)


def peak_hold_bias_db(n_windows):
    """Give the dB that a peak-hold floor sits above the true mean noise floor.

    Each bin of _peak_hold_psd is the maximum of n_windows samples. For noise those
    samples are exponential, and the median of their maximum is
    -ln(1 - 0.5^(1/n)) times the mean. The value grows with the dwell time: +6.3 dB
    at 48 windows and +9.8 dB at 9765. Without this correction every SNR that the
    program reports moves with the Dwell/hop setting.
    """
    n = max(1, int(n_windows))
    return 10.0 * math.log10(-math.log(1.0 - 0.5 ** (1.0 / n)))


def compute_hop_freqs(center_freq, total_span, hop_bw, overlap_pct=0):
    """Give the LO frequency of each hop of one sweep.

    The first hop is at start + step/2 and not at start + hop_bw/2, because the sweep
    keeps only the central `step` Hz of each hop. With hop_bw/2 the whole band moves
    up by overlap/2, and the low end of the requested span is never received."""
    overlap_hz = int(hop_bw * overlap_pct / 100.0)
    step       = hop_bw - overlap_hz
    n_hops     = math.ceil(total_span / step)
    start      = center_freq - total_span // 2
    return [int(start + step // 2 + i * step) for i in range(n_hops)]


def composite_geometry(cfg):
    """Give the geometry of the sweep composite: (n_keep, f_start, f_stop).

    The FFT of each hop covers sample_rate Hz, but the hops move only `step` Hz.
    Thus the program keeps only the central n_keep bins of each hop. The parts then
    join end to end, and one linear map gives the frequency of each bin. The worker
    and the GUI use the same map.

    A slot is `step` Hz wide, and not n_keep bins wide. n_keep is a whole number of
    bins, thus n_keep bins are usually a few kHz more or less than one step. If the
    slot took that width, each hop boundary moved by that difference and the error
    increased at each hop. A step is exact at every boundary, and the cost stays
    inside one slot and below one bin.
    """
    sr   = float(cfg["sample_rate"])
    hops = cfg["hop_freqs"]
    if len(hops) > 1:
        step = float(hops[1] - hops[0])
    else:
        step = min(sr, float(cfg["rx_bw"])) * (1.0 - cfg["overlap_pct"] / 100.0)
    n_keep = max(2, min(FFT_BINS, int(round(FFT_BINS * step / sr))))
    half   = step / 2.0
    return n_keep, hops[0] - half, hops[-1] + half


def bin_freqs(f0, f1, n):
    """Give the centre frequency of each of `n` bins that tile [f0, f1].

    A bin covers (f1 - f0) / n Hz and the value of the bin belongs at its middle.
    `np.linspace(f0, f1, n)` gives the wrong answer twice: it spaces the points by
    (f1 - f0) / (n - 1), and it puts the first point at the edge of the band and not
    at the middle of the first bin. The error is half a bin, about 10 kHz at the
    defaults. It is below the resolution of the detector, but it is a bias in every
    frequency that the program reports, thus it is in every number the paper prints.
    """
    width = (f1 - f0) / float(n)
    return f0 + (np.arange(n) + 0.5) * width


def hz_to_bin(hz, f0, f1, n):
    """The inverse of bin_freqs: the bin index whose middle is nearest to `hz`."""
    width = (f1 - f0) / float(n)
    return int(round((hz - f0) / width - 0.5))


def signal_extent(freqs, psd, min_snr_db=MARK_MIN_SNR_DB,
                  edge_margin_db=MARK_EDGE_MARGIN_DB, smooth_bins=MARK_SMOOTH_BINS):
    """Find the middle and the edges of the strongest signal in a spectrum.

    The function reads the spectrum only. It does not use the center frequency or a
    bandwidth. Thus the result moves with the signal. The function gives
    (f_left, f_center, f_right). If no signal is above the noise floor, it gives None.

      * The edges are where the smooth spectrum goes below floor + edge_margin_db.
      * The middle is the center of the power between the two edges. This value is
        more stable than the single bin of the peak.
    """
    n = len(psd)
    if n == 0 or len(freqs) != n:
        return None
    sm = np.asarray(psd, dtype=np.float64)
    if smooth_bins > 1 and n >= smooth_bins:
        L   = int(smooth_bins)
        k   = np.ones(L) / float(L)
        pad = L // 2
        # Use the edge value and not zero. The dB values are near -80. Thus a zero
        # increases the last bins and makes a false peak at the ends.
        sm  = np.convolve(np.pad(sm, pad, mode="edge"), k, mode="valid")[:n]
    floor = float(np.median(sm))
    pk    = int(np.argmax(sm))
    if sm[pk] - floor < min_snr_db:            # there is no signal
        return None
    thr = floor + edge_margin_db
    l = pk
    while l > 0 and sm[l - 1] >= thr:
        l -= 1
    r = pk
    while r < n - 1 and sm[r + 1] >= thr:
        r += 1
    lin   = np.power(10.0, np.asarray(psd[l:r + 1], dtype=np.float64) / 10.0)
    fseg  = np.asarray(freqs[l:r + 1], dtype=np.float64)
    denom = float(lin.sum())
    f_center = float((fseg * lin).sum() / denom) if denom > 0 else float(freqs[pk])
    return float(freqs[l]), f_center, float(freqs[r])


def device_votes(result):
    """Give the non-ambient detections of a classifier result, strongest first.

    The votes answer "is a device there", where the mean of the buffer answers "how
    much of this capture is the device". The two disagree for every transmitter that
    stops: a capture of a 26% duty video link is 74% silence, thus the mean reads
    noise while a quarter of the segments name the drone. See the defect #29.
    """
    return [d for d in (result or {}).get("detections", [])
            if d["label"] not in AMBIENT_LABELS]


def device_share(result):
    """Give the largest share of the segments that named a device.

    The badge names a device at MIN_SEG_SHARE. A lock holds below that limit on
    purpose, because the two decisions do not cost the same: to keep a lock costs one
    dwell, and to drop one costs the drone. See the defects #29 and #30.
    """
    shares = (result or {}).get("shares")
    if shares:
        return max((v for k, v in shares.items() if k not in AMBIENT_LABELS),
                   default=0.0)
    return max((d["share"] for d in device_votes(result)), default=0.0)


def has_ambient_class(classes):
    """Say whether a class list holds a background class.

    Auto releases on the sum of AMBIENT_LABELS and p_dev subtracts the same sum, thus
    any one of them is enough. A model with a `wifi` class and no `noise` class
    releases a lock and says 'Clear' correctly, and it must not be warned about.

    The function is separate from the window, thus a check reaches it without Qt.
    """
    return any(str(c).lower() in AMBIENT_LABELS for c in (classes or []))


def badge_for(result, device_thresh=None, second_share=None):
    """Turn a classifier result into the badge text and its style.

    The function is separate from the widget. Thus a check can prove the rule that
    decides what the user reads.

    Presence and identity are two different questions.
      1. p_dev = 1 - sum(P(ambient)) is the mean over the segments of the buffer. It
         answers "how much of this capture is the device", and that is presence only
         for a transmitter that does not stop. The badge says "Background" and not
         "Noise", because the sum covers every class of AMBIENT_LABELS and the model
         can name which one.
      2. The votes of the segments answer presence for every other transmitter. A
         capture of a 26% duty video link is 74% silence, thus the mean reads noise
         while a quarter of the segments name the drone. The votes therefore have
         precedence over the mean, for one name and for two. See the defect #29.
      3. Only above device_thresh does a name from the mean mean something. The badge
         reads "unknown device" when a device is present, no single class holds enough
         of the mean, and no vote names one. The vote comes first, because "unknown"
         is the answer of last resort and a vote is a name. A lower device_thresh sent
         a named capture to "unknown device" while this test ran after the mean.

    The two names do not cost the same. The first name needs MIN_SEG_SHARE, thus a
    bursty link that holds a tenth of its capture is still named. A second name needs
    FP_SECOND_NAME_SHARE, which is higher, because a second name is a claim that
    another transmitter is in the room and the catch-all class of a model makes that
    claim cheaply. See the defect #41.

    The percentage follows the statistic that decided. A badge from the votes gives
    the share of the segments, and a badge from the mean gives the probability.

    Gives (text, style). The style is a key of PlutoApp._BADGE_STYLES.
    """
    if device_thresh is None:
        device_thresh = FP_DEVICE_THRESH
    if second_share is None:
        second_share = FP_SECOND_NAME_SHARE
    probs = result.get("probs", {})
    p_dev = 1.0 - sum(probs.get(k, 0.0) for k in AMBIENT_LABELS)
    best, p_best = max(((c, p) for c, p in probs.items() if c not in AMBIENT_LABELS),
                       key=lambda cp: cp[1],
                       default=(result.get("label", "unknown"),
                                result.get("confidence", 0.0)))
    dets = device_votes(result)
    # device_votes gives the strongest first. The first name keeps its own limit and
    # every name after it must reach second_share, thus one transmitter is named as
    # it always was and a weak vote beside it no longer becomes a second drone.
    named = dets[:1] + [d for d in dets[1:] if d["share"] >= second_share]
    if len(named) >= 2:
        return " + ".join(f"{d['label']} ({d['share']:.0%})" for d in named), "device"
    if p_dev >= device_thresh and p_best >= device_thresh:
        return f"{best} ({p_best:.0%})", "device"
    if dets:
        return f"{dets[0]['label']} ({dets[0]['share']:.0%})", "device"
    if p_dev >= device_thresh:
        return f"Unknown Device ({p_dev:.0%})", "other"
    return f"Clear ({1.0 - p_dev:.0%} Background)", "none"


def signal_clipped(freqs, psd, edge_frac=MARK_CLIP_EDGE_FRAC, **kw):
    """Say whether the signal reaches the edge of the receiver window.

    signal_extent puts an edge where the smooth spectrum falls below the floor. An
    edge that sits at the boundary of the window is not an edge: the spectrum never
    fell, thus the signal continues past the receiver and the plot shows a part of it.
    A part of a signal looks like another signal, thus the user must be told.

    Gives (low, high). Each is True when the signal touches that side of the window.

    The function has one limit and it is not small. signal_extent takes its floor from
    the median of the window, thus a signal that fills more than half of the window
    puts the median inside itself and signal_extent gives None. This function then
    gives (False, False) for the widest signal of all. To answer that case the floor
    must come from outside the window, which means a level that the sweep measured in
    the band around the lock. See §9 Phase 4b task 4 of NOTES.md.
    """
    ext = signal_extent(freqs, psd, **kw)
    if ext is None:
        return False, False
    f_left, _f_mid, f_right = ext
    f0, f1 = float(freqs[0]), float(freqs[-1])
    margin = abs(f1 - f0) * float(edge_frac)
    return f_left <= f0 + margin, f_right >= f1 - margin


def band_floor_db(composite, f0, f1, exclude_hz=None,
                  guard_hz=0.0, pct=MARK_BAND_FLOOR_PCT):
    """Give the noise floor of the swept band, measured away from one place in it.

    The sweep covers the band around the lock, thus it holds a reference that the
    narrowband window can not hold: a level that the signal under test does not
    reach. The guard removes the lock and the part of the band next to it, because a
    signal that fills the receiver window continues past it.

    A hop that gave no data keeps EMPTY_SLOT_DB and it is not a floor.

    Gives dB, or None when too little of the band is left to measure.
    """
    comp = np.asarray(composite, dtype=np.float64)
    total = len(comp)
    if total == 0 or f1 <= f0:
        return None
    ok = comp > EMPTY_SLOT_DB + 1.0
    if exclude_hz is not None and guard_hz > 0:
        s = hz_to_bin(exclude_hz - guard_hz, f0, f1, total)
        e = hz_to_bin(exclude_hz + guard_hz, f0, f1, total) + 1
        ok[max(0, min(total, s)):max(0, min(total, e))] = False
    if int(ok.sum()) < MARK_BAND_FLOOR_MIN:
        return None
    return float(np.percentile(comp[ok], pct))


def window_filled(psd, floor_db, margin_db=MARK_FILL_MARGIN_DB,
                  min_share=MARK_FILL_SHARE):
    """Say whether the receiver window is full of signal from one side to the other.

    signal_extent and signal_clipped both take their floor from the median of the
    window that they read, thus neither can answer this question: a signal that fills
    the window puts the median inside itself. Measured on 120 captures of a 20 MHz
    WiFi channel in the 10 MHz window, signal_extent reported a signal of 3.24 MHz at
    the median and signal_clipped reported no edge at the boundary on any of them.
    The program was confident and wrong, which is worse than the silence that this
    task expected.

    The floor must therefore come from outside the window. That is the defect #27
    again: the reference must come from outside the thing that is measured.

    Measured on 280 captures of the room, at margin 10.0 dB: the full window gives a
    share of 1.00, the part window 0.39 at its largest, and the two empty windows
    0.16 at their largest.
    """
    if floor_db is None:
        return False
    return bool(np.mean(np.asarray(psd, dtype=np.float64)
                        > float(floor_db) + float(margin_db)) >= float(min_share))


def window_state(filled, freqs, psd, **kw):
    """Say how the signal sits in the receiver window, in one short word.

    The answers are "—", "Full window", "Low edge" and "High edge". A full window is
    tested first: both of its edges are outside the receiver, thus signal_clipped can
    not measure them.

    "Both edges" is kept for completeness and it can not happen. signal_extent takes
    its floor from the median of the window, thus its threshold is the median plus
    MARK_EDGE_MARGIN_DB and half of the bins sit below the median by definition. A run
    that reaches both boundaries needs every bin above that threshold. A signal that
    runs off both sides therefore arrives here as a full window, which window_filled
    names from a floor that the sweep measured outside it. See the defect #38.

    The function is separate from the widget, as badge_for is, thus a check can prove
    the rule without Qt.
    """
    if filled:
        return "Full window"
    low, high = signal_clipped(freqs, psd, **kw)
    if low and high:
        return "Both edges"
    if low:
        return "Low edge"
    if high:
        return "High edge"
    return "—"


def occupied_span(composite, f0, f1, at_hz, floor_db,
                  margin_db=MARK_FILL_MARGIN_DB):
    """Give the contiguous part of the swept band around at_hz that holds signal.

    The width of a signal cannot be measured inside the receiver window when it is
    wider than the window, and that is the case the band plan needs most: a WiFi
    channel is 20 MHz and the window is 10 MHz. The sweep already covers the band,
    thus the width comes from the composite and not from the zoom.

    Gives (low_hz, high_hz), or None when the bin at at_hz holds no signal.
    """
    comp = np.asarray(composite, dtype=np.float64)
    total = len(comp)
    if total == 0 or f1 <= f0 or floor_db is None:
        return None
    thr = float(floor_db) + float(margin_db)
    i = hz_to_bin(at_hz, f0, f1, total)
    if i < 0 or i >= total or comp[i] <= thr:
        return None
    lo = i
    while lo > 0 and comp[lo - 1] > thr:
        lo -= 1
    hi = i
    while hi < total - 1 and comp[hi + 1] > thr:
        hi += 1
    edges = bin_freqs(f0, f1, total)
    return float(edges[lo]), float(edges[hi])


def near_known_device(freq_hz, known_hz, guard_hz=FP_KNOWN_GUARD_HZ):
    """Say whether a frequency is one the model was trained at.

    The band plan is a guess about what a signal is. This is not a guess: the training
    captures carry the frequency they were recorded at, thus the program knows exactly
    where it has been taught to listen. A candidate there is never walked past,
    whatever the raster says about it.

    It exists because the raster alone is not safe here. The replay sits at 2440 MHz,
    which puts the AT9S at 2438.15 MHz and the DJI at 2441.48 MHz, and WiFi channels 6
    and 7 are at 2437 and 2442. Measured on 2026-08-14 in a real composite, an AT9S of
    9.12 MHz beside the traffic of the room measured 10.68 to 11.25 MHz, thus it
    crossed the 11 MHz WiFi limit on some sweeps and not on others. A width test alone
    would have walked past the drone that this project exists to find, some of the
    time, and that is the worst possible failure: intermittent.

    A test on how much of the WiFi channel is occupied was tried first and rejected by
    measurement. A drone alone in a channel filled 0.70 to 0.84 of it, because the
    room fills the rest, against 1.00 for real WiFi. That does not separate.
    """
    if not known_hz:
        return False
    f = float(freq_hz)
    return any(abs(f - float(k)) <= float(guard_hz) for k in known_hz)


def band_plan_name(freq_hz, width_hz):
    """Name a signal from the public band plan of 2.4 GHz, with no model at all.

    The raster is fixed and public, thus this answer needs no training data and no
    class. It is a second opinion beside the CNN and never a replacement: §9 Phase 3
    warns that a frequency which follows a class becomes a class cue, and that warning
    is about training. A rule applied to a live lock, after the model has answered,
    contaminates nothing.

    **The width decides, not the frequency.** WiFi channels are 5 MHz apart, thus
    every frequency in the band is within 2.5 MHz of one of them and a rule on the
    centre alone names everything WiFi. Measured on 2026-08-14, the replayed AT9S sat
    at 2438.15 MHz and the DJI at 2441.48 MHz, which are 1.15 MHz from channel 6 and
    0.52 MHz from channel 7. A frequency-only rule would have called both drones WiFi.

    What separates them is that WiFi really is wide: channel 11 measured 13.0 to
    15.3 MHz in this room, against 9.02 and 9.12 MHz for the two replayed drones and
    0.93 to 1.17 MHz for Bluetooth LE.

    Gives a string, or None when the band plan has nothing to say.
    """
    f, wide = float(freq_hz), float(width_hz)
    if wide <= BT_MAX_WIDTH_HZ and BT_LOW_HZ <= f <= BT_HIGH_HZ:
        for adv in BLE_ADV_HZ:
            if abs(f - adv) <= BT_TOL_HZ:
                return "Bluetooth LE advertising"
        return "Bluetooth"
    if wide >= WIFI_MIN_WIDTH_HZ:
        best = min(WIFI_CH_HZ, key=lambda n: abs(f - WIFI_CH_HZ[n]))
        if abs(f - WIFI_CH_HZ[best]) <= WIFI_TOL_HZ:
            return f"WiFi ch {best}"
    return None


def lock_snr_db(psd, bias_db=0.0, floor_db=None):
    """Give the SNR of the held signal, for the question "is it still there".

    The reference is the band floor when the sweep measured one, and the median of
    the window when it did not. The difference is not small and it is the defect #38:
    a signal that fills the receiver window puts the median inside itself, thus the
    program reported the strongest case as the weakest one. Measured on the DJI video
    link on 2026-08-14, which fills the window at 10 Msps: 12.4 dB against the median
    and 32 dB against the band floor, at a release threshold of 18 dB. Auto dropped
    the drone every 2.5 s and locked it again a few megahertz away.

    bias_db corrects the peak-hold floor, see peak_hold_bias_db. Both references are
    peak-hold statistics of the same window count, thus the same correction applies.
    """
    base = float(np.median(np.asarray(psd, dtype=np.float64))) \
        if floor_db is None else float(floor_db)
    return float(np.max(psd)) - base + float(bias_db)


def configure_sdr(sdr, cfg):
    """Push the settings of cfg to the radio.

    The function is separate from the window, thus a check can prove the setup with
    no radio and no Qt. It is the same reason that badge_for is a plain function.

    The count of the kernel buffers is not a detail of the driver. libiio keeps four
    and rx() gives the oldest of them, thus a change of rx_lo appears only in the
    fourth buffer that the program reads, and no wait changes that. Each hop of
    _sweep_once reads one buffer, thus the composite carried the spectrum of the hop
    three places earlier: measured on 2026-08-14, the wideband plot put every signal
    3.00 hops above its true place, which is 16.8 MHz at the default span. See the
    defect #36.
    """
    sdr.sample_rate     = int(cfg["sample_rate"])
    sdr.rx_rf_bandwidth = int(cfg["rx_bw"])
    sdr.rx_lo           = int(cfg["hop_freqs"][0])
    # Use a manual gain. A constant level is necessary for the fingerprints.
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        pass
    sdr.rx_hardwaregain_chan0 = int(cfg["gain"])
    # The count applies when the buffer is made. rx_buffer_size makes a new one, thus
    # the order of the two lines below is not free.
    try:
        sdr._rxadc.set_kernel_buffers_count(RX_KERNEL_BUFFERS)
    except Exception:
        pass                        # an older libiio keeps its own count
    sdr.rx_buffer_size = max(
        1024, int(cfg["sample_rate"] * cfg["dwell_ms"] / 1000.0))


def _write_iq_sidecar(iq_path, device, session, freq, cfg, n_samples, ts):
    """Write the JSON metadata file next to a recorded .iq file."""
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


# ── Sweep worker ──────────────────────────────────────────────────────────────

class SweepWorker(QtCore.QThread):
    sweep_ready       = QtCore.pyqtSignal(object, object)        # composite, hop_bufs
    # freqs, psd, held_freq, the band floor in dB or None when no sweep measured one
    zoom_ready        = QtCore.pyqtSignal(object, object, float, object)
    fingerprint_ready = QtCore.pyqtSignal(object)               # the result dictionary
    mode_changed      = QtCore.pyqtSignal(str, float)           # the mode, held_freq
    caught_changed    = QtCore.pyqtSignal(object)               # the caught frequencies
    hop_progress      = QtCore.pyqtSignal(int, int)
    status_msg        = QtCore.pyqtSignal(str)
    files_changed     = QtCore.pyqtSignal(int, int)             # this run, on the disk

    _BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)

    def __init__(self, sdr, cfg, engine=None):
        super().__init__()
        self.sdr        = sdr
        self.cfg        = cfg
        self.engine     = engine           # a FingerprintModel, or None
        self._stop      = False
        self._fq        = collections.deque()   # the paths of the recorded files
        self._disk_files = None            # the .iq count on the disk. None = not read.
        self._rec_i      = 0               # the counter for record_every_n
        self._seq        = 0               # makes each file name unique
        self._held_freq = None
        self._last_composite = None        # the last full sweep, kept during a lock
        self._sweep_hold = collections.deque(maxlen=BAND_HOLD_SWEEPS)   # for the width
        self._last_present_t = 0.0         # the last time that the signal was present
        self._last_infer_t   = 0.0         # the last time that the classifier ran
        self._caught         = []          # [(freq_hz, t_caught)]
        self._round_done     = False       # the scan has been round, #40
        self._lock_t         = 0.0         # the time when the current lock started
        self._last_class     = None        # the last result from the classifier
        self._last_device_t  = 0.0         # the last time that a vote named a device
        self._cand           = None        # (freq, best_snr_db, hits) of the candidate
        self._psd_bias_db    = 0.0         # the bias of the last peak-hold floor

    def stop(self):
        """Stop the worker. Wait for one hop, then force the thread to terminate.

        The worker tests _stop between two hops only. Thus the wait must cover one
        settle plus one dwell plus the read. A fixed wait of 4 s made a terminate
        certain at each mode change when the dwell was large."""
        self._stop = True
        budget = 2000 + 3 * (int(self.cfg.get("settle_ms", 0))
                             + int(self.cfg.get("dwell_ms", 0)))
        if not self.wait(budget):
            self.status_msg.emit(
                f"Warning: sweep thread blocked on sdr.rx() for {budget} ms "
                f"— forcing terminate."
            )
            self.terminate()
            self.wait(1000)

    # ── The geometry and the DSP ──────────────────────────────────────────────

    def _band_edges(self):
        _n, f0, f1 = composite_geometry(self.cfg)
        return f0, f1

    def _note_sweep(self, composite):
        """Keep the last sweep, and the hold that the band plan measures widths on."""
        self._last_composite = composite
        self._sweep_hold.append(composite)

    def _band_plan_at(self, freq):
        """What the public band plan says about a frequency.

        Gives (name, width_hz), or (None, None). Two things about where the numbers
        come from, and both are the difference between working and not.

        The width comes from the composite and not from the receiver window, because
        the widest service is wider than the window: a WiFi channel is 20 MHz and the
        window is 10 MHz.

        And it comes from a hold of BAND_HOLD_SWEEPS sweeps and not from one. WiFi is
        bursty, thus one sweep of it is full of gaps and occupied_span stops at the
        first one. Measured on ten single sweeps of channel 11: the run of bins read
        4.6 to 13.0 MHz and passed the 11 MHz limit twice. The same channel held over
        four sweeps reads 12.95 to 15.31 MHz, every time.
        """
        if freq is None or not self._sweep_hold:
            return None, None
        held = self._sweep_hold[0] if len(self._sweep_hold) == 1 else \
            np.maximum.reduce(list(self._sweep_hold))
        f0, f1 = self._band_edges()
        floor = band_floor_db(held, f0, f1, float(freq),
                              float(self.cfg["sample_rate"]) * MARK_BAND_GUARD_FRAC)
        if floor is None:
            return None, None
        span = occupied_span(held, f0, f1, float(freq), floor)
        if span is None:
            return None, None
        lo, hi = span
        return band_plan_name((lo + hi) / 2.0, hi - lo), (hi - lo)

    def _known_freqs(self):
        """The frequencies the model was trained at, from the model or the constant."""
        freqs = getattr(self.engine, "train_freqs", None)
        return freqs if freqs else FP_KNOWN_DEVICE_HZ

    def _band_floor(self):
        """The noise floor of the band around the lock, from the last full sweep.

        It is the reference that the narrowband window can not hold itself. Gives
        None before the first sweep, and in a mode that does not sweep."""
        if self._last_composite is None or self._held_freq is None:
            return None
        f0, f1 = self._band_edges()
        guard = float(self.cfg["sample_rate"]) * MARK_BAND_GUARD_FRAC
        return band_floor_db(self._last_composite, f0, f1,
                             float(self._held_freq), guard)

    def _sweep_once(self):
        """Do one sweep of the full band. Gives (composite, hop_bufs)."""
        hop_freqs  = self.cfg["hop_freqs"]
        n_hops     = len(hop_freqs)
        n_keep, _f0, _f1 = composite_geometry(self.cfg)
        b0         = (FFT_BINS - n_keep) // 2       # the central part of each hop
        composite  = np.full(n_hops * n_keep, EMPTY_SLOT_DB, dtype=np.float32)
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
            psd = self._peak_hold_psd(raw)
            composite[i * n_keep:(i + 1) * n_keep] = psd[b0:b0 + n_keep]
        return composite, hop_bufs

    def _peak_hold_psd(self, iq):
        """Make a 1024-bin dB spectrum and keep the maximum of all the windows.

        The unit is dB and not dBFS. The value is 20*log10(|FFT(x*w)| / N) on the
        scale that pyadi-iio gives, with no correction for the coherent gain of the
        Blackman window, which is about -7.5 dB. Thus the number has no reference to
        the full scale of the converter. Nothing in the program is wrong because of
        it, because every decision compares two such numbers and the offset cancels.
        A report that prints the axis must say this, or must calibrate first.

        The windows touch each other and they cover the full buffer, at every dwell
        time. Thus a short burst at any time in the buffer is visible at its true
        amplitude. One window only would not find the burst.

        The program does the work in blocks of PSD_CHUNK_WINS windows and it keeps a
        running maximum. Thus the memory stays the same at a long dwell, and no part
        of the buffer is lost."""
        n = len(iq)
        if n < FFT_BINS:
            iq = np.pad(iq, (0, FFT_BINS - n)); n = FFT_BINS
        nwin = n // FFT_BINS
        self._psd_bias_db = peak_hold_bias_db(nwin)
        # Remove a mean for each window, not one mean for the buffer. The artifact of
        # the receiver near 0 Hz is not a constant, thus one mean removes nothing:
        # measured on 2026-08-10, the 0 Hz bin stood 9 to 39 dB above the median with
        # the old code and the value did not move by 0.05 dB when it ran. A mean for
        # each window takes 2437 MHz at gain 0 from 9.46 dB to 0.71 dB.
        peak = None
        for a in range(0, nwin, PSD_CHUNK_WINS):
            b   = min(a + PSD_CHUNK_WINS, nwin)
            seg = iq[a * FFT_BINS:b * FFT_BINS].reshape(b - a, FFT_BINS)
            seg = seg - seg.mean(axis=1, keepdims=True)
            mag = np.abs(np.fft.fftshift(np.fft.fft(seg * self._BLACKMAN, axis=1),
                                         axes=1)).max(axis=0)
            peak = mag if peak is None else np.maximum(peak, mag)
        return (20.0 * np.log10(peak / FFT_BINS + 1e-10)).astype(np.float32)

    def _narrowband_psd(self, iq, center):
        """Give the spectrum of a held capture on an absolute frequency axis."""
        psd = self._peak_hold_psd(iq)
        sr = self.cfg["sample_rate"]
        freqs = bin_freqs(center - sr / 2, center + sr / 2, FFT_BINS)
        return freqs, psd

    def _composite_with_hold(self, held_freq, psd):
        """Put the live spectrum of the held band into the last full sweep. Thus the
        wideband plot keeps the full band during a lock."""
        base = self._last_composite
        if base is None:                       # there is no sweep yet: make a floor
            n_keep, _f0, _f1 = composite_geometry(self.cfg)
            base = np.full(len(self.cfg["hop_freqs"]) * n_keep, EMPTY_SLOT_DB,
                           dtype=np.float32)
        comp = base.copy()
        f_min, f_max = self._band_edges()
        span = f_max - f_min
        if span <= 0:
            return comp
        total = len(comp)
        sr = self.cfg["sample_rate"]
        s = hz_to_bin(held_freq - sr / 2, f_min, f_max, total)
        e = hz_to_bin(held_freq + sr / 2, f_min, f_max, total) + 1
        s = max(0, min(total, s)); e = max(0, min(total, e))
        if e - s >= 2:
            comp[s:e] = np.interp(np.linspace(0.0, 1.0, e - s),
                                  np.linspace(0.0, 1.0, len(psd)),
                                  psd).astype(np.float32)
        return comp

    def _detect_new_peak(self, composite):
        """Find the strongest peak that is not in the memory of the caught signals.

        The floor is the median of the slots that received data. A hop that failed
        keeps EMPTY_SLOT_DB. If those slots stay in the median, the floor falls,
        and each real bin then looks like a large peak.

        peak_hold_bias_db corrects the median. Thus the value that comes back is a
        true SNR and it does not change with the dwell time."""
        f_min, f_max = self._band_edges()
        span  = f_max - f_min
        total = len(composite)
        filled = composite > EMPTY_SLOT_DB + 1.0
        if not filled.any():
            return None, -999.0
        med   = float(np.median(composite[filled])) - self._psd_bias_db
        masked = composite.copy()
        guard = float(self.cfg.get("fp_memory_guard_hz", FP_MEMORY_GUARD_HZ))
        for cf, _t in self._caught:
            s = hz_to_bin(cf - guard, f_min, f_max, total)
            e = hz_to_bin(cf + guard, f_min, f_max, total) + 1
            s = max(0, min(total, s)); e = max(0, min(total, e))
            if e > s:
                masked[s:e] = -1e9
        # A candidate must hold its place. One sweep is 50 ms for each hop, thus a
        # burst that lands in one sweep and not in the next is not a transmitter that
        # the program can follow. The candidate of the last sweeps must repeat before
        # it causes a lock, and a new candidate must be near the old one to count as
        # the same signal. Without this the lock moves to the side whenever another
        # emitter is loud for one sweep. See the defect #31.
        idx = int(np.argmax(masked))
        if masked[idx] <= -1e8:                     # all the band is in the memory
            self._cand = None
            self._round_done = True
            return None, -999.0
        freq = float(bin_freqs(f_min, f_max, total)[idx])
        snr  = float(composite[idx]) - med
        # The round is over when the loudest thing left is not worth a lock. That is
        # the moment every candidate of this band has had one turn, and it is what
        # clears the memory. See _prune_memory and the defect #40.
        self._round_done = snr < float(
            self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB))
        need = int(self.cfg.get("peak_hits", FP_PEAK_HITS))
        guard = float(self.cfg.get("fp_memory_guard_hz", FP_MEMORY_GUARD_HZ))
        if self._cand is not None and abs(self._cand[0] - freq) <= guard:
            # The same signal. Keep the strongest reading of it and count the hit.
            best = max(self._cand[1], snr)
            hits = self._cand[2] + 1
            self._cand = (freq, best, hits)
        else:
            self._cand = (freq, snr, 1)
        if self._cand[2] < need:
            return None, -999.0
        return self._cand[0], self._cand[1]

    def _refine_lock(self, freq):
        """Move a new lock to the middle of the signal, one time only.

        The peak search gives the strongest bin of the composite, and a bin of the
        composite is wide. The middle of the signal is not that bin: a link is not
        symmetric about its strongest bin, and the narrowband image that the
        classifier reads is cut around the tuned frequency.

        The program therefore tunes to the candidate, reads one buffer, and takes the
        centroid that signal_extent gives. The centroid is used and not the argmax,
        because the argmax jitters from bin to bin while the centroid does not.

        This runs one time, at the acquisition, and never during the hold. Section 4
        says the lock frequency does not move, because a lock that drifts makes the
        narrowband plot and the recorded captures inconsistent. A refinement before
        the first capture keeps that rule and still centres the signal.

        Gives the new frequency, or the old one when there is nothing to centre on.
        """
        sr = float(self.cfg["sample_rate"])
        limit = sr * FP_REFINE_MAX_FRAC
        try:
            self.sdr.rx_lo = int(freq)
            self.msleep(int(self.cfg.get("settle_ms", HOP_SETTLE_MS)))
            iq = self.sdr.rx()
        except Exception:
            return freq                      # keep the candidate, see the tune error
        psd = self._peak_hold_psd(iq)
        freqs = bin_freqs(freq - sr / 2, freq + sr / 2, len(psd))
        ext = signal_extent(freqs, psd)
        if ext is None:
            return freq
        _l, f_mid, _r = ext
        if abs(f_mid - freq) > limit:
            # A middle far from the candidate is another signal in the same window,
            # not the same one. Keep the candidate.
            return freq
        return float(f_mid)

    def _remember(self, freq):
        """Put a caught frequency in the memory. Remove the near duplicates first."""
        guard = float(self.cfg.get("fp_memory_guard_hz", FP_MEMORY_GUARD_HZ))
        self._caught = [(f, t) for (f, t) in self._caught if abs(f - freq) > guard]
        self._caught.append((float(freq), time.time()))
        self.caught_changed.emit([f for f, _t in self._caught])

    def _release_lock(self, msg):
        """Release the lock, put the frequency in the memory, and go back to SCAN.
        The caller must also set its local `mode` to "SCAN"."""
        self._remember(self._held_freq)
        self._held_freq = None
        self.mode_changed.emit("SCAN", 0.0)
        self.status_msg.emit(msg)

    def _prune_memory(self):
        """Clear the memory when the scan has been round once, not on a wall clock.

        A wall clock alone is not fair and it loses the drone. The search takes the
        loudest thing that the memory does not mask, thus the order of the search is
        the order of the levels in the room. Each candidate costs a dwell to judge.
        With five emitters above the drone that is about 30 s of judging, and a
        30 s timer returns the first of them to the search at the moment the drone
        would have had its turn. The scan then cycles the loud background for ever
        and never offers the drone to the classifier at all. Measured on 2026-08-18
        over 119.3 s with the drone above the threshold on 84 of 84 sweeps. See the
        defect #40.

        The memory therefore holds a frequency until nothing above the threshold is
        left unmasked, which _detect_new_peak reports as `_round_done`. Every
        candidate has had exactly one turn at that moment, thus clearing is fair.

        The wall clock stays as a safety valve. A round that never ends, because the
        room keeps making new emitters, must not blind the scanner for ever.
        """
        if self._round_done:
            self._round_done = False
            if self._caught:
                self._caught = []
                self.caught_changed.emit([])
            return
        ttl = float(self.cfg.get("fp_memory_ttl_s", FP_MEMORY_TTL_S))
        max_age = ttl * float(self.cfg.get("fp_memory_ttl_rounds",
                                           FP_MEMORY_TTL_ROUNDS))
        now = time.time()
        kept = [(f, t) for (f, t) in self._caught if now - t < max_age]
        if len(kept) != len(self._caught):
            self._caught = kept
            self.caught_changed.emit([f for f, _t in self._caught])

    def _count_on_disk(self):
        """Count the .iq files that fingerprint_data already holds.

        The ring removes the files of this worker only. Thus the number on the disk
        is larger, and it is the number that tells you the space that you use."""
        try:
            return sum(1 for _ in Path(RECORD_DIR).rglob("*.iq"))
        except OSError:
            return 0

    def _record_step(self):
        """Give True when this buffer or this sweep must go to the disk.

        The narrowband record writes one file for each buffer, which is one file
        each dwell_ms. At the default that is 20 files and 80 MB each second. The
        divisor makes the same quantity of files cover more time, and it gives more
        different data."""
        n = max(1, int(self.cfg.get("record_every_n", RECORD_EVERY_N)))
        take = (self._rec_i % n) == 0
        self._rec_i += 1
        return take

    def _save_iq(self, iq, device, session, freq):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(device)).strip("._") or "device"
        d = os.path.join(RECORD_DIR, safe, f"session_{session}")
        os.makedirs(d, exist_ok=True)
        if self._disk_files is None:            # one scan, before the first write
            self._disk_files = self._count_on_disk()
        ts = int(time.time() * 1000)
        # A sweep writes every hop in one burst, thus the millisecond is not unique.
        # The sequence number makes the name unique, and the loop covers a restart.
        while True:
            fpath = os.path.join(d, f"{safe}_s{session}_{ts}_{self._seq:05d}.iq")
            self._seq += 1
            if not os.path.exists(fpath):
                break
        iq.astype(np.complex64).tofile(fpath)
        _write_iq_sidecar(fpath, safe, session, freq, self.cfg, len(iq), ts)
        self._disk_files += 1
        self._fq.append(fpath)
        max_files = int(self.cfg.get("record_max_files", 1000))
        while len(self._fq) > max_files:
            old = self._fq.popleft()
            for p in (old, os.path.splitext(old)[0] + ".json"):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                        if p.endswith(".iq"):
                            self._disk_files -= 1
                    except OSError as e:
                        self.status_msg.emit(f"Warning: could not remove {p}: {e}")
        self.files_changed.emit(len(self._fq), self._disk_files)

    def _maybe_classify(self, iq):
        """Run the classifier, but not more than one time each ml_interval_s.

        The identity of a signal does not change quickly. Thus a limit on the runs
        keeps the plots and the markers fast."""
        if self.engine is None or not self.cfg.get("ml_enabled", True):
            return
        now = time.time()
        if now - self._last_infer_t < float(self.cfg.get("ml_interval_s", ML_INTERVAL_S)):
            return
        self._last_infer_t = now
        try:
            res = self.engine.classify_iq(iq)
            # The band plan travels with the result as a second opinion. badge_for
            # never sees it, thus the rule that decides the name stays one rule.
            plan, _w = self._band_plan_at(self._held_freq)
            res["band_plan"] = plan
            self._last_class = res          # Auto uses this to judge the lock
            if device_share(res) >= float(self.cfg.get("hold_share", FP_HOLD_SHARE)):
                # The time of the last device, not of the last buffer that held one.
                # A hopping link is silent between its bursts. See the defect #30.
                self._last_device_t = now
            self.fingerprint_ready.emit(res)
        except Exception as e:
            self.status_msg.emit(f"Inference error: {e}")

    @staticmethod
    def _ambient_prob(res):
        """Give the summed probability of the background classes. Auto releases a lock
        on it, thus a confident WiFi lock releases as a noise lock does. If the model
        has none of them, or if the classifier did not run, the value is 0.0."""
        probs = (res or {}).get("probs", {})
        return float(sum(probs.get(k, 0.0) for k in AMBIENT_LABELS))

    # ── The modes ─────────────────────────────────────────────────────────────

    def run(self):
        op = self.cfg.get("op_mode", "locking")
        if op == "wideband":
            self._run_wideband()
        elif op == "focus":
            self._run_focus()
        else:                       # Locking and Auto use the same loop
            self._run_locking()

    def _run_locking(self):
        """Sweep, lock the strongest new signal, and hold that frequency.

        The frequency does not move during the hold. Thus the narrowband plot is
        stable. The hold continues until the user clicks Skip, or until the signal
        stops. Then the program puts the frequency in the memory and continues the
        sweep. The memory clears when the scan has been round once, thus every signal
        of the band gets one turn before any of them gets a second. See #40."""
        mode = "SCAN"
        self.cfg["skip_lock"] = False
        self.cfg["jump_to"] = None
        self.mode_changed.emit("SCAN", 0.0)
        while not self._stop:
            jt = self.cfg.get("jump_to")
            if jt:                              # go to a frequency that the user chose
                self.cfg["jump_to"] = None
                self._held_freq, mode = int(jt), "LOCK"
                self._last_present_t = self._lock_t = time.time()
                self._last_infer_t = 0.0        # classify the new lock now
                self._last_class = None         # Auto: use only the new results
                self._last_device_t = 0.0       # no device has been seen on this lock
                self.mode_changed.emit("LOCK", float(jt))
                self.status_msg.emit(f"Jumped to {jt/1e6:.3f} MHz — Skip to advance")
            if mode == "SCAN":
                t0 = time.perf_counter()
                composite, hop_bufs = self._sweep_once()
                if composite is None:
                    return
                self._note_sweep(composite)
                self.sweep_ready.emit(composite, hop_bufs)
                self.status_msg.emit(
                    f"Scan: {(time.perf_counter()-t0)*1000:.0f} ms  |  "
                    f"{len(self.cfg['hop_freqs'])} hops")
                self._prune_memory()
                f, peak_db = self._detect_new_peak(composite)
                if f is not None and peak_db >= float(
                        self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB)):
                    # Walk past what the band plan explains. A scan exists to find what
                    # is not on the public raster, and a WiFi channel that the model
                    # calls a drone otherwise holds the lock for ever. See #37.
                    #
                    # A frequency the model was trained at is never walked past. The
                    # raster cannot be trusted there: the replay sits between WiFi
                    # channels 6 and 7, and a drone beside the traffic of the room
                    # measured 10.68 to 11.25 MHz against an 11 MHz limit, thus the
                    # width test would drop the drone on some sweeps and not others.
                    skip = (self.cfg.get("op_mode") == "auto"
                            and self.cfg.get("skip_band_plan", FP_SKIP_BAND_PLAN)
                            and not near_known_device(
                                f, self._known_freqs(),
                                self.cfg.get("known_guard_hz", FP_KNOWN_GUARD_HZ)))
                    plan, wide = self._band_plan_at(f) if skip else (None, None)
                    if plan is not None:
                        self._remember(f)
                        self._cand = None
                        self.status_msg.emit(
                            f"Passed {plan} @ {f/1e6:.3f} MHz, "
                            f"{wide/1e6:.1f} MHz wide — still scanning")
                        continue
                    if self.cfg.get("refine_lock", FP_REFINE_LOCK):
                        f = self._refine_lock(f)
                    self._held_freq, mode = f, "LOCK"
                    self._last_present_t = self._lock_t = time.time()
                    self._last_infer_t = 0.0    # classify the new lock now
                    self._last_class = None     # Auto: use only the new results
                    self._last_device_t = 0.0   # no device has been seen on this lock
                    self._cand = None           # the candidate became a lock
                    self.mode_changed.emit("LOCK", f)
                    self.status_msg.emit(
                        f"Locked +{peak_db:.0f} dB @ {f/1e6:.3f} MHz — Skip to advance")
            else:  # LOCK: hold the frequency
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
                self.zoom_ready.emit(freqs, psd, float(self._held_freq),
                                     self._band_floor())
                comp = self._composite_with_hold(self._held_freq, psd)
                if comp is not None:
                    self.sweep_ready.emit(comp, {})
                self._maybe_classify(iq)
                # Auto: after the dwell time, release the lock if the classifier calls
                # the signal noise. The program keeps a lock on a true device.
                if self.cfg.get("op_mode") == "auto":
                    dwell = float(self.cfg.get("auto_dwell_ms", FP_AUTO_DWELL_MS)) / 1000.0
                    thr   = float(self.cfg.get("auto_noise_pct", FP_AUTO_NOISE_PCT)) / 100.0
                    ambient_p = self._ambient_prob(self._last_class)
                    # A device that was seen recently holds the lock against the mean.
                    # A bursty link puts most of its capture below the noise floor and
                    # is silent between its bursts, thus the current buffer alone
                    # releases a drone that is still there. See the defects #29 and #30.
                    hold = float(self.cfg.get("device_hold_s", FP_DEVICE_HOLD_S))
                    if (self._last_class is not None
                            and time.time() - self._last_device_t >= hold
                            and time.time() - self._lock_t >= dwell
                            and ambient_p >= thr):
                        self._release_lock(
                            f"Auto-skip: background {ambient_p:.0%} ≥ {thr:.0%} "
                            f"after {dwell*1000:.0f} ms")
                        mode = "SCAN"
                        continue
                # Release the lock if the signal is absent for the gone time.
                thresh = float(self.cfg.get("fp_peak_thresh_db", FP_PEAK_THRESH_DB))
                gone = float(self.cfg.get("fp_gone_s", FP_GONE_S))
                snr = lock_snr_db(psd, self._psd_bias_db, self._band_floor())
                if snr >= thresh:
                    self._last_present_t = time.time()
                elif time.time() - self._last_present_t >= gone:
                    self._release_lock("Signal gone — moving on")
                    mode = "SCAN"
                    continue
                self.status_msg.emit(
                    f"Locked @ {self._held_freq/1e6:.3f} MHz — Skip to advance")

    def _run_wideband(self):
        """Sweep the full band continuously. There is no lock.

        If the record is on, the program saves the raw IQ data of every
        NOISE_REC_EVERY_N-th sweep to the noise class."""
        self.mode_changed.emit("WIDEBAND", 0.0)
        while not self._stop:
            t0 = time.perf_counter()
            composite, hop_bufs = self._sweep_once()
            if composite is None:
                return
            self._note_sweep(composite)
            self.sweep_ready.emit(composite, hop_bufs)
            rec = (bool(self.cfg.get("record"))
                   and self.cfg.get("record_kind") == "noise_band")
            every = max(1, int(self.cfg.get("record_every_n", RECORD_EVERY_N)))
            if rec and self._record_step():
                session = self.cfg.get("record_session", "1")
                for i, raw in hop_bufs.items():
                    if len(raw):
                        self._save_iq(raw, "noise", session,
                                      int(self.cfg["hop_freqs"][i]))
            el  = (time.perf_counter() - t0) * 1000
            tag = (f"  |  REC noise: {len(self._fq)} files "
                   f"(1/{every} sweeps)" if rec else "")
            self.status_msg.emit(
                f"Wideband: {el:.0f} ms  |  {len(self.cfg['hop_freqs'])} hops{tag}")

    def _run_focus(self):
        """Hold one frequency (focus_freq) and show the narrowband plot only.

        The program classifies the signal if a model is available. If the record is
        on, the program saves the IQ data. This mode makes the device recordings."""
        freq = int(self.cfg.get("focus_freq", self.cfg["center_freq"]))
        try:
            self.sdr.rx_lo = freq
        except Exception as e:
            self.status_msg.emit(f"Focus tune error: {e}"); return
        self.mode_changed.emit("FOCUS", float(freq))
        time.sleep(self.cfg.get("fp_hold_settle_ms", FP_HOLD_SETTLE_MS) / 1000.0)
        self._last_infer_t = 0.0        # classify the held signal now
        while not self._stop:
            nf = int(self.cfg.get("focus_freq", freq))      # the user can change this
            if nf != freq:
                try:
                    self.sdr.rx_lo = nf
                except Exception as e:
                    # Keep the old frequency. If the program keeps nf, the captures
                    # get a frequency that the radio did not tune to.
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
            # No band floor here: this mode never sweeps, thus nothing measured the
            # band around the frequency. A window full of signal is not reported in
            # Narrowband. See §9 Phase 4b task 4 of NOTES.md.
            self.zoom_ready.emit(freqs, psd, float(freq), None)
            self._maybe_classify(iq)
            kind = self.cfg.get("record_kind", "device")
            if self.cfg.get("record") and kind in ("device", "noise_freq"):
                # noise_freq saves to the noise class at the frequency of the device
                label = "noise" if kind == "noise_freq" else \
                    self.cfg.get("record_device", "deviceA")
                every = max(1, int(self.cfg.get("record_every_n", RECORD_EVERY_N)))
                if self._record_step():
                    self._save_iq(iq, label, self.cfg.get("record_session", "1"), freq)
                self.status_msg.emit(
                    f"{'NOISE' if kind == 'noise_freq' else 'DEVICE'} REC "
                    f"{label}/s{self.cfg.get('record_session','1')} @ "
                    f"{freq/1e6:.3f} MHz: {len(self._fq)} files (1/{every} buffers)")
            else:
                self.status_msg.emit(f"Focus @ {freq/1e6:.3f} MHz")


# ── Main application ──────────────────────────────────────────────────────────

class PlutoApp(QtWidgets.QMainWindow):

    def __init__(self, connect=True):
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
            # The worker reads these keys live. A change is effective immediately.
            "op_mode"             : "auto",       # auto | locking | wideband | focus
            "ml_enabled"          : True,         # run the classifier
            "ml_interval_s"       : ML_INTERVAL_S,
            "record"              : False,
            "record_kind"         : "device",     # device | noise_band | noise_freq
            "skip_lock"           : False,        # go to the next signal
            "jump_to"             : None,         # lock on this frequency now
            "auto_dwell_ms"       : FP_AUTO_DWELL_MS,
            "auto_noise_pct"      : FP_AUTO_NOISE_PCT,
            "record_device"       : "deviceA",
            "record_session"      : "1",
            "focus_freq"          : CENTER_FREQ,
            "record_max_files"    : 1000,
            "record_every_n"      : RECORD_EVERY_N,
            "fp_peak_thresh_db"   : FP_PEAK_THRESH_DB,
            "fp_hold_settle_ms"   : FP_HOLD_SETTLE_MS,
            "fp_gone_s"           : FP_GONE_S,
            "fp_memory_ttl_s"     : FP_MEMORY_TTL_S,
            "fp_memory_guard_hz"  : FP_MEMORY_GUARD_HZ,
        }
        self._recompute_hops()

        self._engine      = None

        self._build_ui()

        if os.path.exists(DEFAULT_MODEL_PATH):
            self.w_model_path.setText(DEFAULT_MODEL_PATH)
            self._load_model(DEFAULT_MODEL_PATH)

        self.wf_data = self._make_waterfall_buf()
        self.img.setImage(self.wf_data, autoLevels=False)
        self._update_waterfall_rect()

        self.sdr = None
        if connect:
            self.connect_sdr()

    # ── The SDR and the hops ──────────────────────────────────────────────────

    def connect_sdr(self):
        """Open the radio and start the sweep. Give True if the radio answers.

        The construction and the connection are two operations, because a check must
        build the window with no radio and then drive a handler with an array from a
        file. `PlutoApp()` connects, `PlutoApp(connect=False)` does not, thus nothing
        changes for the user."""
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
            self.sdr = None
            return False
        self._start_worker()
        return True

    def _recompute_hops(self):
        effective_bw = min(self.cfg["sample_rate"], self.cfg["rx_bw"])
        self.cfg["hop_freqs"] = compute_hop_freqs(
            self.cfg["center_freq"], self.cfg["total_span"],
            effective_bw, self.cfg["overlap_pct"])
        self.n_hops        = len(self.cfg["hop_freqs"])
        # The axis and the waterfall must cover the same range as the composite.
        n_keep, f0, f1     = composite_geometry(self.cfg)
        self.total_bins    = self.n_hops * n_keep
        self.f_global_min  = f0
        self.f_global_max  = f1
        self._slot_hz      = (f1 - f0) / self.n_hops
        self._effective_bw = effective_bw

    def _push_sdr_settings(self):
        configure_sdr(self.sdr, self.cfg)

    def _make_waterfall_buf(self):
        return np.full((self.total_bins, WATERFALL_ROWS),
                       WF_SCALE_MIN_DB, dtype=np.float32)

    # ── The user interface ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self.win = pg.GraphicsLayoutWidget()
        root.addWidget(self.win, stretch=5)
        _axis_w = 64   # one width for all the left axes. Thus the plots are aligned.

        # ── The wideband spectrum ───────────────────────────────────────────────
        self.p1 = self.win.addPlot(row=0, col=0, title="Wideband Spectrum")
        self.p1.setLabel("bottom", "Frequency", units="Hz")
        self.p1.setLabel("left",   "Power",     units="dB")
        # Stop the automatic range. If not, each sweep changes the view.
        self.p1.enableAutoRange(x=False, y=False)
        self.p1.setXRange(self.f_global_min, self.f_global_max, padding=0)
        self.p1.setYRange(WF_SCALE_MIN_DB, WF_SCALE_MAX_DB, padding=0)
        self.p1.setMouseEnabled(x=True, y=True)
        self.p1.setMenuEnabled(False)
        self.p1.showGrid(x=True, y=True, alpha=0.25)
        self.p1.getAxis("left").setWidth(_axis_w)
        self.hop_lines = []
        self._rebuild_hop_lines()
        self.curve = self.p1.plot(pen=pg.mkPen("y", width=1))

        # ── The wideband waterfall ──────────────────────────────────────────────
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
        self.img.setLevels([WF_SCALE_MIN_DB, WF_SCALE_MAX_DB])

        # ── The narrowband spectrum of the held signal ──────────────────────────
        self.p_zoom = self.win.addPlot(row=0, col=1, title="Narrowband Spectrum")
        self.p_zoom.setLabel("bottom", "Frequency", units="Hz")
        self.p_zoom.setLabel("left",   "Power",     units="dB")
        self.p_zoom.setMouseEnabled(x=True, y=True)
        self.p_zoom.setMenuEnabled(True)
        self.p_zoom.showGrid(x=True, y=True, alpha=0.25)
        self.p_zoom.setYRange(WF_SCALE_MIN_DB, WF_SCALE_MAX_DB, padding=0)
        self.p_zoom.getAxis("left").setWidth(_axis_w)
        self.zoom_curve = self.p_zoom.plot(pen=pg.mkPen("c", width=1))

        # The markers: a red line at the middle of the signal and two dashed red
        # lines at the edges. signal_extent() gives the positions for each frame.
        _mid_pen  = pg.mkPen(color=(255, 40, 40), width=2)
        _edge_pen = pg.mkPen(color=(255, 40, 40), width=1, style=QtCore.Qt.DashLine)
        self.mid_line   = pg.InfiniteLine(angle=90, movable=False, pen=_mid_pen)
        self.edge_lines = [pg.InfiniteLine(angle=90, movable=False, pen=_edge_pen)
                           for _ in range(2)]
        for _ln in (self.mid_line, *self.edge_lines):
            _ln.setVisible(False)
            _ln.setZValue(10)                 # keep the markers above the curve
            self.p_zoom.addItem(_ln, ignoreBounds=True)
        self._last_zoom = None            # the last (freqs, psd, filled) for the markers

        # ── The narrowband waterfall ────────────────────────────────────────────
        self.p_zoom_wf = self.win.addPlot(row=1, col=1, title="Narrowband Waterfall")
        self.p_zoom_wf.setLabel("bottom", "Frequency", units="Hz")
        self.p_zoom_wf.setLabel("left",   "Time",      units="holds")
        self.p_zoom_wf.setXLink(self.p_zoom)
        self.p_zoom_wf.setMouseEnabled(x=True, y=False)
        self.p_zoom_wf.setMenuEnabled(True)
        self.p_zoom_wf.getViewBox().setAutoVisible(x=False, y=False)
        self.p_zoom_wf.enableAutoRange(x=False, y=False)
        self.p_zoom_wf.getAxis("left").setWidth(_axis_w)
        self.zoom_wf_img = pg.ImageItem(axisOrder="col-major")
        self.p_zoom_wf.addItem(self.zoom_wf_img)
        self.zoom_wf_img.setLookupTable(cmap.getLookupTable())
        self.zoom_wf_img.setLevels([WF_SCALE_MIN_DB, WF_SCALE_MAX_DB])
        self.zoom_wf_data = np.full((FFT_BINS, WATERFALL_ROWS),
                                    WF_SCALE_MIN_DB, dtype=np.float32)
        self.zoom_wf_img.setImage(self.zoom_wf_data, autoLevels=False)
        _zc, _zsr = self.cfg["center_freq"], self.cfg["sample_rate"]
        self.zoom_wf_img.setRect(QtCore.QRectF(_zc - _zsr / 2, 0, _zsr, WATERFALL_ROWS))
        self.p_zoom.setXRange(_zc - _zsr / 2, _zc + _zsr / 2, padding=0)
        self.p_zoom_wf.setYRange(0, WATERFALL_ROWS, padding=0)

        # Make the wideband column wider than the narrowband column.
        try:
            self.win.ci.layout.setColumnStretchFactor(0, 2)
            self.win.ci.layout.setColumnStretchFactor(1, 1)
        except Exception:
            pass

        # ── The panel on the right side ────────────────────────────────────────
        panel = QtWidgets.QWidget()
        # One stylesheet for the full panel. Thus each widget in the panel has the
        # correct colors, and no widget needs its own setStyleSheet call.
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

        # ── The ML inference panel ─────────────────────────────────────────────
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
        # The switch for the classifier. If it is off, the worker only reads the data
        # and shows it. This makes the program much faster.
        self.w_ml_toggle = QtWidgets.QPushButton("ML Inference: ON")
        self.w_ml_toggle.setCheckable(True)
        self.w_ml_toggle.setChecked(self.cfg.get("ml_enabled", True))
        self.w_ml_toggle.setFixedHeight(28)
        self.w_ml_toggle.toggled.connect(self._on_ml_toggle)
        load_row.addWidget(load_btn, 1)
        load_row.addWidget(self.w_ml_toggle)
        vbox.addLayout(load_row)

        # ── The mode panel ─────────────────────────────────────────────────────
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

        # These two buttons are for Auto and Locking: skip the lock, or go to a
        # frequency in the caught list.
        skip_row = QtWidgets.QHBoxLayout()
        self.w_skip_btn = QtWidgets.QPushButton("Skip lock")
        self.w_skip_btn.setFixedHeight(26)
        self.w_skip_btn.setEnabled(True)         # Auto is the default mode
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

        # The parameters of the Auto mode: the dwell time on each lock, and the
        # probability of noise that releases the lock.
        auto_row = QtWidgets.QHBoxLayout()
        auto_cap = QtWidgets.QLabel("Auto skip:")
        auto_cap.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_auto_dwell = QtWidgets.QSpinBox()
        self.w_auto_dwell.setRange(100, 60000)
        self.w_auto_dwell.setSingleStep(100)
        self.w_auto_dwell.setSuffix(" ms")
        self.w_auto_dwell.setValue(FP_AUTO_DWELL_MS)
        self.w_auto_dwell.setToolTip("Hold each lock this long before you judge it")
        self.w_auto_noise = QtWidgets.QSpinBox()
        self.w_auto_noise.setRange(1, 100)
        self.w_auto_noise.setSuffix(" % background")
        self.w_auto_noise.setValue(FP_AUTO_NOISE_PCT)
        self.w_auto_noise.setToolTip(
            "Release the lock at this summed probability of the background classes")
        for w in (self.w_auto_dwell, self.w_auto_noise):
            w.setFixedHeight(26)
            w.setEnabled(True)              # Auto is the default mode
            w.valueChanged.connect(self._on_auto_params)
        auto_row.addWidget(auto_cap)
        auto_row.addWidget(self.w_auto_dwell, 1)
        auto_row.addWidget(self.w_auto_noise, 1)
        vbox.addLayout(auto_row)

        # ── The SDR panel ──────────────────────────────────────────────────────
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

        self.w_sr   = cell(0, 0, "Sample Rate (Hz)",
                           hz_combo([2e6, 4e6, 5e6, 10e6, 15e6, 20e6, 30e6, 56e6], SAMPLE_RATE))
        self.w_bw   = cell(0, 1, "Bandwidth (Hz)",
                           hz_combo([1e6, 2e6, 4e6, 5e6, 8e6, 10e6, 20e6, 40e6], RX_BW_HZ))
        self.w_gain = cell(0, 2, "RX Gain (dB)", QtWidgets.QSpinBox())
        self.w_gain.setRange(-3, 71)
        self.w_gain.setValue(GAIN)
        self.w_center = cell(2, 0, "Center Freq (Hz)", QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_span   = cell(2, 1, "Total Span (Hz)",
                             hz_combo([10e6, 20e6, 40e6, 80e6, 100e6, 200e6, 400e6],
                                      TOTAL_SPAN_HZ))
        self.w_olap_pct = cell(2, 2, "Overlap (%)", QtWidgets.QComboBox())
        self.w_olap_pct.addItems(["0", "10", "20", "30", "40", "50", "60", "75"])
        self.w_olap_pct.setCurrentText(str(HOP_OVERLAP_PCT))
        # dwell = the time to read data at one hop. settle = the wait after a retune.
        self.w_dwell = cell(4, 0, "Dwell/hop (ms)", QtWidgets.QSpinBox())
        self.w_dwell.setRange(1, 5000)
        self.w_dwell.setValue(HOP_DWELL_MS)
        self.w_settle = cell(4, 1, "Settle (ms)", QtWidgets.QSpinBox())
        self.w_settle.setRange(0, 500)
        self.w_settle.setValue(HOP_SETTLE_MS)
        vbox.addLayout(grid)

        section("Waterfall Scale  (dB)")
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(4)
        min_lbl = QtWidgets.QLabel("min:")
        min_lbl.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_wf_min = hz_combo([-120, -110, -100, -90, -80, -70, -60, -50, -40,
                                  -30, -20, -10, 0], int(WF_SCALE_MIN_DB))
        self.w_wf_min.setFixedWidth(64)
        max_lbl = QtWidgets.QLabel("max:")
        max_lbl.setStyleSheet("color:#cccccc; font-size:11px;")
        self.w_wf_max = hz_combo([-20, -10, 0, 10, 20, 30, 40], int(WF_SCALE_MAX_DB))
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

        # ── The marker panel ───────────────────────────────────────────────────
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

        # The kind of record. The Record button then starts the correct mode.
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

        # The device label and the focus frequency are for a Device record only.
        # The session and the maximum files are for all the kinds of record.
        self.w_rec_device  = rec_cell(0, 0, "Device label",    QtWidgets.QLineEdit("deviceA"))
        self.w_rec_session = rec_cell(0, 1, "Session",         QtWidgets.QLineEdit("1"))
        self.w_rec_freq    = rec_cell(2, 0, "Focus freq (Hz)", QtWidgets.QLineEdit(str(CENTER_FREQ)))
        self.w_rec_max     = rec_cell(2, 1, "Max files",       QtWidgets.QSpinBox())
        self.w_rec_max.setRange(1, 200000)
        self.w_rec_max.setValue(1000)
        self.w_rec_every   = rec_cell(4, 0, "Keep every Nth",  QtWidgets.QSpinBox())
        self.w_rec_every.setRange(1, 1000)
        self.w_rec_every.setValue(RECORD_EVERY_N)
        self.w_rec_every.setToolTip(
            "Save 1 buffer of N. Narrowband gives one buffer each dwell, thus N=1 "
            "writes about 80 MB each second at the default settings.")
        self.w_rec_rate    = rec_cell(4, 1, "Write rate", QtWidgets.QLineEdit())
        self.w_rec_rate.setReadOnly(True)
        for _w in (self.w_rec_every, self.w_rec_max):
            _w.valueChanged.connect(self._update_rec_rate_label)
        self.w_dwell.valueChanged.connect(self._update_rec_rate_label)
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
        self._on_record_kind("device")     # enable the correct fields
        self._update_rec_rate_label()

        vbox.addSpacing(8)
        apply_btn = QtWidgets.QPushButton("⟳  Apply Settings")
        apply_btn.setFixedHeight(34)
        apply_btn.clicked.connect(self._apply_settings)
        vbox.addWidget(apply_btn)

        section("Status")
        _ss = "color: #cccccc; font-size: 11px;"   # one style for all the rows
        self.model_info_lbl = QtWidgets.QLabel("")
        self.model_info_lbl.setWordWrap(True)
        self.model_info_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.model_info_lbl)
        self.infer_stat_lbl = QtWidgets.QLabel("Inference: —")
        self.infer_stat_lbl.setWordWrap(True)
        self.infer_stat_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.infer_stat_lbl)
        # The band plan is a second opinion beside the model, thus it sits beside the
        # result of the model and never inside the badge.
        self.bandplan_lbl = QtWidgets.QLabel("Band plan: —")
        self.bandplan_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.bandplan_lbl)
        self.mode_lbl = QtWidgets.QLabel("Mode: —")
        self.mode_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.mode_lbl)
        self.caught_lbl = QtWidgets.QLabel("Caught: —")
        self.caught_lbl.setWordWrap(True)
        self.caught_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.caught_lbl)
        # Whether the signal reaches past the receiver window. It was in the title of
        # the narrowband plot, where it moved the title at every frame.
        self.window_lbl = QtWidgets.QLabel("Window: —")
        self.window_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.window_lbl)
        # The floor of the swept band. band_floor_db measures it at every lock and only
        # the title of the narrowband plot used it. It is the one number that says
        # whether the room is quiet, and a bench session asks that question first.
        self.floor_lbl = QtWidgets.QLabel("Band floor: —")
        self.floor_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.floor_lbl)
        # A warning row that is empty must not be visible. An always-present
        # "Warning: —" teaches the user to read past the word.
        self.warn_lbl = QtWidgets.QLabel("")
        self.warn_lbl.setWordWrap(True)
        self.warn_lbl.setStyleSheet("color: #ff6666; font-size: 11px; font-weight: bold;")
        self.warn_lbl.setVisible(False)
        vbox.addWidget(self.warn_lbl)
        self.hop_info_lbl = QtWidgets.QLabel("")
        self.hop_info_lbl.setStyleSheet(_ss)
        vbox.addWidget(self.hop_info_lbl)
        self._update_hop_info_label()
        self.status_lbl = QtWidgets.QLabel("Starting…")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet(_ss)
        self.file_lbl = QtWidgets.QLabel("Files: 0 this run · 0 on disk")
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

        # Remove the small arrows from each spin box. Type or scroll to set a value.
        for _sb in panel.findChildren(QtWidgets.QAbstractSpinBox):
            _sb.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)

        panel_scroll = QtWidgets.QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(340)
        panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel_scroll.setWidget(panel)
        root.addWidget(panel_scroll, stretch=0)

    # ── The badge ─────────────────────────────────────────────────────────────

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

    def _show_warnings(self, warns):
        """Put the warnings on one row of the panel, or hide the row.

        The badge held these until 2026-08-17. The badge is one line of 40 px, thus a
        warning pushed the name of the device sideways or out of the widget."""
        self.warn_lbl.setText("Warning: " + " · ".join(warns) if warns else "")
        self.warn_lbl.setVisible(bool(warns))

    # ── The bars of the probabilities ─────────────────────────────────────────

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

    # ── The model ─────────────────────────────────────────────────────────────

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

    def _sr_mismatch(self) -> bool:
        """True if the model was trained at a different sample rate than the radio
        gives now. One spectrogram row is sample_rate / n_fft Hz, thus the model
        then reads the wrong frequency for each row. A model that has no rate in its
        meta gives False, and the program says nothing about it."""
        sr = getattr(self._engine, "sample_rate", None)
        return bool(sr) and abs(sr - float(self.cfg["sample_rate"])) > 1.0

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
            self.det_badge.setText("SCANNING")
            self._set_badge_style("none")
            short = os.path.basename(path)
            self._no_ambient_class = not has_ambient_class(engine.classes)
            warn = ("<br>⚠ No background class — Auto mode's auto-skip will never "
                    "trigger, and the badge can never say 'Clear'"
                    if self._no_ambient_class else "")
            if self._sr_mismatch():
                warn += (f"<br>⚠ Trained at {engine.sample_rate/1e6:.3f} Msps, the "
                         f"radio gives {float(self.cfg['sample_rate'])/1e6:.3f} Msps "
                         "— every frequency in the spectrogram is wrong")
            # The row of the panel now, and not at the first result. A model that can
            # not say 'Clear' must say so before it reads anything.
            self._show_warnings(
                (["no background class"] if self._no_ambient_class else [])
                + (["sample rate"] if self._sr_mismatch() else []))
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
        """Set the switch of the classifier. The worker reads self.cfg. Thus the
        change is effective at the next loop, and no restart is necessary."""
        self.cfg["ml_enabled"] = checked
        self.w_ml_toggle.setText(f"ML Inference: {'ON' if checked else 'OFF'}")
        if not checked:
            # Clear the values. Thus an old result does not look like a new result.
            self.det_badge.setText("ML OFF")
            self._set_badge_style("none")
            for cls, bar in self._conf_bars.items():
                bar.setValue(0)
                self._conf_labels[cls].setText("0%")
            self.infer_stat_lbl.setText("Inference: off")
            # The band plan travels with a result, thus it is as old as the last one.
            self.bandplan_lbl.setText("Band plan: —")
        elif self._engine is not None:
            self.det_badge.setText("SCANNING")

    # ── The results of the classifier ─────────────────────────────────────────

    def _on_fingerprint_ready(self, result):
        label = result["label"]
        conf  = result["confidence"]
        probs = result["probs"]
        text, kind = badge_for(result, FP_DEVICE_THRESH)
        # The band plan is a second opinion and never the name. If the two disagree,
        # that is information for the user and not a fault to hide. It goes on the
        # panel, thus the badge holds the answer of the model alone.
        # See §9 Phase 5b item 5.
        plan = result.get("band_plan")
        self.bandplan_lbl.setText(f"Band plan: {_hl(plan) if plan else '—'}")
        warns = []
        if getattr(self, "_no_ambient_class", False):
            warns.append("no background class")
        if self._sr_mismatch():
            warns.append("sample rate")
        self._show_warnings(warns)
        if warns and kind == "none":
            kind = "error"
        self.det_badge.setText(text)
        self._set_badge_style(kind)
        for cls, bar in self._conf_bars.items():
            p = probs.get(cls, 0.0)
            bar.setValue(int(p * 100))
            self._conf_labels[cls].setText(f"{p:.0%}")
        self.infer_stat_lbl.setText(f"Inference: {_hl(f'{label} @ {conf:.1%}')}")

    def _on_zoom_ready(self, freqs, psd, held_freq, band_floor=None):
        self.zoom_curve.setData(freqs, psd)
        if band_floor is None:
            self.floor_lbl.setText("Band floor: — (no sweep in this mode)")
        else:
            self.floor_lbl.setText(f"Band floor: {_hl(f'{band_floor:.1f} dB')}")
        sr = self.cfg["sample_rate"]
        # Move the view only for a new lock. Thus the program does not cancel the
        # movements of the user at each frame.
        if (getattr(self, "_zoom_center", None) is None
                or abs(held_freq - self._zoom_center) > sr / 4):
            self.p_zoom.setXRange(held_freq - sr / 2, held_freq + sr / 2, padding=0)
            self._zoom_center = held_freq
            self.zoom_wf_data[:] = -200.0      # remove the data of the last lock
        self.p_zoom.setTitle(f"Narrowband Spectrum ({held_freq / 1e6:.3f} MHz)")
        # A full window is tested first. Its edges are both outside the receiver, thus
        # the program can not measure them and must not draw them.
        filled = window_filled(psd, band_floor)
        self.window_lbl.setText(f"Window: {window_state(filled, freqs, psd)}")
        self.zoom_wf_data = np.roll(self.zoom_wf_data, -1, axis=1)
        self.zoom_wf_data[:, -1] = psd
        self.zoom_wf_img.setImage(self.zoom_wf_data, autoLevels=False)
        self.zoom_wf_img.setRect(QtCore.QRectF(held_freq - sr / 2, 0, sr, WATERFALL_ROWS))
        self._last_zoom = (freqs, psd, filled)
        self._update_signal_markers(freqs, psd, filled)

    # ── The markers of the narrowband signal ──────────────────────────────────

    def _on_marker_toggle(self, _checked=False):
        """Hide the markers that the user set to off. Then draw the other markers
        with the last spectrum. Thus a marker is immediately visible, also in the
        SCAN mode where there is no new narrowband data."""
        if not self.w_show_mid.isChecked():
            self.mid_line.setVisible(False)
        if not self.w_show_borders.isChecked():
            for ln in self.edge_lines:
                ln.setVisible(False)
        if self._last_zoom is not None:
            self._update_signal_markers(*self._last_zoom)

    def _update_signal_markers(self, freqs, psd, filled=False):
        """Put the red lines at the middle and the edges of the signal. The function
        signal_extent gives the positions. If the signal is below the noise floor,
        the program hides the lines.

        A window that is full of signal hides them as well. Both edges are outside
        the receiver there, thus signal_extent measures a bump inside the signal and
        not the signal: 3.24 MHz at the median for a 20 MHz WiFi channel, measured on
        120 captures. The title says why the lines are absent."""
        show_mid  = self.w_show_mid.isChecked()
        show_edge = self.w_show_borders.isChecked()
        if not (show_mid or show_edge):
            return
        extent = None if filled else signal_extent(freqs, psd)
        if extent is None:                         # there is no signal to mark
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
        self.w_jump_combo.clear()
        for f in freqs:
            self.w_jump_combo.addItem(f"{f / 1e6:.3f} MHz", int(f))

    # ── The waterfall ─────────────────────────────────────────────────────────

    def _update_waterfall_rect(self):
        span = self.f_global_max - self.f_global_min
        self.img.setRect(QtCore.QRectF(self.f_global_min, 0, span, WATERFALL_ROWS))
        # Keep the view on the data. Thus the image is not larger than the axis.
        self.p2.setYRange(0, WATERFALL_ROWS, padding=0)

    def _apply_wf_scale(self):
        try:
            v_min = float(self.w_wf_min.currentText())
            v_max = float(self.w_wf_max.currentText())
        except (ValueError, AttributeError):
            return                       # the text is not a number yet
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

    # ── The controls of the record ────────────────────────────────────────────

    def _switch_mode(self, key: str):
        """Set op_mode to `key` and start the worker again. The record does not change."""
        self.cfg["op_mode"] = key
        is_lock = key in ("locking", "auto")    # the two modes that lock
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
        # A manual change of the mode stops the record. Each mode saves other data.
        self.cfg["record"] = False
        self.w_rec_btn.setChecked(False)
        self._update_record_btn_style(False)
        self._sync_record_cfg()
        self._switch_mode(key)

    def _on_skip_lock(self):
        self.cfg["skip_lock"] = True

    def _on_jump_to(self):
        f = self.w_jump_combo.currentData()
        if f:
            self.cfg["jump_to"] = int(f)

    def _on_auto_params(self, _v=0):
        self.cfg["auto_dwell_ms"]  = self.w_auto_dwell.value()
        self.cfg["auto_noise_pct"] = self.w_auto_noise.value()

    def _on_record_toggle(self, checked: bool):
        if self.sdr is None:                 # there is no worker
            self.w_rec_btn.setChecked(False)
            self.status_lbl.setText("SDR not connected.")
            return
        self._sync_record_cfg()
        if not checked:
            self.cfg["record"] = False
            self._update_record_btn_style(False)
            return
        # Start the mode that gives the correct data. A device or the noise at one
        # frequency needs the Narrowband mode. The noise of the band needs Wideband.
        kind   = self.cfg.get("record_kind", "device")
        target = "wideband" if kind == "noise_band" else "focus"
        self.cfg["record"] = True
        self.w_rec_btn.setChecked(True)
        self._update_record_btn_style(True)
        if self.cfg.get("op_mode") != target:
            self._switch_mode(target)

    def _on_record_kind(self, kind: str):
        """Set the kind of data that the Record button saves: a device, the noise of
        the full band, or the noise at the focus frequency."""
        self.cfg["record_kind"] = kind
        parked = kind in ("device", "noise_freq")     # one frequency, not a sweep
        self.w_rec_device.setEnabled(kind == "device")
        self.w_rec_freq.setEnabled(parked)
        if kind in self._rec_kind_btns:
            self._rec_kind_btns[kind].setChecked(True)
        # If the record is on, change to the correct mode now.
        if self.cfg.get("record") and hasattr(self, "worker"):
            self._on_record_toggle(True)

    def _update_rec_rate_label(self, _v=0):
        """Show the size that a narrowband record writes each second.

        One buffer is sample_rate * dwell_ms / 1000 samples of complex64, and the
        program reads one buffer each dwell. Thus the rate does not change with the
        dwell, and only the divisor moves it."""
        every = max(1, self.w_rec_every.value())
        n = max(1024, int(self.cfg["sample_rate"] * self.w_dwell.value() / 1000.0))
        mb = n * 8 / 1e6
        per_s = mb * (1000.0 / max(1, self.w_dwell.value())) / every
        cap_gb = mb * self.w_rec_max.value() / 1000.0
        self.w_rec_rate.setText(f"{per_s:.0f} MB/s · cap {cap_gb:.1f} GB")

    def _sync_record_cfg(self):
        self.cfg["record_device"]    = self.w_rec_device.text().strip() or "deviceA"
        self.cfg["record_session"]   = self.w_rec_session.text().strip() or "1"
        self.cfg["record_max_files"] = self.w_rec_max.value()
        self.cfg["record_every_n"]   = self.w_rec_every.value()
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
            self.w_rec_btn.setStyleSheet("")  # use the style of the panel

    def _on_grab_lock(self):
        f = getattr(self, "_last_held_freq", None)
        if f:
            self.w_rec_freq.setText(str(int(f)))
            self.cfg["focus_freq"] = int(f)

    # ── The settings ──────────────────────────────────────────────────────────

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
            self._sync_record_cfg()
            self.cfg["record"]             = False     # a change stops the record
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
            self.p1.setYRange(WF_SCALE_MIN_DB, WF_SCALE_MAX_DB, padding=0)
            self._apply_wf_scale()
            self.status_lbl.setText(
                f"Applied. {self.n_hops} hops · "
                f"{self.cfg['center_freq'] / 1e6:.1f} MHz center")
        except Exception as e:
            # Do not start a worker on a configuration that was applied in part. The
            # composite length would disagree with self.total_bins, _on_sweep_ready
            # would drop every frame, and the plot would freeze with no message that
            # says why. Correct the fields and press Apply Settings again.
            self.status_lbl.setText(f"Apply error: {e}. The sweep is stopped. "
                                    f"Correct the settings and apply again.")
            return
        self._start_worker()

    def _start_worker(self):
        self._zoom_center = None        # the next lock moves the narrowband view
        self.cfg["skip_lock"] = False
        self.caught_lbl.setText("Caught: —")     # each start clears the memory
        self.w_jump_combo.clear()
        # Release the previous worker. It is stopped already, but its signals stay
        # connected to these handlers and Qt keeps the object alive. A long session
        # with many mode changes then holds every worker it ever made, and one sweep
        # calls each handler once for every one of them.
        old = getattr(self, "worker", None)
        if old is not None:
            for sig in (old.sweep_ready, old.zoom_ready, old.fingerprint_ready,
                        old.mode_changed, old.caught_changed, old.hop_progress,
                        old.status_msg, old.files_changed):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass                # nothing was connected, or Qt deleted it
            old.deleteLater()
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

    # ── The handlers of the Qt signals ────────────────────────────────────────

    def _on_sweep_ready(self, composite: np.ndarray, hop_bufs: dict):
        if len(composite) != self.total_bins:
            return
        freqs = bin_freqs(self.f_global_min, self.f_global_max, self.total_bins)
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

    def _on_files_changed(self, this_run, on_disk):
        cap = self.cfg.get("record_max_files", 1000)
        warn = "  ⚠ over the cap" if on_disk > cap else ""
        self.file_lbl.setText(
            f"Files: {_hl(this_run)} this run · {_hl(on_disk)} on disk{warn}")

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
        if self.sdr is not None:
            del self.sdr
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # You must set the scale policy before the QApplication exists. PassThrough
    # keeps a fractional scale correct, for example 125%. Thus the plots stay
    # aligned when the user moves the window to a monitor with a different scale.
    try:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except AttributeError:
        pass  # the policy is not available before Qt 5.14
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