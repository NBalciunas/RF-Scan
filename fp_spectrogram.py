"""
fp_spectrogram.py — shared spectrogram front-end + model for v2 fingerprinting.

Both train_model.py (training) and terminal_v2.py (live inference) import from here,
so preprocessing can never drift between the two — the same lesson the original
pipeline learned with instance_norm.

Representation follows the RFUAV paper (arXiv:2503.09033): a 256-point STFT
spectrogram of the captured IQ.  Two deliberate adaptations for our scale:

  * Single-channel log-magnitude image, NOT the paper's RGB 'Hot' colormap.  The
    colormap only matters when feeding an ImageNet-pretrained RGB net (ViT/ResNet)
    — which is why the paper tuned it.  A compact CNN trained from scratch reads
    the raw dB values directly, so the colormap would be a cosmetic no-op.
  * Magnitude (not complex) discards phase, exactly as the paper does.  Fine for
    model/type-level ID; a known limit for same-unit SEI.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn

N_FFT    = 256          # paper's STFT sweet spot (STFTP=256)
STFT_HOP = 64           # hop between STFT frames within a spectrogram
SEG_LEN  = 4096         # IQ samples per spectrogram (~0.4 ms @ 10 Msps)
SEG_HOP  = 2048         # hop between successive spectrogram segments
INFER_MAX_SEGS = 24     # live inference: cap segments per buffer (evenly subsampled).
                        # Averaging softmax over a representative subset is as good as
                        # all ~240 and ~10x cheaper. 0 = use every segment.

_WINDOW = np.hanning(N_FFT).astype(np.float32)


def iq_to_spectrogram(iq, n_fft=N_FFT, hop=STFT_HOP):
    """complex IQ -> (1, n_fft, n_frames) per-image-standardised log-mag (dB)."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < n_fft:
        iq = np.pad(iq, (0, n_fft - len(iq)))
    n   = (len(iq) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = iq[idx] * _WINDOW                              # (n, n_fft)
    S   = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    mag = 20.0 * np.log10(np.abs(S) / n_fft + 1e-10)        # dB
    spec = mag.T.astype(np.float32)                         # (n_fft, n)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)       # per-image standardise
    return spec[None]                                       # (1, n_fft, n)


def iq_segments_to_specs(iq, seg_len=SEG_LEN, seg_hop=SEG_HOP,
                         n_fft=N_FFT, hop=STFT_HOP, max_segs=0):
    """Slice an IQ buffer into seg_len chunks -> (k, 1, n_fft, frames).

    max_segs > 0 evenly subsamples to at most that many segments (keeps the temporal
    spread) — the live-inference speed lever, mirroring the trainer's max_segs_per_file."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < seg_len:
        return iq_to_spectrogram(iq, n_fft, hop)[None]      # single (1,1,nfft,fr)
    k = (len(iq) - seg_len) // seg_hop + 1
    idx = np.arange(k)
    if max_segs and k > max_segs:
        idx = np.unique(np.linspace(0, k - 1, max_segs).round().astype(int))
    return np.stack([iq_to_spectrogram(iq[i * seg_hop:i * seg_hop + seg_len],
                                       n_fft, hop) for i in idx])


class SpecCNN(nn.Module):
    """Compact 2-D CNN over single-channel STFT spectrograms."""

    def __init__(self, n_classes, base=16, dropout=0.3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.GELU(),
                # ceil_mode so a small time axis never pools down to length 0.
                nn.MaxPool2d(2, ceil_mode=True),
            )

        self.cnn = nn.Sequential(
            block(1, base), block(base, base * 2),
            block(base * 2, base * 4), block(base * 4, base * 4),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(base * 4, n_classes),
        )

    def forward(self, x):
        return self.head(self.cnn(x))


class FingerprintModel:
    """Qt/SDR-free inference wrapper used by the v2 GUI.

    Give it a captured IQ buffer; it spectrograms every segment, averages the
    softmax across segments, and returns the device verdict — or 'unknown' when
    the top class is below `unknown_thresh` (an A/B classifier has no native
    "none of the above", so the threshold supplies one).
    """

    def __init__(self, model_path, unknown_thresh=0.8, infer_max_segs=INFER_MAX_SEGS):
        meta_path = os.path.splitext(model_path)[0] + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        self.classes  = meta.get("classes", [])
        self.n_fft    = int(meta.get("n_fft",    N_FFT))
        self.stft_hop = int(meta.get("stft_hop", STFT_HOP))
        self.seg_len  = int(meta.get("seg_len",  SEG_LEN))
        self.seg_hop  = int(meta.get("seg_hop",  SEG_HOP))
        base          = int(meta.get("base_ch",  16))
        self.unknown_thresh = float(meta.get("unknown_thresh", unknown_thresh))
        self.infer_max_segs = int(infer_max_segs)

        self.net = SpecCNN(len(self.classes), base=base)
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()

    def classify_iq(self, iq):
        specs = iq_segments_to_specs(iq, self.seg_len, self.seg_hop,
                                     self.n_fft, self.stft_hop,
                                     max_segs=self.infer_max_segs)
        x = torch.from_numpy(specs.astype(np.float32))
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1).mean(0).numpy()
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        label = self.classes[idx] if conf >= self.unknown_thresh else "unknown"
        return {
            "label"     : label,
            "confidence": conf,
            "probs"     : {c: float(p) for c, p in zip(self.classes, probs)},
            "unknown"   : conf < self.unknown_thresh,
        }
