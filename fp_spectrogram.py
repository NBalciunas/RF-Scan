"""The shared spectrogram front end and the model.

train_model.py and terminal.py import this module. Thus the two programs
always prepare the data in the same way.

The representation is a 256-point STFT spectrogram of the IQ data. The image has
one channel of log magnitude. The phase is not kept.
"""

import os
import json
import functools

import numpy as np
import torch
import torch.nn as nn

N_FFT    = 256
STFT_HOP = 64           # samples between two STFT frames
SEG_LEN  = 4096         # IQ samples for one spectrogram (0.4 ms at 10 Msps)
DC_HP_WIN = 256         # moving average of remove_dc. About +-20 kHz at 10 Msps.
SEG_HOP  = 2048         # samples between two spectrogram segments
INFER_MAX_SEGS = 24     # live inference: maximum segments for one buffer (0 = all)
# A class is present when it wins this part of the segments. The value decides how
# often a bursty link is seen, thus it was measured and not chosen: on the validation
# session, 0.10 gives 98% of the DJI captures and 32% of the hopping AT9S captures
# with no false alarm in 100 noise captures. 0.05 reaches 63% and costs 2%, and 0.20
# gave the AT9S 2%. See the defect #29 and evaluate.py --sweep.
MIN_SEG_SHARE  = 0.10
VOTE_THRESH    = 0.5    # a segment votes when its best class has this probability

# The classes that are background, not a device. The votes, the badge and the Auto
# release all read these as "not a drone", and the trainer keeps their frequencies out
# of train_freqs. One set for the four decisions, thus terminal.py and train_model.py can
# not disagree about what a device is. A model holds only the classes its dataset held,
# thus a name here that the model does not carry costs nothing: its probability is 0 and
# no folder carries its label. See §9.2 job 1 of NOTES.md.
AMBIENT_LABELS = ("noise", "wifi", "bluetooth")


@functools.lru_cache(maxsize=8)
def _window(n_fft):
    return np.hanning(n_fft).astype(np.float32)


def remove_dc(iq, win=DC_HP_WIN):
    """Remove the artifact of the receiver near 0 Hz from an IQ buffer.

    The artifact is not a constant. Measured on the PlutoSDR on 2026-08-10 with a
    50 ohm load, at 4 frequencies and 6 gains. On one segment of SEG_LEN samples,
    which is what the classifier sees, `|mean(iq)| / rms(iq)` is -24.3 dB and the
    0 Hz bin stands 8.6 dB above the median bin, of which a mean removes 0.95 dB.
    A mean therefore removes almost nothing. The earlier version of this function
    subtracted a mean and it had no measurable effect on any real capture.

    The function subtracts a moving average of `win` samples, which is a high pass.
    At 10 Msps and win = 256 the notch is about +-20 kHz: a tone 39 kHz from 0 Hz
    loses 0.12 dB, one at 20 kHz loses 8.5 dB, one at 10 kHz loses 19.6 dB. A carrier
    exactly on the LO is lost, and no method that removes the artifact can keep it.

    Do not make `win` smaller to remove more. Measured over 24 real captures on a
    50 ohm load, the height of the DC row of the spectrogram is +1.119 sigma with a
    mean, **-0.099 sigma at win = 256**, -1.340 at win = 128 and -2.815 at win = 64.
    A hole is a class cue in the same way that a line is.

    Call this function before any frequency shift. After a shift the artifact is no
    longer at 0 Hz and a high pass at 0 Hz can not find it."""
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        return iq
    if win < 2 or n <= win:
        return (iq - iq.mean()).astype(np.complex64)
    # Hold the edge value, thus the ends do not bend towards zero.
    pad = np.concatenate([np.full(win // 2, iq[0]), iq,
                          np.full(win - win // 2 - 1, iq[-1])])
    # complex128 for the sum. A complex64 cumsum drifts over a long capture.
    c = np.cumsum(np.concatenate([[0.0 + 0.0j], pad]), dtype=np.complex128)
    return (iq - (c[win:] - c[:-win]) / win).astype(np.complex64)


def iq_to_spectrogram(iq, n_fft=N_FFT, hop=STFT_HOP):
    """Make one log-magnitude spectrogram (1, n_fft, n_frames) from complex IQ data.

    The function normalizes each image with its own mean and standard deviation.

    The 0 Hz row is replaced by the median row. `remove_dc` takes the artifact of the
    receiver out of the signal, but it can not make the row identical for every class:
    it runs before the frequency shift, thus the notch that it leaves moves away from
    0 Hz for a segment that the augmentation shifts, and stays at 0 Hz for one that it
    does not. The device classes are shifted and the noise class is not, so the depth
    of the row at 0 Hz followed the class. Measured: -0.89 sigma against -1.16 sigma.
    The replacement makes the row carry nothing, for every class and every path."""
    iq = remove_dc(iq)
    if len(iq) < n_fft:
        iq = np.pad(iq, (0, n_fft - len(iq)))
    n   = (len(iq) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    frames = iq[idx] * _window(n_fft)
    S   = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    mag = 20.0 * np.log10(np.abs(S) / n_fft + 1e-10)
    spec = mag.T.astype(np.float32)
    spec[n_fft // 2] = np.median(spec, axis=0)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec[None]


def iq_segments_to_specs(iq, seg_len=SEG_LEN, seg_hop=SEG_HOP,
                         n_fft=N_FFT, hop=STFT_HOP, max_segs=0):
    """Cut an IQ buffer into segments -> (k, 1, n_fft, frames).

    If max_segs is more than 0, the function keeps that many segments at equal
    distances. Thus the segments still cover the full time of the buffer."""
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < seg_len:
        return iq_to_spectrogram(iq, n_fft, hop)[None]
    k = (len(iq) - seg_len) // seg_hop + 1
    idx = np.arange(k)
    if max_segs and k > max_segs:
        idx = np.unique(np.linspace(0, k - 1, max_segs).round().astype(int))
    return np.stack([iq_to_spectrogram(iq[i * seg_hop:i * seg_hop + seg_len],
                                       n_fft, hop) for i in idx])


class SpecCNN(nn.Module):
    """A small 2-D CNN for the single-channel STFT spectrograms."""

    def __init__(self, n_classes, base=16, dropout=0.3):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.GELU(),
                # ceil_mode: a short time axis must not become length 0
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


def segment_vote(seg_probs, classes, thresh, min_share):
    """Find each class that is present in the buffer, and not only the strongest one.

    A segment votes for its best class if the probability is thresh or more. A class
    is present if it wins min_share of all the segments. Thus the function finds two
    or more transmitters that send at different times in one capture. The function
    can not divide two signals that are in the same segment.

    thresh is VOTE_THRESH and not unknown_thresh. One segment is 0.4 ms, thus its
    probability is always lower than the mean of a full buffer. A limit that is
    correct for the mean stops all the votes.

    The function gives [{label, share, confidence}, ...]. The strongest is first.
    """
    win, wconf = seg_probs.argmax(1), seg_probs.max(1)
    k = len(seg_probs)
    out = []
    for i, cls in enumerate(classes):
        m = (win == i) & (wconf >= thresh)
        # A class with no segment is not present at any share. The test also keeps an
        # empty selection out of the mean below, because that value is NaN.
        if m.sum() and m.sum() / k >= min_share:
            out.append({"label": cls, "share": float(m.sum() / k),
                        "confidence": float(seg_probs[m, i].mean())})
    return sorted(out, key=lambda d: d["share"], reverse=True)


class FingerprintModel:
    """The inference wrapper that the GUI uses. It does not use Qt or the SDR.

    The wrapper makes a spectrogram of each segment of an IQ buffer. Then it
    calculates the mean of the probabilities. If the best class is below
    unknown_thresh, the label is 'unknown'.
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
        self.vote_thresh    = float(meta.get("vote_thresh", VOTE_THRESH))
        # The operating point of the badge. It belongs to the model and not to the
        # code, because it is chosen against one dataset. A model from before this
        # field falls back to the constant.
        self.min_seg_share  = float(meta.get("min_seg_share", MIN_SEG_SHARE))
        # The centre frequencies of the training captures. A scanner uses them to know
        # where it has been taught to listen, thus it never walks past one because the
        # band plan called it WiFi. A model from before this field gives an empty list
        # and the caller falls back to its own constant.
        self.train_freqs    = [float(f) for f in meta.get("train_freqs", [])]
        self.infer_max_segs = int(infer_max_segs)
        # The rate of the captures that trained the model. A spectrogram row is
        # sample_rate / n_fft Hz, thus another rate makes every frequency in the
        # image wrong. A model from before this field gives None, and the caller
        # then makes no comparison.
        _sr = meta.get("sample_rate")
        self.sample_rate = float(_sr) if _sr else None

        self.net = SpecCNN(len(self.classes), base=base)
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.net.eval()

    def classify_iq(self, iq):
        specs = iq_segments_to_specs(iq, self.seg_len, self.seg_hop,
                                     self.n_fft, self.stft_hop,
                                     max_segs=self.infer_max_segs)
        x = torch.from_numpy(specs.astype(np.float32))
        with torch.no_grad():
            seg_probs = torch.softmax(self.net(x), dim=1).numpy()
        probs = seg_probs.mean(0)
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        label = self.classes[idx] if conf >= self.unknown_thresh else "unknown"
        # Every share, and then the same list at MIN_SEG_SHARE. One rule makes both,
        # thus a caller that holds a lock below the limit of the badge reads the same
        # numbers that the badge reads.
        votes = segment_vote(seg_probs, self.classes, self.vote_thresh, 0.0)
        return {
            "label"     : label,
            "confidence": conf,
            "probs"     : {c: float(p) for c, p in zip(self.classes, probs)},
            "unknown"   : conf < self.unknown_thresh,
            "shares"    : {d["label"]: d["share"] for d in votes},
            "detections": [d for d in votes if d["share"] >= self.min_seg_share],
        }


if __name__ == "__main__":
    # self-check: segment_vote must find the two transmitters in one buffer
    cls = ["deviceA", "droneB", "noise"]
    p = np.array([[0.95, 0.03, 0.02]] * 5      # 5 segments of deviceA
                 + [[0.05, 0.90, 0.05]] * 4    # 4 segments of droneB
                 + [[0.40, 0.35, 0.25]] * 1)   # 1 segment with a low probability
    dets = segment_vote(p, cls, thresh=0.8, min_share=0.2)
    assert [d["label"] for d in dets] == ["deviceA", "droneB"], dets
    assert abs(dets[0]["share"] - 0.5) < 1e-9 and abs(dets[1]["share"] - 0.4) < 1e-9
    assert segment_vote(p[-1:], cls, 0.8, 0.2) == []
    print("segment_vote self-check OK:", dets)
