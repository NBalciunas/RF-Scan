"""End-to-end self-check. Type `python tests/test_end_to_end.py`.

This is the check that answers "does the whole program still work?". It makes a
synthetic dataset of two drones and a noise class, it runs the real trainer on it, and
it gives the result to the same FingerprintModel wrapper that the GUI loads. Then it
asks the wrapper to name a capture that it has never seen.

The two synthetic drones are separable, thus the val accuracy must be high. A low
number here means that something in the chain is broken, not that the task is hard.

It is the slowest check. It needs about one minute on a CPU.
"""

import io
import json
import time
import shutil
import tempfile
import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch

from _support import Checks, run
import train_model
from train_model import train
from fp_spectrogram import (FingerprintModel, iq_segments_to_specs,
                            SEG_LEN, SEG_HOP, VOTE_THRESH)

SESSIONS, FILES, SEGS = 3, 6, 20
FILE_LEN = SEG_LEN + (SEGS - 1) * SEG_HOP
# The two tones stay apart after the +-10% frequency-shift augmentation:
# droneA lands in [0.10, 0.30] and droneB in [-0.40, -0.15].
TONES = {"droneA": 0.20, "droneB": -0.25}


def _noise(n, sigma=0.02, rng=None):
    r = rng or np.random.RandomState(0)
    return ((r.randn(n) + 1j * r.randn(n)) * (sigma / np.sqrt(2))).astype(np.complex64)


def _capture(cls, rng, continuous=False):
    """A device capture is bursts with silence between them, as a real one is.

    continuous=True fills the whole buffer with the transmission. That is what the
    radio sees while the drone is actually sending, thus it is the right input for
    the two-drone vote."""
    iq = _noise(FILE_LEN, rng=rng)
    if cls == "noise":
        return iq
    f = TONES[cls]
    if continuous:
        t = np.arange(FILE_LEN)
        burst = np.exp(2j * np.pi * f * t)
        if cls == "droneB":
            burst = burst + 0.6 * np.exp(2j * np.pi * (f - 0.05) * t)
        return (iq + burst).astype(np.complex64)
    # Three bursts with silence between them, as a parked capture has. droneB has a
    # second line close to its first, thus the two drones differ by more than one
    # tone and the task is a small fingerprint, not a single frequency.
    for start in (0, 35, 70):
        s = start * FILE_LEN // 100
        e = s + FILE_LEN // 5
        t = np.arange(e - s)
        burst = np.exp(2j * np.pi * f * t)
        if cls == "droneB":
            burst = burst + 0.6 * np.exp(2j * np.pi * (f - 0.05) * t)
        iq[s:e] += burst.astype(np.complex64)
    return iq


def build_tree(root):
    rng = np.random.RandomState(7)
    for cls in ("droneA", "droneB", "noise"):
        for s in range(1, SESSIONS + 1):
            d = root / cls / f"session_{s}"
            d.mkdir(parents=True)
            for i in range(FILES):
                _capture(cls, rng).tofile(d / f"{cls}_s{s}_{i}.iq")
    return root


def _args(data_dir, out, **over):
    a = dict(data_dir=str(data_dir), out=str(out), seg_len=SEG_LEN, seg_hop=SEG_HOP,
             batch_size=64, lr=1e-3, unknown_thresh=0.8, vote_thresh=VOTE_THRESH,
             seed=0, preset="test",
             epochs=25, base_ch=8, max_files_per_class=0, max_segs_per_file=0,
             max_segs_per_class=0, store_dtype="float16", gate_margin_db=3.0,
             snr_aug_p=0.5, freq_shift_frac=0.10, quick=False, cpu=True)
    a.update(over)
    return argparse.Namespace(**a)


def main():
    c = Checks("End to end: dataset -> trainer -> GUI wrapper")
    tmp = Path(tempfile.mkdtemp(prefix="rfscan_e2e_"))
    try:
        root = build_tree(tmp / "fingerprint_data")
        out = tmp / "models" / "e2e_model.pt"            # the trainer makes the folder
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            train(_args(root, out))
        log = buf.getvalue()
        c.note(f"the trainer ran in {time.time() - t0:.0f} s")

        @c.check("the trainer writes the weights and the meta side by side")
        def _():
            assert out.exists(), out
            assert out.with_suffix(".meta.json").exists(), "no .meta.json"

        @c.check("the meta holds everything that FingerprintModel reads")
        def _():
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            for k in ("classes", "n_fft", "stft_hop", "seg_len", "seg_hop",
                      "base_ch", "unknown_thresh", "val_acc", "weak_val_acc"):
                assert k in meta, f"the meta has no '{k}'"
            assert meta["classes"] == ["droneA", "droneB", "noise"], meta["classes"]
            assert meta["base_ch"] == 8

        @c.check("the trainer prints the confusion matrix and the per-class figures")
        def _():
            assert "Confusion matrix" in log, log[-400:]
            assert "Per-class (val)" in log, log[-400:]
            for cls in ("droneA", "droneB", "noise"):
                assert f"{cls:>10}: precision" in log, cls

        @c.check("the two synthetic drones are separated on the held-out session")
        def _():
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            acc = float(meta["val_acc"])
            c.note(f"ValAcc {acc:.1%}")
            assert acc > 0.90, \
                f"only {acc:.1%} on a separable task, the chain is broken somewhere"

        @c.check("the weak-signal accuracy is measured and reported")
        def _():
            meta = json.loads(out.with_suffix(".meta.json").read_text())
            assert meta["weak_val_acc"] is not None, "no weak-signal number"
            c.note(f"weak-signal val {float(meta['weak_val_acc']):.1%}")
            assert "Weak-signal val" in log

        @c.check("the GUI wrapper loads that model and names an unseen capture")
        def _():
            fm = FingerprintModel(str(out))
            assert fm.classes == ["droneA", "droneB", "noise"]
            rng = np.random.RandomState(999)          # a session that never existed
            for cls in ("droneA", "droneB", "noise"):
                iq = np.concatenate([_capture(cls, rng) for _ in range(3)])
                res = fm.classify_iq(iq)
                top = max(res["probs"], key=res["probs"].get)
                c.note(f"{cls:<7} -> {top} ({res['probs'][top]:.0%})")
                assert top == cls, f"a {cls} capture was called {top}"

        def _two_drone_buffer():
            rng = np.random.RandomState(1234)
            return np.concatenate([_capture("droneA", rng, continuous=True),
                                   _capture("droneB", rng, continuous=True)])

        def _seg_probs(fm, iq):
            """The per-segment probabilities that classify_iq votes on."""
            specs = iq_segments_to_specs(iq, fm.seg_len, fm.seg_hop, fm.n_fft,
                                         fm.stft_hop, max_segs=fm.infer_max_segs)
            with torch.no_grad():
                return torch.softmax(
                    fm.net(torch.from_numpy(specs.astype(np.float32))), 1).numpy()

        @c.check("every segment of a two-drone buffer is put in the correct class")
        def _():
            fm = FingerprintModel(str(out))
            p = _seg_probs(fm, _two_drone_buffer())
            win = p.argmax(1)
            got = {cls: int((win == i).sum()) for i, cls in enumerate(fm.classes)}
            c.note(f"segment winners {got}, confidence median "
                   f"{np.median(p.max(1)):.2f}, max {p.max(1).max():.2f}")
            assert got["noise"] == 0, got
            assert got["droneA"] >= len(p) * 0.3 and got["droneB"] >= len(p) * 0.3, got

        @c.check("the badge path reports both drones through classify_iq")
        def _():
            # The vote gates each segment at vote_thresh, not at unknown_thresh.
            # unknown_thresh is for the mean over a whole buffer. Applied to one
            # 0.4 ms segment it stops every vote, and the badge never names two
            # drones. That is the whole two-drone feature.
            fm = FingerprintModel(str(out))
            res = fm.classify_iq(_two_drone_buffer())
            dets = {d["label"]: d["share"] for d in res["detections"]}
            c.note(f"vote_thresh {fm.vote_thresh} (unknown_thresh "
                   f"{fm.unknown_thresh}) gives {dets}")
            assert {"droneA", "droneB"} <= set(dets), dets
            assert dets["droneA"] > 0.2 and dets["droneB"] > 0.2, dets

        @c.check("one drone alone does not report a second one")
        def _():
            # The lower vote threshold must not invent a transmitter.
            fm = FingerprintModel(str(out))
            rng = np.random.RandomState(4242)
            for cls in ("droneA", "droneB"):
                iq = _capture(cls, rng, continuous=True)
                names = {d["label"] for d in fm.classify_iq(iq)["detections"]}
                assert names <= {cls, "noise"}, f"{cls} also reported {names}"

        @c.check("the training loop survives a short --seg_len")
        def _():
            # The SpecAugment mask widths are clamped to the size of the image. A
            # seg_len of 512 gives 5 frames, and an unclamped mask of 12 raises.
            old = train_model.SPEC_MASK_P
            train_model.SPEC_MASK_P = 1.0          # make the mask certain
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    train(_args(root, tmp / "models" / "short.pt", seg_len=512,
                                seg_hop=256, epochs=1, max_segs_per_file=8))
            finally:
                train_model.SPEC_MASK_P = old

        @c.check("the trainer creates the output directory itself")
        def _():
            missing = tmp / "does_not_exist" / "m.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                train(_args(root, missing, epochs=1, max_segs_per_file=4))
            assert missing.exists()

        return c.report()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run(main)
