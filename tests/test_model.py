"""Self-check for the network and the inference wrapper. Type `python tests/test_model.py`.

The checks cover SpecCNN, the vote of the segments, and the FingerprintModel wrapper
that the GUI loads. The wrapper is written to a temporary directory and read back.
Thus the check proves that the trainer and the GUI agree about the two files.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from _support import Checks, run
from fp_spectrogram import (SpecCNN, segment_vote, FingerprintModel,
                            N_FFT, SEG_LEN, MIN_SEG_SHARE)

CLASSES = ["droneA", "droneB", "noise"]


def _write_model(dirpath, name="m.pt", classes=CLASSES, base=8, **meta_over):
    path = os.path.join(dirpath, name)
    torch.save(SpecCNN(len(classes), base=base).state_dict(), path)
    meta = {"classes": classes, "n_fft": N_FFT, "stft_hop": 64,
            "seg_len": SEG_LEN, "seg_hop": 2048, "base_ch": base,
            "unknown_thresh": 0.8}
    meta.update(meta_over)
    with open(Path(path).with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f)
    return path


def _noise(n, seed=0):
    r = np.random.RandomState(seed)
    return ((r.randn(n) + 1j * r.randn(n)) / np.sqrt(2)).astype(np.complex64)


def main():
    c = Checks("Network and inference (fp_spectrogram.py)")

    # ── SpecCNN ───────────────────────────────────────────────────────────────

    @c.check("SpecCNN turns a batch of spectrograms into one logit for each class")
    def _():
        net = SpecCNN(3, base=16).eval()
        out = net(torch.zeros(5, 1, N_FFT, 61))
        assert out.shape == (5, 3), out.shape

    @c.check("ceil_mode keeps a short time axis alive through the four pools")
    def _():
        # 5 frames become 3, 2, 1, 1. Without ceil_mode the third pool gives 0.
        net = SpecCNN(3, base=8).eval()
        assert net(torch.zeros(1, 1, N_FFT, 5)).shape == (1, 3)
        assert net(torch.zeros(1, 1, N_FFT, 1)).shape == (1, 3)

    @c.check("the parameter count is the value that README.md gives")
    def _():
        for base, want in ((16, 60_000), (24, 135_000)):
            n = sum(p.numel() for p in SpecCNN(3, base=base).parameters())
            assert abs(n - want) < want * 0.05, f"base {base}: {n:,} parameters"
            c.note(f"base {base}: {n:,} parameters")

    @c.check("the network does not change its answer between two calls")
    def _():
        net = SpecCNN(3, base=8).eval()
        x = torch.randn(4, 1, N_FFT, 61)
        with torch.no_grad():
            assert torch.equal(net(x), net(x))

    # ── The vote of the segments ──────────────────────────────────────────────

    @c.check("the vote finds two transmitters that send at different times")
    def _():
        p = np.array([[0.95, 0.03, 0.02]] * 5      # 5 segments of droneA
                     + [[0.05, 0.90, 0.05]] * 4    # 4 segments of droneB
                     + [[0.40, 0.35, 0.25]] * 1)   # 1 segment, no confidence
        dets = segment_vote(p, CLASSES, thresh=0.8, min_share=0.2)
        assert [d["label"] for d in dets] == ["droneA", "droneB"], dets
        assert abs(dets[0]["share"] - 0.5) < 1e-9
        assert abs(dets[1]["share"] - 0.4) < 1e-9
        assert dets[0]["confidence"] > 0.9

    @c.check("a buffer with no confident segment gives no name")
    def _():
        p = np.array([[0.40, 0.35, 0.25]] * 4)
        assert segment_vote(p, CLASSES, 0.8, 0.2) == []

    @c.check("the vote sorts the strongest share first")
    def _():
        p = np.array([[0.95, 0.03, 0.02]] * 2 + [[0.05, 0.90, 0.05]] * 7)
        dets = segment_vote(p, CLASSES, 0.8, 0.2)
        assert [d["label"] for d in dets] == ["droneB", "droneA"], dets
        assert dets[0]["share"] > dets[1]["share"]

    @c.check("a class that wins less than min_share is not reported")
    def _():
        p = np.array([[0.95, 0.03, 0.02]] * 1 + [[0.05, 0.90, 0.05]] * 9)
        dets = segment_vote(p, CLASSES, 0.8, 0.2)
        assert [d["label"] for d in dets] == ["droneB"], dets

    @c.check("a quiet channel gives the noise class only")
    def _():
        p = np.array([[0.02, 0.03, 0.95]] * 10)
        dets = segment_vote(p, CLASSES, 0.8, MIN_SEG_SHARE)
        assert [d["label"] for d in dets] == ["noise"], dets

    # ── FingerprintModel ──────────────────────────────────────────────────────

    tmp = tempfile.mkdtemp(prefix="rfscan_model_")
    try:
        @c.check("the trainer and the GUI calculate the same .meta.json path")
        def _():
            # The trainer uses Path.with_suffix, FingerprintModel uses splitext.
            for name in ("trained_model.pt", "fast_demo_model.pt",
                         "out/m.v2.pt", "mymodel", "a.b.c.pt"):
                trainer = Path(name).with_suffix(".meta.json")
                loader = Path(os.path.splitext(name)[0] + ".meta.json")
                assert trainer == loader, f"{name}: {trainer} vs {loader}"

        @c.check("a saved model loads and classifies an IQ buffer")
        def _():
            fm = FingerprintModel(_write_model(tmp))
            res = fm.classify_iq(_noise(50_000))
            assert set(res) == {"label", "confidence", "probs", "unknown",
                                "shares", "detections"}
            assert res["label"] in CLASSES + ["unknown"]
            assert sorted(res["probs"]) == sorted(CLASSES)
            assert abs(sum(res["probs"].values()) - 1.0) < 1e-5
            assert isinstance(res["detections"], list)

        @c.check("the wrapper reads the geometry from the meta, not from the constants")
        def _():
            p = _write_model(tmp, name="geo.pt", seg_len=2048, seg_hop=1024,
                             unknown_thresh=0.55)
            fm = FingerprintModel(p)
            assert fm.seg_len == 2048 and fm.seg_hop == 1024, (fm.seg_len, fm.seg_hop)
            assert abs(fm.unknown_thresh - 0.55) < 1e-9
            assert fm.classes == CLASSES

        @c.check("the wrapper reads the sample rate of the training data")
        def _():
            fm = FingerprintModel(_write_model(tmp, name="sr.pt",
                                               sample_rate=10_000_000))
            assert fm.sample_rate == 10_000_000.0, fm.sample_rate

        @c.check("a meta with no sample rate gives None, not a crash")
        def _():
            # Every model that was trained before the field exists is in this state.
            # None must mean 'do not compare', and never 0 Hz.
            fm = FingerprintModel(_write_model(tmp, name="nosr.pt"))
            assert fm.sample_rate is None, fm.sample_rate

        @c.check("a confidence below unknown_thresh gives the label 'unknown'")
        def _():
            # Three untrained classes give about 1/3 each, which is below any
            # sensible threshold.
            fm = FingerprintModel(_write_model(tmp, name="unk.pt",
                                               unknown_thresh=0.99))
            res = fm.classify_iq(_noise(20_000))
            assert res["label"] == "unknown" and res["unknown"] is True, res["label"]

        @c.check("the inference cap holds the segment count down")
        def _():
            fm = FingerprintModel(_write_model(tmp, name="cap.pt"), infer_max_segs=4)
            assert fm.infer_max_segs == 4
            fm.classify_iq(_noise(500_000))       # must not be slow and must not raise

        @c.check("a missing .meta.json is reported, not ignored")
        def _():
            lonely = os.path.join(tmp, "lonely.pt")
            torch.save(SpecCNN(2, base=8).state_dict(), lonely)
            try:
                FingerprintModel(lonely)
            except FileNotFoundError:
                return
            raise AssertionError("a model with no meta loaded without an error")

        return c.report()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run(main)
