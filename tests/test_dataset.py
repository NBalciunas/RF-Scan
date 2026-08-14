"""Self-check for the data path of the trainer. Type `python tests/test_dataset.py`.

The check builds a small fingerprint_data tree in a temporary directory: two drones
and a noise class, two sessions each, and each device capture is half burst and half
silence. Then it puts that tree through load_split.

The three properties that decide whether a training run means anything are:
the session split, the energy gate, and which segments the augmentation touches.
"""

import io
import json
import shutil
import tempfile
import contextlib
from pathlib import Path

import numpy as np
import torch

from _support import Checks, run
# dataset_info moved to tools/ on 2026-08-13. The folder has no __init__.py and it
# needs none: a namespace package works from the repo root that _support puts on
# sys.path, which is what tests/test_prepare_clip.py does with transmitting/.
from tools import dataset_info
from train_model import (load_split, file_to_specs, file_to_segments,
                         SegmentDataset, _seg_powers_db, _natkey, WEAK_VAL_DB,
                         _dataset_sample_rate)
from fp_spectrogram import SEG_LEN, SEG_HOP, N_FFT

NOISE_SIGMA = 0.01
FILE_LEN = SEG_LEN + 7 * SEG_HOP          # 8 segments for each file


def _noise(n, sigma=NOISE_SIGMA, rng=None):
    r = rng or np.random.RandomState(0)
    return ((r.randn(n) + 1j * r.randn(n)) * (sigma / np.sqrt(2))).astype(np.complex64)


def _device_capture(f_norm, rng):
    """Half burst and half silence, as a real parked capture is."""
    iq = _noise(FILE_LEN, rng=rng)
    half = FILE_LEN // 2
    iq[:half] += (np.exp(2j * np.pi * f_norm * np.arange(half))).astype(np.complex64)
    return iq


def build_tree(root, sessions=2, files=2):
    """Write noise/, droneA/ and droneB/ with `sessions` sessions each."""
    rng = np.random.RandomState(42)
    for s in range(1, sessions + 1):
        for cls, f_norm in (("droneA", 0.15), ("droneB", -0.20)):
            d = root / cls / f"session_{s}"
            d.mkdir(parents=True)
            for i in range(files):
                _device_capture(f_norm, rng).tofile(d / f"{cls}_s{s}_{i}.iq")
        d = root / "noise" / f"session_{s}"
        d.mkdir(parents=True)
        for i in range(files):
            _noise(FILE_LEN, rng=rng).tofile(d / f"noise_s{s}_{i}.iq")
    return root


def _load(root, seed=0, **kw):
    """Run load_split quietly. Gives (result, the text that it printed)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = load_split(root, SEG_LEN, SEG_HOP, np.random.RandomState(seed), **kw)
    return res, buf.getvalue()


def main():
    c = Checks("Trainer data path (train_model.py)")
    tmp = Path(tempfile.mkdtemp(prefix="rfscan_data_"))
    try:
        root = build_tree(tmp / "fingerprint_data")

        # ── The split ─────────────────────────────────────────────────────────

        @c.check("a folder is a class, and the classes are sorted")
        def _():
            (classes, *_rest), _out = _load(root)
            assert classes == ["droneA", "droneB", "noise"], classes

        @c.check("the last session is held out, and it is not in the train data")
        def _():
            (_cls, Str, ytr, _atr, Xva, yva, _w, _yw, _p), out = _load(root)
            assert "val = session session_2" in out, out
            # 2 files x 8 segments, one session for train and one for val
            for lab in (0, 1, 2):
                assert (ytr == lab).sum() == 16, (lab, (ytr == lab).sum())
                assert (yva == lab).sum() == 16, (lab, (yva == lab).sum())
            assert Str.shape[1:] == (SEG_LEN,), Str.shape       # raw IQ, not images
            assert Str.dtype == np.complex64, Str.dtype
            assert Xva.shape[1:] == (1, N_FFT, 61), Xva.shape   # val stays images

        @c.check("one session alone gives a random split and a warning")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_solo_"))
            try:
                build_tree(solo / "fingerprint_data", sessions=1, files=3)
                (_cls, Str, _y, _a, Xva, _yv, _w, _yw, _p), out = \
                    _load(solo / "fingerprint_data")
                assert "[warn]" in out and "single session" in out, out
                assert "RANDOM split" in out, out
                assert len(Xva) > 0 and len(Str) > 0
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("session_10 sorts after session_9, not after session_1")
        def _():
            names = ["session_1", "session_10", "session_2", "session_9"]
            assert sorted(names, key=_natkey) == \
                ["session_1", "session_2", "session_9", "session_10"]

        @c.check("a tree with no captures is refused with a clear message")
        def _():
            empty = Path(tempfile.mkdtemp(prefix="rfscan_empty_"))
            try:
                (empty / "droneA" / "session_1").mkdir(parents=True)
                try:
                    _load(empty)
                except RuntimeError as e:
                    assert "train and a val split" in str(e), e
                    return
                raise AssertionError("an empty tree did not raise")
            finally:
                shutil.rmtree(empty, ignore_errors=True)

        # ── The energy gate ───────────────────────────────────────────────────

        @c.check("the energy gate removes the silence between the bursts")
        def _():
            (_c0, _S0, yo, *_r0), _o0 = _load(root, gate=False)
            (_c1, _S1, yg, *_r1), out = _load(root, gate=True)
            assert "[gate] noise floor" in out, out
            before, after = int((yo == 0).sum()), int((yg == 0).sum())
            c.note(f"droneA train segments: {before} without the gate, {after} with it")
            assert 0 < after < before, f"{before} -> {after}"

        @c.check("the gate never removes a segment of the noise class")
        def _():
            (_c0, _S0, yo, *_r0), _o0 = _load(root, gate=False)
            (_c1, _S1, yg, *_r1), _o1 = _load(root, gate=True)
            assert (yo == 2).sum() == (yg == 2).sum(), "the noise class was gated"

        @c.check("with no noise class the gate says so and does not run")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_nonoise_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                shutil.rmtree(d / "noise")
                (_r, out) = _load(d, gate=True)
                assert "energy gate disabled" in out, out
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("_seg_powers_db gives the mean power of each segment in dB")
        def _():
            raw = np.concatenate([np.full(SEG_LEN, 1.0, np.complex64),
                                  np.full(SEG_LEN, 0.1, np.complex64)])
            got = _seg_powers_db(raw, SEG_LEN, SEG_LEN)
            assert len(got) == 2, got
            assert abs(got[0] - 0.0) < 1e-6, got[0]           # power 1   -> 0 dB
            assert abs(got[1] - (-20.0)) < 1e-6, got[1]       # power .01 -> -20 dB

        # ── The augmentation ──────────────────────────────────────────────────

        @c.check("the load does not augment: the train split is raw IQ")
        def _():
            # The augmentation moved into SegmentDataset, thus the loader gives the
            # same raw segments whatever the settings say. See the defect #7.
            (_c0, S0, *_r0), _o0 = _load(root, snr_aug_p=0.0, f_shift=0.0)
            (_c1, S1, *_r1), _o1 = _load(root, snr_aug_p=1.0, f_shift=0.2)
            assert np.array_equal(S0, S1), "the loader augmented the train data"

        @c.check("the val split never changes with the augmentation settings")
        def _():
            (_c0, _S0, _y0, _a0, Xva0, *_r0), _o0 = _load(root, snr_aug_p=0.0)
            (_c1, _S1, _y1, _a1, Xva1, *_r1), _o1 = _load(root, snr_aug_p=1.0)
            assert np.array_equal(Xva0, Xva1), "the val split was augmented"

        @c.check("a device segment gives a new image at every epoch")
        def _():
            # This is the whole point of #7. A cache of images gives one realisation
            # for the run, and 30 epochs then see the same picture 30 times.
            (_c, Str, ytr, Atr, *_r, pool), _o = _load(root, snr_aug_p=1.0,
                                                       f_shift=0.10)
            ds = SegmentDataset(Str, ytr, Atr, pool, seed=0,
                                snr_aug_p=1.0, f_shift=0.10)
            i = int(np.flatnonzero(Atr)[0])
            a, la = ds[i]
            b, lb = ds[i]
            assert la == lb, "the label moved"
            assert a.shape == (1, N_FFT, 61), a.shape
            assert not torch.equal(a, b), "the two epochs gave the same image"

        @c.check("a noise segment is never augmented, in any epoch")
        def _():
            (_c, Str, ytr, Atr, *_r, pool), _o = _load(root, snr_aug_p=1.0,
                                                       f_shift=0.10)
            j = int(np.flatnonzero(~Atr)[0])
            ds = SegmentDataset(Str, ytr, Atr, pool, seed=0,
                                snr_aug_p=1.0, f_shift=0.10)
            assert torch.equal(ds[j][0], ds[j][0]), "a noise segment was augmented"
            assert ytr[j] == 2, "the unaugmented class is not noise"

        @c.check("the per-class cap is held while loading, not after")
        def _():
            # A reservoir keeps the cap during the read, thus the peak memory is the
            # cap and not the whole class. See the defect #8.
            (_c, Str, ytr, *_r), _o = _load(root, max_segs_class=5)
            for lab in (0, 1, 2):
                assert (ytr == lab).sum() <= 5, (lab, (ytr == lab).sum())
            assert len(Str) == len(ytr)

        @c.check("the weak-signal copy exists and covers the device classes only")
        def _():
            (_c, _S, _y, _a, _Xva, _yv, Xwk, ywk, _p), out = _load(root, snr_aug_p=0.5)
            assert Xwk is not None and len(Xwk) > 0, "no weak copy was built"
            assert set(np.unique(ywk).tolist()) == {0, 1}, np.unique(ywk)
            assert len(Xwk) == len(ywk), (len(Xwk), len(ywk))
            assert "weak" in out, out
            c.note(f"weak copy: {len(Xwk)} segments at "
                   f"{WEAK_VAL_DB[0]:.0f}-{WEAK_VAL_DB[1]:.0f} dB SNR")

        @c.check("with no noise pool the SNR augmentation says so and stops")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_nopool_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                shutil.rmtree(d / "noise")
                (_r, out) = _load(d, snr_aug_p=0.5)
                assert "SNR augmentation disabled" in out or "no noise" in out, out
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("a capture shorter than one segment is skipped, not a crash")
        def _():
            short = Path(tmp) / "short.iq"
            _noise(SEG_LEN // 2).tofile(short)
            assert file_to_specs(short, SEG_LEN, SEG_HOP) is None

        @c.check("the frequency shift moves the signal of a device segment")
        def _():
            f = Path(tmp) / "tone.iq"
            (np.exp(2j * np.pi * 0.05 * np.arange(FILE_LEN))
             ).astype(np.complex64).tofile(f)
            rng = np.random.RandomState(5)
            plain = file_to_specs(f, SEG_LEN, SEG_HOP)
            moved = file_to_specs(f, SEG_LEN, SEG_HOP, f_shift=0.2, rng=rng)
            rows_p = {int(s[0].mean(axis=1).argmax()) for s in plain}
            rows_m = {int(s[0].mean(axis=1).argmax()) for s in moved}
            assert rows_p == {N_FFT // 2 + 13}, rows_p
            assert rows_m != rows_p, "the shift moved nothing"

        @c.check("the frequency shift does not move a receiver DC artifact")
        def _():
            # The shift runs on train device segments only. If the DC artifact moved
            # with the signal, "the artifact is not in the middle" would become a
            # class cue that has nothing to do with a drone. The DC blocker removes
            # the artifact first, thus there is nothing left to move.
            f = Path(tmp) / "dconly.iq"
            (np.full(FILE_LEN, 10.0, np.complex64) + _noise(FILE_LEN)).tofile(f)
            for shift, rng in ((0.0, None), (0.2, np.random.RandomState(5))):
                specs = file_to_specs(f, SEG_LEN, SEG_HOP, f_shift=shift, rng=rng)
                worst = max(float(s[0].mean(axis=1).max()
                                  - np.median(s[0].mean(axis=1))) for s in specs)
                assert worst < 1.0, \
                    f"shift {shift}: one row stands {worst:.2f} sigma above the rest"

        @c.check("a device class and the noise class get the same DC treatment")
        def _():
            # The asymmetry that mattered: the device class is shifted and the noise
            # class is not. Both must end with the DC row looking like any other.
            (_cls, Str, ytr, Atr, *_r, pool), _o = _load(root, snr_aug_p=0.5,
                                                         f_shift=0.10)
            ds = SegmentDataset(Str, ytr, Atr, pool, seed=1,
                                snr_aug_p=0.5, f_shift=0.10)
            for lab, name in ((0, "droneA"), (2, "noise")):
                idx = np.flatnonzero(ytr == lab)
                imgs = np.stack([ds[i][0].numpy()[0] for i in idx])
                rows = imgs.mean(axis=(0, 2))
                excess = float(rows[N_FFT // 2] - np.median(rows))
                c.note(f"{name:<7} DC row sits {excess:+.2f} sigma from the median")
                assert abs(excess) < 1.0, f"{name}: {excess:+.2f} sigma"

        # ── The inventory tool ────────────────────────────────────────────────

        @c.check("dataset_info counts the files, the segments and the sessions")
        def _():
            tree = dataset_info.scan(root)
            assert sorted(tree) == ["droneA", "droneB", "noise"], sorted(tree)
            assert sorted(tree["droneA"]) == ["session_1", "session_2"]
            s = tree["droneA"]["session_1"]
            assert s["files"] == 2, s["files"]
            assert s["segments"] == 16, s["segments"]      # 2 files x 8 segments

        @c.check("dataset_info reads the seconds and the gain from the sidecars")
        def _():
            # The synthetic tree has no .json sidecars, thus the tool must say so
            # instead of reporting a wrong duration.
            tree = dataset_info.scan(root)
            s = tree["droneA"]["session_1"]
            assert s["no_sidecar"] == 2, s["no_sidecar"]
            assert s["seconds"] == 0.0 and not s["gain"]

        @c.check("dataset_info warns about a class with one session only")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_inv1_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d, sessions=1, files=2)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dataset_info.report(d)
                out = buf.getvalue()
                assert "has 1 session" in out, out
                assert "random split" in out, out
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("dataset_info warns when the noise class is missing")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_inv2_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                shutil.rmtree(d / "noise")
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dataset_info.report(d)
                assert "there is no 'noise' class" in buf.getvalue()
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("dataset_info warns when the dataset mixes RX gains")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_inv3_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                for i, f in enumerate(sorted(d.rglob("*.iq"))):
                    f.with_suffix(".json").write_text(json.dumps(
                        {"sample_rate": 10e6, "gain_db": 10 + 20 * (i % 2),
                         "center_freq": 2.44e9}))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dataset_info.report(d)
                out = buf.getvalue()
                assert "mixes RX gains" in out, out
                assert "seconds" in out and "2440.000" in out, out
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("the meta takes the sample rate from the sidecars of the dataset")
        def _():
            # The rate is not in the code, thus it must come from the data. It goes
            # in the meta, and the GUI compares it against the radio.
            solo = Path(tempfile.mkdtemp(prefix="rfscan_sr1_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                for f in sorted(d.rglob("*.iq")):
                    f.with_suffix(".json").write_text(json.dumps(
                        {"sample_rate": 10e6, "gain_db": 10}))
                assert _dataset_sample_rate(d) == 10e6, _dataset_sample_rate(d)
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("two sample rates in one dataset give None and a warning")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_sr2_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                for i, f in enumerate(sorted(d.rglob("*.iq"))):
                    f.with_suffix(".json").write_text(json.dumps(
                        {"sample_rate": 10e6 if i % 2 else 5e6, "gain_db": 10}))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    sr = _dataset_sample_rate(d)
                assert sr is None, sr
                assert "mixes the sample rates" in buf.getvalue(), buf.getvalue()
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("a dataset with no sidecar gives None and says so")
        def _():
            solo = Path(tempfile.mkdtemp(prefix="rfscan_sr3_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    sr = _dataset_sample_rate(d)
                assert sr is None, sr
                assert "no sidecar gives a sample rate" in buf.getvalue(), buf.getvalue()
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("dataset_info finds a capture that the disk truncated")
        def _():
            # §8 #20. tofile on a full disk writes a short file and np.fromfile reads
            # it without a complaint. The sidecar holds the count that was intended,
            # thus the two can be compared.
            solo = Path(tempfile.mkdtemp(prefix="rfscan_inv4_"))
            try:
                d = solo / "fingerprint_data"
                build_tree(d)
                files = sorted(d.rglob("*.iq"))
                for f in files:
                    f.with_suffix(".json").write_text(json.dumps(
                        {"sample_rate": 10e6, "gain_db": 10, "center_freq": 2.44e9,
                         "n_samples": FILE_LEN}))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dataset_info.report(d)
                assert "truncated" not in buf.getvalue(), "a warning with no cause"

                # Cut one file in half, as a disk that filled would.
                victim = files[0]
                half = np.fromfile(str(victim), dtype=np.complex64)[:FILE_LEN // 2]
                half.tofile(str(victim))
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    dataset_info.report(d)
                out = buf.getvalue()
                assert "truncated" in out, out
                assert victim.name in out, out
            finally:
                shutil.rmtree(solo, ignore_errors=True)

        @c.check("dataset_info refuses a folder that does not exist")
        def _():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = dataset_info.report(Path(tmp) / "nothing_here")
            assert rc == 1 and "does not exist" in buf.getvalue()

        return c.report()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run(main)
