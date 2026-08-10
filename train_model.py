"""The trainer for the spectrogram fingerprint classifier.

Each top-level folder in --data_dir is one class. The session_* subfolders give the
split: the program trains on the early sessions and measures the accuracy with the
last session. A session that the program did not train on gives the only correct
accuracy.

The program reads the raw IQ captures from terminal.py. It writes a .pt file and
a .meta.json file. The GUI reads the two files with the same module.

    python train_model.py                              # balanced, the PRESET below
    python train_model.py --preset fast                # a quick check
    python train_model.py --preset best                # slow, most accurate
    python train_model.py --preset best --epochs 15    # a flag has precedence
    python train_model.py --max_segs_per_class 4000    # less memory
"""

import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (TensorDataset, DataLoader, WeightedRandomSampler,
                              Dataset as TorchDataset)

from fp_spectrogram import (iq_to_spectrogram, remove_dc, SpecCNN, VOTE_THRESH,
                            N_FFT, STFT_HOP, SEG_LEN, SEG_HOP)

DATA_DIR   = "./fingerprint_data"
OUTPUT     = "./trained_model.pt"   # the name that terminal.py loads at the start
BATCH_SIZE = 512
EPOCHS     = 8
LR         = 1e-3
BASE_CH    = 16
UNKNOWN_THRESH = 0.8
SEED       = 42

# ── the budget for memory and time. A CLI flag has precedence. ────────────────
PRESET              = "balanced"  # the preset for a run without --preset (from the IDE)
QUICK               = False      # True = the same as --preset fast
MAX_FILES_PER_CLASS = 5000       # 0 = all; .iq files for each class and session
MAX_SEGS_PER_FILE   = 150        # 0 = all; segments for each file
MAX_SEGS_PER_CLASS  = 20000      # 0 = all; segments for each class and split
STORE_DTYPE         = "float16"  # the cache in memory: "float16" or "float32"
FORCE_CPU           = False      # True = use the CPU although CUDA is available

# ── the presets for --preset. The times are for a CPU. ────────────────────────
PRESETS = {
    # a quick check (1-2 min). The accuracy is not the final value.
    "fast":     dict(epochs=5,  max_files_per_class=15, max_segs_per_file=40,
                     max_segs_per_class=8000,  base_ch=16),
    # the usual run (minutes)
    "balanced": dict(epochs=EPOCHS, max_files_per_class=MAX_FILES_PER_CLASS,
                     max_segs_per_file=MAX_SEGS_PER_FILE,
                     max_segs_per_class=MAX_SEGS_PER_CLASS, base_ch=BASE_CH),
    # the most accurate run (hours on a CPU): more data and a larger network
    "best":     dict(epochs=30, max_files_per_class=0, max_segs_per_file=0,
                     max_segs_per_class=40000, base_ch=24),
}

# ── the energy gate: it removes the quiet segments from the device classes ────
GATE_DEVICE_SEGS = True          # True = keep only the segments above the noise floor
GATE_MARGIN_DB   = 3.0           # dB that a device segment must be above the floor
NOISE_CLASS      = "noise"       # the class that gives the floor. The gate ignores it.

# ── the SNR augmentation: it puts the device segments into recorded noise ─────
SNR_AUG_P      = 0.5             # the part of the train device segments to mix (0 = off)
SNR_AUG_DB     = (0.0, 20.0)     # the range of the target signal-to-noise ratio in dB
NOISE_POOL_MAX = 2000            # the noise segments in the cache (64 MB at 4096)

# ── the frequency-shift augmentation: it moves the device segments in frequency ─
FREQ_SHIFT_FRAC = 0.10           # the maximum shift as a part of the sample rate (0 = off)

# ── the weak-signal accuracy: the same val segments, but at a low SNR ─────────
WEAK_VAL_DB = (0.0, 10.0)        # the SNR range for the weak copies

# ── SpecAugment-lite: it hides a part of each spectrogram during the training ──
SPEC_MASK_P    = 0.5             # the probability that a batch has a mask (0 = off)
SPEC_MASK_FREQ = 32              # the maximum masked frequency bins (of 256)
SPEC_MASK_TIME = 12              # the maximum masked time frames (of 61)


def _natkey(s):
    """Give a sort key that puts session_10 after session_9, and not after session_1."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _seg_powers_db(raw, seg_len, seg_hop):
    """Give the mean power in dB of each segment of one IQ buffer."""
    k = (len(raw) - seg_len) // seg_hop + 1
    if k <= 0:
        return np.empty(0, np.float64)
    p = np.array([np.mean(np.abs(raw[i * seg_hop:i * seg_hop + seg_len]) ** 2)
                  for i in range(k)], dtype=np.float64)
    return 10.0 * np.log10(p + 1e-12)


def _mix_noise(seg, noise_pool, rng, snr_lo, snr_hi):
    """Put a device segment into recorded noise at a random signal-to-noise ratio.

    The function decreases the power of the strong capture to snr dB above the power
    of a noise segment. Then it adds that noise. The added noise becomes the new
    floor. Thus the result is a correct simulation of a weak signal."""
    noise = noise_pool[rng.randint(len(noise_pool))]
    ps = float(np.mean(np.abs(seg) ** 2))
    pn = float(np.mean(np.abs(noise) ** 2))
    if ps <= 0.0 or pn <= 0.0:
        return seg
    g = np.sqrt(pn / ps * 10.0 ** (rng.uniform(snr_lo, snr_hi) / 10.0))
    return (seg * np.complex64(g) + noise).astype(np.complex64, copy=False)


def _freq_shift(seg, rng, max_frac):
    """Move a segment in frequency by a random part of the sample rate.

    The recordings always put the signal at the same spectrogram bins. Thus the
    network can learn the position and not the fingerprint. A random shift prevents
    this. A live lock and a channel change also move the signal."""
    nu = rng.uniform(-max_frac, max_frac)               # cycles for each sample
    ramp = np.exp((2j * np.pi * nu) * np.arange(len(seg)))
    return (seg * ramp).astype(np.complex64, copy=False)


def file_to_segments(path: Path, seg_len: int, seg_hop: int, max_segs: int = 0,
                     min_power_db=None) -> np.ndarray:
    """Give the raw IQ segments (k, seg_len) of one .iq file. None if there are none.

    The DC offset goes here, before the gate measures the power and before any
    frequency shift. A shift puts the offset away from 0 Hz, and a mean can not
    find it after that.
    min_power_db removes the segments that are more quiet than that value. These
    quiet segments are the gaps between the bursts, thus they are noise. The gate
    uses the original power, thus a quiet segment can not become a device.
    max_segs keeps that many segments at equal distances."""
    raw = remove_dc(np.fromfile(str(path), dtype=np.complex64))
    if len(raw) < seg_len:
        return None
    k = (len(raw) - seg_len) // seg_hop + 1
    sel = np.arange(k)
    if min_power_db is not None:
        sel = sel[_seg_powers_db(raw, seg_len, seg_hop) >= min_power_db]
        if len(sel) == 0:
            return None
    if max_segs and len(sel) > max_segs:
        sel = sel[np.unique(np.linspace(0, len(sel) - 1, max_segs).round().astype(int))]
    return np.stack([remove_dc(raw[i * seg_hop:i * seg_hop + seg_len]) for i in sel])


def augment_segments(segs, rng, noise_pool=None, aug_p=SNR_AUG_P,
                     snr_db=SNR_AUG_DB, f_shift=0.0):
    """Shift the segments in frequency, then put them into recorded noise.

    The order matters. The shift is first, thus the noise that the mix adds becomes
    the floor at the place where a real floor would be."""
    if rng is None:
        return segs
    out = segs
    if f_shift > 0:
        out = [_freq_shift(s, rng, f_shift) for s in out]
    if noise_pool:
        out = [_mix_noise(s, noise_pool, rng, *snr_db) if rng.rand() < aug_p else s
               for s in out]
    return out


def segments_to_specs(segs, store_dtype=np.float32):
    return np.stack([iq_to_spectrogram(s) for s in segs]).astype(store_dtype,
                                                                 copy=False)


def file_to_specs(path: Path, seg_len: int, seg_hop: int, max_segs: int = 0,
                  store_dtype=np.float32, min_power_db=None,
                  noise_pool=None, aug_p=SNR_AUG_P, snr_db=SNR_AUG_DB,
                  f_shift=0.0, rng=None) -> np.ndarray:
    """Make the spectrograms (k, 1, N_FFT, frames) of one .iq file.

    The val split and the weak-signal copy use this function, because they must not
    change between two epochs. The train split keeps the raw segments and it makes
    the images in SegmentDataset instead."""
    segs = file_to_segments(path, seg_len, seg_hop, max_segs, min_power_db)
    if segs is None:
        return None
    return segments_to_specs(augment_segments(segs, rng, noise_pool, aug_p,
                                              snr_db, f_shift), store_dtype)


class _Reservoir:
    """Keep a uniform random sample of k rows from a stream of unknown length.

    load_split used to build a whole class and then remove the extra rows. The peak
    memory was the full uncapped class, and --preset best makes that very large.
    The reservoir never holds more than k rows."""

    def __init__(self, k, rng):
        self.k, self.rng, self.n, self.buf = int(k or 0), rng, 0, []

    def extend(self, block):
        for row in block:
            if not self.k or len(self.buf) < self.k:
                self.buf.append(np.array(row, copy=True))
            else:
                j = self.rng.randint(self.n + 1)
                if j < self.k:
                    self.buf[j] = np.array(row, copy=True)
            self.n += 1

    def stack(self):
        return np.stack(self.buf) if self.buf else None


class SegmentDataset(TorchDataset):
    """Make a spectrogram from a raw IQ segment, with a new augmentation each time.

    The augmentation runs here and not at the load. Thus every epoch sees another
    realisation of the noise and of the frequency shift. A cache of images gives one
    realisation for the whole run, and 30 epochs then see the same image 30 times.

    One segment costs about 320 us of CPU. A raw complex64 segment of 4096 samples
    is 32 kB, and a float16 image of that segment is 31 kB. Thus the memory does not
    change."""

    def __init__(self, segs, labels, aug_ok, noise_pool, seed,
                 snr_aug_p=0.0, snr_db=SNR_AUG_DB, f_shift=0.0):
        self.segs, self.labels, self.aug_ok = segs, labels, aug_ok
        self.pool = list(noise_pool) if noise_pool else None
        self.rng = np.random.RandomState(seed)
        self.snr_aug_p, self.snr_db, self.f_shift = snr_aug_p, snr_db, f_shift

    def __len__(self):
        return len(self.segs)

    def __getitem__(self, i):
        seg = self.segs[i]
        if self.aug_ok[i]:
            if self.f_shift > 0:
                seg = _freq_shift(seg, self.rng, self.f_shift)
            if self.pool and self.rng.rand() < self.snr_aug_p:
                seg = _mix_noise(seg, self.pool, self.rng, *self.snr_db)
        return torch.from_numpy(iq_to_spectrogram(seg)), int(self.labels[i])


def load_split(data_dir: Path, seg_len: int, seg_hop: int, rng, *,
               max_files: int = 0, max_segs_file: int = 0, max_segs_class: int = 0,
               store_dtype=np.float32, gate: bool = False, gate_margin_db: float = 3.0,
               noise_class: str = "noise", snr_aug_p: float = 0.0,
               f_shift: float = 0.0):
    """Read the dataset. A folder is a class. A session gives the split.

    Gives (classes, Str, ytr, Atr, Xva, yva, Xwk, ywk, noise_pool).

      * Str is the raw IQ of the train segments, (n, seg_len) complex64.
        SegmentDataset makes the image and augments it again at every epoch.
      * Atr says which train segments may be augmented: the device classes that
        have two sessions or more.
      * Xva and Xwk are spectrograms, made one time. An evaluation set must not
        change between two epochs.

    The caps max_files, max_segs_file and max_segs_class limit the memory and the
    time. max_segs_class also prevents a large noise class in the loss. A reservoir
    holds the cap while the program reads, thus the peak memory is the cap.

    gate=True removes from each device class the segments that are below the noise
    floor. The floor is the 95th percentile of the noise class plus gate_margin_db.
    A parked capture of a device is mostly noise between the bursts. Without the gate
    these segments have the label of the device, and the classifier fails.

    snr_aug_p and f_shift do not augment anything here. They decide only whether the
    function collects a noise pool, and what the log says. The function also makes a
    weak copy of the device val segments at WEAK_VAL_DB, and that copy is fixed."""
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found under {data_dir}.")

    # One pass on the train sessions of the noise class gives the gate threshold and
    # the pool for the augmentation. The noise of the val session stays out of both.
    gate_thr, noise_pool = None, []
    if noise_class in classes and (gate or snr_aug_p > 0):
        nfiles = defaultdict(list)
        for f in (data_dir / noise_class).rglob("*.iq"):
            nfiles[f.parent.name].append(f)
        sess = sorted(nfiles, key=_natkey)
        train_sess = sess[:-1] if len(sess) >= 2 else sess
        files = sorted(f for s in train_sess for f in nfiles[s])[:80]
        per_file = max(1, NOISE_POOL_MAX // max(1, len(files)))
        npw = []
        for f in files:
            raw = remove_dc(np.fromfile(str(f), dtype=np.complex64))
            if len(raw) < seg_len:
                continue
            if gate:
                npw.append(_seg_powers_db(raw, seg_len, seg_hop))
            if snr_aug_p > 0:
                k = (len(raw) - seg_len) // seg_hop + 1
                for i in rng.permutation(k)[:per_file]:
                    noise_pool.append(
                        remove_dc(raw[i * seg_hop:i * seg_hop + seg_len]))
        if gate and npw:
            ceiling = float(np.percentile(np.concatenate(npw), 95))
            gate_thr = ceiling + gate_margin_db
            print(f"  [gate] noise floor 95th-pct {ceiling:.1f} dB -> keep non-noise "
                  f"segments >= {gate_thr:.1f} dB  (margin +{gate_margin_db:.0f} dB)")
        elif gate:
            print("  [gate] noise class has no usable files — gate disabled")
    elif gate:
        print(f"  [gate] no '{noise_class}' class found — energy gate disabled")
    if snr_aug_p > 0:
        if noise_pool:
            print(f"  [aug]  weak-signal SNR aug: p={snr_aug_p:.2f}, "
                  f"{SNR_AUG_DB[0]:.0f}-{SNR_AUG_DB[1]:.0f} dB, "
                  f"{len(noise_pool)} noise segs pooled")
        else:
            print("  [aug]  no noise segments available — SNR augmentation disabled")
    if f_shift > 0:
        print(f"  [aug]  freq-shift aug: ±{f_shift:.2f} × fs on train device segments")
    if snr_aug_p > 0 or f_shift > 0:
        print("  [aug]  both run again at every epoch, not once at the load")

    Str, ytr, Atr, Xva, yva, Xwk, ywk = [], [], [], [], [], [], []
    for lab, cls in enumerate(classes):
        files_by_sess = defaultdict(list)
        for f in (data_dir / cls).rglob("*.iq"):
            files_by_sess[f.parent.name].append(f)
        sessions = sorted(files_by_sess, key=_natkey)
        if not sessions:
            print(f"  {cls:<10}: no .iq files, skipped")
            continue

        if len(sessions) >= 2:
            val_sessions, random_val = {sessions[-1]}, False
        else:
            val_sessions, random_val = set(), True
            print(f"  [warn] class '{cls}' has a single session ({sessions[0]}); "
                  f"using a RANDOM split — that val accuracy is optimistic.")

        min_pdb = None if (gate_thr is None or cls == noise_class) else gate_thr
        # A random split does not divide the train data and the val data by session.
        # Thus a class with one session stays without augmentation.
        aug_dev = cls != noise_class and not random_val
        # The reservoirs hold the cap. Thus the peak memory is the cap and not the
        # full class. See the defect #8.
        tr_res = _Reservoir(max_segs_class, rng)
        va_res = _Reservoir(max_segs_class, rng)
        wk_res = _Reservoir(max_segs_class, rng)
        for sess, files in files_by_sess.items():
            if max_files and len(files) > max_files:        # keep a random subset
                files = [files[i] for i in rng.permutation(len(files))[:max_files]]
            for f in files:
                segs = file_to_segments(f, seg_len, seg_hop, max_segs_file, min_pdb)
                if segs is None:
                    continue
                if random_val:
                    perm = rng.permutation(len(segs))
                    n_va = max(1, int(0.2 * len(segs)))
                    va_res.extend(segments_to_specs(segs[perm[:n_va]], store_dtype))
                    tr_res.extend(segs[perm[n_va:]])
                elif sess in val_sessions:
                    va_res.extend(segments_to_specs(segs, store_dtype))
                    # The weak copy uses the noise of the train pool. That makes the
                    # task more difficult only, thus the accuracy stays correct.
                    if aug_dev and noise_pool:
                        wk_res.extend(segments_to_specs(
                            augment_segments(segs, rng, noise_pool, 1.0,
                                             WEAK_VAL_DB, 0.0), store_dtype))
                else:
                    # The train split keeps the raw IQ. SegmentDataset makes the
                    # image and augments it again at every epoch. See the defect #7.
                    tr_res.extend(segs)

        tr, va, wk = tr_res.stack(), va_res.stack(), wk_res.stack()
        n_tr = 0 if tr is None else len(tr)
        n_va = 0 if va is None else len(va)
        n_wk = 0 if wk is None else len(wk)
        if n_tr:
            Str.append(tr); ytr.append(np.full(n_tr, lab, np.int64))
            Atr.append(np.full(n_tr, aug_dev, bool))
        if n_va:
            Xva.append(va); yva.append(np.full(n_va, lab, np.int64))
        if n_wk:
            Xwk.append(wk); ywk.append(np.full(n_wk, lab, np.int64))

        held = "random" if random_val else f"session {sorted(val_sessions)[0]}"
        weak = f" + {n_wk:,} weak" if n_wk else ""
        print(f"  {cls:<10}: {n_tr:,} train / {n_va:,} val{weak}  (val = {held})")

    if not Str or not Xva:
        raise RuntimeError("Not enough data to form both a train and a val split.")
    return (classes,
            np.concatenate(Str), np.concatenate(ytr), np.concatenate(Atr),
            np.concatenate(Xva), np.concatenate(yva),
            np.concatenate(Xwk) if Xwk else None,
            np.concatenate(ywk) if ywk else None,
            noise_pool)


def print_confusion(model, loader, classes, device):
    """Print the confusion matrix and the per-class figures. Give them back too.

    The caller writes them to a .metrics.json file. The numbers are what a report
    needs, and a print alone loses them."""
    model.eval()
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device).float()).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                cm[int(t), int(p)] += 1
    print("\nConfusion matrix (val — rows = true, cols = pred):")
    print(" " * 11 + "".join(f"{c:>12}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>10} " + "".join(f"{cm[i, j]:>12,}" for j in range(n)))
    print("\nPer-class (val):")
    per_class = {}
    for i, c in enumerate(classes):
        tp, col, row = int(cm[i, i]), int(cm[:, i].sum()), int(cm[i, :].sum())
        prec = tp / col if col else 0.0
        rec  = tp / row if row else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": row, "predicted": col}
        print(f"  {c:>10}: precision {prec:6.1%}   recall {rec:6.1%}   f1 {f1:6.1%}")
    return cm.tolist(), per_class


def spec_mask(xb, p=SPEC_MASK_P):
    """Hide a frequency band and a time stripe of each image, with probability p.

    Each image of the batch gets its own mask and its own draw. One mask for the
    full batch gives the network the same hole 512 times, thus it learns much less.

    The value 0 is the mean of an image, because each image is standardized. The
    function multiplies and it does not write in place, because xb can share memory
    with the cache. The widths are clamped to the size of the image."""
    b, _c, nf, nt = xb.shape
    dev = xb.device
    fw = torch.randint(1, min(SPEC_MASK_FREQ, nf) + 1, (b,), device=dev)
    fs = (torch.rand(b, device=dev) * (nf - fw + 1).float()).long()
    tw = torch.randint(1, min(SPEC_MASK_TIME, nt) + 1, (b,), device=dev)
    ts = (torch.rand(b, device=dev) * (nt - tw + 1).float()).long()
    on = (torch.rand(b, device=dev) < p)          # which images get a mask at all
    fi = torch.arange(nf, device=dev)[None, :]
    ti = torch.arange(nt, device=dev)[None, :]
    fkeep = ~(((fi >= fs[:, None]) & (fi < (fs + fw)[:, None])) & on[:, None])
    tkeep = ~(((ti >= ts[:, None]) & (ti < (ts + tw)[:, None])) & on[:, None])
    return xb * fkeep[:, None, :, None] * tkeep[:, None, None, :]


def _git_commit():
    """Give the short commit of the working tree, or None outside a repository."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL, text=True).strip() or None
    except Exception:
        return None


def train(args):
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available())
                          else "cuda")
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32

    if args.preset == "fast":
        print("\n*** FAST PRESET — small caps + few epochs. Sanity check only; "
              "the val number here is NOT final. ***")

    print(f"\nPreset : {args.preset}   |   {args.epochs} epochs, base_ch {args.base_ch}")
    print(f"Device : {device}   |   cache dtype {store_dtype.__name__}")
    print(f"STFT   : {N_FFT}-pt, hop {STFT_HOP}   |   segment {args.seg_len}/{args.seg_hop}")
    caps = []
    if args.max_files_per_class: caps.append(f"<={args.max_files_per_class} files/class/session")
    if args.max_segs_per_file:   caps.append(f"<={args.max_segs_per_file} segs/file")
    if args.max_segs_per_class:  caps.append(f"<={args.max_segs_per_class} segs/class/split")
    print(f"Caps   : {', '.join(caps) if caps else 'none (full dataset)'}")
    print("\nLoading captures…")
    classes, Str, ytr, Atr, Xva, yva, Xwk, ywk, noise_pool = load_split(
        Path(args.data_dir), args.seg_len, args.seg_hop, rng,
        max_files=args.max_files_per_class, max_segs_file=args.max_segs_per_file,
        max_segs_class=args.max_segs_per_class, store_dtype=store_dtype,
        gate=GATE_DEVICE_SEGS, gate_margin_db=args.gate_margin_db,
        noise_class=NOISE_CLASS, snr_aug_p=args.snr_aug_p,
        f_shift=args.freq_shift_frac)

    present = np.unique(ytr)
    if len(present) < 2:
        raise RuntimeError(
            f"Only one class has training data: '{classes[present[0]]}'. A model of "
            f"one class reaches 100% and it means nothing. Record a second class, "
            f"and a '{NOISE_CLASS}' class as well.")
    empty = [c for i, c in enumerate(classes) if i not in present]
    if empty:
        print(f"  [warn] no train segments for {empty}. Those classes get an output "
              f"that the network can never learn.")

    mem = (Str.nbytes + Xva.nbytes) / 1e6
    frames = (args.seg_len - N_FFT) // STFT_HOP + 1
    print(f"\nClasses  : {classes}")
    print(f"Train/Val: {len(Str):,} raw segments / {len(Xva):,} spectrograms"
          f"   image (1, {N_FFT}, {frames})   (~{mem:.0f} MB cached)")
    print(f"Aug      : {int(Atr.sum()):,} of {len(Atr):,} train segments are "
          f"augmented again at every epoch")

    tr_ds = SegmentDataset(Str, ytr, Atr, noise_pool, args.seed,
                           snr_aug_p=args.snr_aug_p, snr_db=SNR_AUG_DB,
                           f_shift=args.freq_shift_frac)
    va_ds = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))

    counts = np.bincount(ytr, minlength=len(classes))
    w_cls  = 1.0 / (counts + 1e-6)
    sw     = torch.tensor([w_cls[l] for l in ytr], dtype=torch.float)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    model = SpecCNN(len(classes), base=args.base_ch).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"Params   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValAcc':>9}  {'LR':>9}")
    print("-" * 52)

    best_acc, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = tc = tn = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            if SPEC_MASK_P:
                xb = spec_mask(xb, SPEC_MASK_P)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(yb)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(yb)

        model.eval()
        vc = vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device).float(), yb.to(device)
                vc += (model(xb).argmax(1) == yb).sum().item()
                vn += len(yb)

        sched.step()
        tr_acc, va_acc = tc / tn, vc / vn
        mark = "  <- best" if va_acc > best_acc else ""
        print(f"{epoch:>6}  {tl/tn:>10.4f}  {tr_acc:>8.2%}  {va_acc:>8.2%}  "
              f"{sched.get_last_lr()[0]:>9.2e}{mark}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:              # the accuracy can stay 0.0 after a failure
        model.load_state_dict(best_state)
    confusion, per_class = print_confusion(model, va_loader, classes, device)

    # The weak-signal accuracy: the same val segments, but at a low SNR. This value
    # decreases if the program becomes worse for a weak signal.
    weak_acc, weak_per_class = None, {}
    if Xwk is not None:
        wk_loader = DataLoader(TensorDataset(torch.from_numpy(Xwk),
                                             torch.from_numpy(ywk)),
                               batch_size=args.batch_size, shuffle=False)
        model.eval()
        hit = np.zeros(len(classes), np.int64)
        tot = np.zeros(len(classes), np.int64)
        with torch.no_grad():
            for xb, yb in wk_loader:
                pred = model(xb.to(device).float()).argmax(1).cpu().numpy()
                for t, p in zip(yb.numpy(), pred):
                    tot[t] += 1; hit[t] += int(t == p)
        weak_acc = float(hit.sum() / max(1, tot.sum()))
        weak_per_class = {classes[i]: {"accuracy": float(hit[i] / tot[i]),
                                       "support": int(tot[i])}
                          for i in range(len(classes)) if tot[i]}
        per = "   ".join(f"{classes[i]} {hit[i]/tot[i]:.1%}"
                         for i in range(len(classes)) if tot[i])
        print(f"\nWeak-signal val ({WEAK_VAL_DB[0]:.0f}-{WEAK_VAL_DB[1]:.0f} dB SNR): "
              f"{weak_acc:.2%}   ({per})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    meta = {
        "classes"       : classes,
        "n_classes"     : len(classes),
        "n_fft"         : N_FFT,
        "stft_hop"      : STFT_HOP,
        "seg_len"       : args.seg_len,
        "seg_hop"       : args.seg_hop,
        "base_ch"       : args.base_ch,
        "unknown_thresh": args.unknown_thresh,
        "vote_thresh"   : args.vote_thresh,
        "val_acc"       : best_acc,
        "model"         : "SpecCNN",
        "representation": "stft256_logmag_1ch",
        "preset"        : args.preset,
        "snr_aug_p"     : args.snr_aug_p,
        "snr_aug_db"    : list(SNR_AUG_DB),
        "freq_shift_frac": args.freq_shift_frac,
        "weak_val_acc"  : weak_acc,
        # Provenance. Without it you can not say later which code and which flags
        # made a model, and a report needs that.
        "git_commit"    : _git_commit(),
        "trained_at"    : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args"          : {k: v for k, v in sorted(vars(args).items())},
    }
    # Use with_suffix. Thus the path is always the same as the path that
    # FingerprintModel calculates with splitext.
    with open(Path(args.out).with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # The figures of the report go in their own file. A print alone loses them, and
    # the meta must stay small because the GUI reads it at every load.
    metrics = {
        "classes"        : classes,
        "val_acc"        : best_acc,
        "weak_val_acc"   : weak_acc,
        "weak_val_db"    : list(WEAK_VAL_DB),
        "confusion"      : confusion,
        "confusion_rows" : "true", "confusion_cols": "pred",
        "per_class"      : per_class,
        "weak_per_class" : weak_per_class,
        "n_train"        : int(len(Str)),
        "n_val"          : int(len(Xva)),
        "n_weak"         : 0 if Xwk is None else int(len(Xwk)),
        "epochs"         : args.epochs,
        "preset"         : args.preset,
        "git_commit"     : meta["git_commit"],
        "trained_at"     : meta["trained_at"],
        "train_minutes"  : (time.time() - t0) / 60.0,
    }
    metrics_path = Path(args.out).with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[ok] Model -> {args.out}")
    print(f"     Metrics -> {metrics_path}")
    print(f"     Best val accuracy : {best_acc:.2%}")
    print(f"     Total time        : {(time.time() - t0)/60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Spectrogram RF-fingerprint trainer")
    p.add_argument("--data_dir",       default=DATA_DIR)
    p.add_argument("--out",            default=OUTPUT)
    p.add_argument("--seg_len",        type=int,   default=SEG_LEN)
    p.add_argument("--seg_hop",        type=int,   default=SEG_HOP)
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--unknown_thresh", type=float, default=UNKNOWN_THRESH,
                   help="the probability of the mean of a buffer that gives a name")
    p.add_argument("--vote_thresh",    type=float, default=VOTE_THRESH,
                   help="the probability of one segment that gives a vote. It is "
                        "lower than unknown_thresh, because one segment is 0.4 ms.")
    p.add_argument("--seed",           type=int,   default=SEED)
    # The default of these flags is None. Thus the program can find a value that
    # the user gives, and use the preset for the other values.
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="fast = a quick check | balanced = the usual run "
                        "(default) | best = the slowest and the most accurate")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--max_files_per_class", type=int, default=None,
                   help="the maximum .iq files for each class and session (0 = all)")
    p.add_argument("--max_segs_per_file",   type=int, default=None,
                   help="the maximum segments for each file (0 = all)")
    p.add_argument("--max_segs_per_class",  type=int, default=None,
                   help="the maximum segments for each class and split (0 = all)")
    p.add_argument("--store_dtype", choices=["float16", "float32"], default=STORE_DTYPE,
                   help="the data type of the cache in memory (float16 uses one half)")
    p.add_argument("--gate_margin_db", type=float, default=GATE_MARGIN_DB,
                   help="the dB that a device segment must be above the noise floor. "
                        "To stop the gate, set GATE_DEVICE_SEGS to False in the code.")
    p.add_argument("--snr_aug_p", type=float, default=SNR_AUG_P,
                   help="the part of the train device segments to put into recorded "
                        "noise at a random signal-to-noise ratio (0 = off). "
                        "SNR_AUG_DB in the code gives the range.")
    p.add_argument("--freq_shift_frac", type=float, default=FREQ_SHIFT_FRAC,
                   help="move the train device segments in frequency by this part of "
                        "the sample rate (0 = off)")
    p.add_argument("--quick", action="store_true", default=QUICK,
                   help="the same as --preset fast")
    p.add_argument("--cpu", action="store_true", default=FORCE_CPU,
                   help="use the CPU although CUDA is available")
    args = p.parse_args()
    # the order of precedence: a CLI flag, then --preset, then the PRESET constant
    if args.preset is None:
        args.preset = "fast" if args.quick else PRESET
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


if __name__ == "__main__":
    train(parse_args())
