"""
train_drone.py  –  Self-contained drone/noise detector trainer.

One script, one run: raw .iq captures  ->  trained model.
No PSD cache, no checkpoint/resume — just load, gate, train, save.
(Replaces the old two-stage  ml-precomp.py -> ml-train.py  pipeline.)

Why burst-gating matters
─────────────────────────
The drone emission is bursty.  Each .iq capture is sliced into ~1 ms FFT
windows, so most "drone" windows were recorded *between* bursts and contain
only the noise floor — they are effectively mislabeled.  diagnose_psd.py
measured ~85% of drone windows as statistically indistinguishable from noise,
which caps any model at ~58% accuracy (see diag_mean_psd.png: the class means
sit on top of each other, but diag_burst_score.png shows drone has a real
population out at 30-60 dB peak-above-floor where noise has nothing).

This script removes that label noise by keeping a drone window only when it
actually contains a burst:

    burst_score = max(psd) - median(psd)   (dB, per window)
    keep drone window  iff  burst_score >= --min_burst_db

Noise windows are kept as-is (silent floor is a valid negative).  Gating leaves
the classes imbalanced, so a WeightedRandomSampler rebalances the train split.

IMPORTANT — the inference side must match (terminal-ml.py):
    Training keeps burst windows; inference must NOT average windows together
    (that dilutes a burst back into the noise floor).  terminal-ml.py classifies
    the most-burst-like window per sweep and peak-holds across sweeps.

Frequency branch
────────────────
Pluto scans several hops and stitches them into one wideband spectrum, so each
capture spans the whole scanned band.  The model keeps a small freq branch fed
[center_freq_norm, span_norm] so it knows which band/span it is looking at.  If
you always scan the same band these inputs are constant and the branch simply
learns to ignore them — harmless, and it keeps the saved model compatible with
terminal-ml.py's inference engine.

Usage
─────
    python train_drone.py
    python train_drone.py --min_burst_db 30 --epochs 50
    python train_drone.py --min_burst_db 0          # disable gating (old behaviour)
    python train_drone.py --base_channels 8 --dropout 0.5 --weight_decay 1e-3
"""

import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── config ───────────────────────────────────────────────────────────────────
DATA_DIR      = "./training_data"
OUTPUT_MODEL  = "./detector_quick.pt"
FFT_BINS      = 1024
WINDOW_HOP    = 512        # samples between successive PSD windows
FLOOR_DB      = -120.0     # clamp dead-bin outliers so they don't dominate stats

BATCH_SIZE    = 256
EPOCHS        = 50
LR            = 1e-3
VAL_SPLIT     = 0.15
SEED          = 42

MIN_BURST_DB  = 30.0       # drone windows below this peak-above-floor are dropped
BASE_CHANNELS = 16         # conv widths scale 1->C->2C->4C->8C; lower to regularise
DROPOUT       = 0.3
WEIGHT_DECAY  = 1e-4

_BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Raw IQ -> PSD windows  (+ XML metadata)
# ══════════════════════════════════════════════════════════════════════════════

def load_meta(iq_path: Path) -> dict:
    """Read the fields the trainer needs from the paired .xml.

    center_freq / freq_span describe the (stitched) band this capture covers and
    feed the model's freq branch.  Wideband captures carry FreqMin/FreqMax; for
    narrowband captures span falls back to 0.  Defaults are used if the .xml is
    missing or unparseable so a stray capture never crashes the run.
    """
    out = dict(center_freq=0, scale_factor=1.0, freq_span=0)
    xml_path = iq_path.with_suffix(".xml")
    if not xml_path.exists():
        return out
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return out

    def get(tag, cast, default):
        el = root.find(tag)
        if el is None or not el.text:
            return default
        try:
            return cast(el.text.strip())
        except (ValueError, TypeError):
            return default

    out["center_freq"]  = get("CenterFrequency", int,   0)
    out["scale_factor"] = get("ScaleFactor",     float, 1.0)
    freq_min = get("FreqMin", int, None)
    freq_max = get("FreqMax", int, None)
    if freq_min is not None and freq_max is not None:
        out["freq_span"] = freq_max - freq_min
    return out


def compute_psd_windows(raw: np.ndarray, scale: float) -> np.ndarray:
    """Overlapping Blackman-windowed PSDs in dBFS — (n_windows, FFT_BINS)."""
    n = (len(raw) - FFT_BINS) // WINDOW_HOP + 1
    if n <= 0:
        return np.empty((0, FFT_BINS), dtype=np.float32)
    strides = (raw.strides[0] * WINDOW_HOP, raw.strides[0])
    chunks  = np.lib.stride_tricks.as_strided(raw, shape=(n, FFT_BINS), strides=strides)
    windowed = chunks * np.float32(scale) * _BLACKMAN
    psds = 20.0 * np.log10(
        np.abs(np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)) / FFT_BINS + 1e-10
    )
    np.maximum(psds, FLOOR_DB, out=psds)
    return psds.astype(np.float32)


def burst_score(windows: np.ndarray) -> np.ndarray:
    """Peak-above-floor (dB) per window: high = burst, low = silent gap."""
    return windows.max(axis=1) - np.median(windows, axis=1)


def load_dataset(data_dir: Path, classes: list, min_burst_db: float):
    """
    Load every capture, compute PSD windows, gate drone windows by burst score.

    Returns:
        X        (N, FFT_BINS) float32   raw dBFS windows (not yet normalised)
        y        (N,)          int64     class index
        file_id  (N,)          int64     which capture file each window came from
                                         (used for a leak-free file-level split)
        cf       (N,)          float64   center_freq (Hz) of the capture
        span     (N,)          float64   freq span (Hz) of the capture
    """
    X_parts, y_parts, fid_parts, cf_parts, span_parts = [], [], [], [], []
    fid = 0
    for label, cls in enumerate(classes):
        iq_files = sorted((data_dir / cls).glob("*.iq"))
        kept_w = raw_w = 0
        for fpath in iq_files:
            meta = load_meta(fpath)
            raw = np.fromfile(str(fpath), dtype=np.complex64)
            w = compute_psd_windows(raw, meta["scale_factor"])
            if len(w) == 0:
                continue
            raw_w += len(w)

            # Burst-gate the drone class only; noise floor is a valid negative.
            if cls == "drone" and min_burst_db > 0:
                w = w[burst_score(w) >= min_burst_db]
                if len(w) == 0:
                    continue
            kept_w += len(w)

            X_parts.append(w)
            y_parts.append(np.full(len(w), label, dtype=np.int64))
            fid_parts.append(np.full(len(w), fid, dtype=np.int64))
            cf_parts.append(np.full(len(w), meta["center_freq"], dtype=np.float64))
            span_parts.append(np.full(len(w), meta["freq_span"], dtype=np.float64))
            fid += 1
        gate = f"  (gated {raw_w:,}->{kept_w:,})" if cls == "drone" and min_burst_db > 0 \
               else ""
        print(f"  {cls:<6}: {len(iq_files)} files, {kept_w:,} windows{gate}")

    if not X_parts:
        raise RuntimeError(f"No usable windows found under {data_dir}.")
    return (np.concatenate(X_parts), np.concatenate(y_parts),
            np.concatenate(fid_parts), np.concatenate(cf_parts),
            np.concatenate(span_parts))


# ══════════════════════════════════════════════════════════════════════════════
# Dataset wrapper (per-bin normalisation + optional RF augmentation)
# ══════════════════════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    def __init__(self, X, y, cf, span, bin_mean, bin_std,
                 freq_mean, freq_std, span_mean, span_std, augment=False):
        self.X = X
        self.y = y
        self.cf = cf
        self.span = span
        self.bin_mean  = bin_mean
        self.bin_std   = bin_std
        self.freq_mean = freq_mean
        self.freq_std  = freq_std
        self.span_mean = span_mean
        self.span_std  = span_std
        self.augment   = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.bin_mean) / self.bin_std   # per-bin standardise

        if self.augment:
            x = x.copy()
            x = np.roll(x, np.random.randint(-8, 9))                       # CFO drift
            x += np.float32(np.random.uniform(-0.2, 0.2))                  # gain drift
            x += np.random.normal(0.0, 0.08, x.shape).astype(np.float32)   # rx noise
            if np.random.rand() < 0.3:                                     # freq mask
                f0 = np.random.randint(0, FFT_BINS - 32)
                x[f0:f0 + np.random.randint(8, 32)] = 0.0

        freq_n = (self.cf[idx]   - self.freq_mean) / self.freq_std
        span_n = (self.span[idx] - self.span_mean) / self.span_std
        meta_t = torch.tensor([freq_n, span_n], dtype=torch.float32)

        return (torch.from_numpy(x.astype(np.float32)).unsqueeze(0),
                meta_t, int(self.y[idx]))


# ══════════════════════════════════════════════════════════════════════════════
# Model — 1-D spectral CNN with SE attention + 2-input frequency branch.
# Architecture mirrors terminal-ml.py's inference net so the saved state_dict
# loads there without changes.
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False), nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class SpectralCNNSlim(nn.Module):
    def __init__(self, n_classes, n_bins=FFT_BINS, base_channels=16, dropout=0.3):
        super().__init__()

        def block(ci, co, k, pool):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(co), nn.GELU(), SEBlock(co), nn.MaxPool1d(pool),
            )

        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        self.cnn = nn.Sequential(
            block(1,  c1, 15, 4), block(c1, c2, 7, 4),
            block(c2, c3, 5, 4),  block(c3, c4, 3, 2),
        )
        cnn_flat = c4 * (n_bins // (4 * 4 * 4 * 2))
        self.freq_branch = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 16),
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_flat + 16, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, spectrum, meta):
        cnn_out  = self.cnn(spectrum).flatten(1)
        freq_out = self.freq_branch(meta)
        return self.head(torch.cat([cnn_out, freq_out], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════════════════

def file_level_split(y, file_id, val_split, rng):
    """Split whole capture files (not windows) so near-duplicate windows from one
    capture never straddle train/val.  Stratified by class."""
    files_by_class = defaultdict(list)
    idx_by_file    = defaultdict(list)
    for gi, fid in enumerate(file_id):
        idx_by_file[int(fid)].append(gi)
    for fid, idxs in idx_by_file.items():
        files_by_class[int(y[idxs[0]])].append(fid)

    train_idx, val_idx = [], []
    for lab, fids in files_by_class.items():
        fids = sorted(fids)
        rng.shuffle(fids)
        n_val = max(1, round(len(fids) * val_split))
        val_files = set(fids[:n_val])
        for fid in fids:
            (val_idx if fid in val_files else train_idx).extend(idx_by_file[fid])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def train(args):
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    classes  = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    print(f"\nDevice         : {device}")
    print(f"Classes        : {classes}")
    print(f"Burst gate     : >= {args.min_burst_db} dB (drone)"
          if args.min_burst_db > 0 else "Burst gate     : OFF")
    print(f"Augmentation   : {'OFF' if args.no_augment else 'ON (weak)'}")

    print("\nLoading captures…")
    X, y, file_id, cf, span = load_dataset(data_dir, classes, args.min_burst_db)
    print(f"Total windows  : {len(X):,}  "
          f"({int((y==0).sum()):,} {classes[0]} / {int((y==1).sum()):,} {classes[1]})")

    train_idx, val_idx = file_level_split(y, file_id, args.val_split, rng)
    print(f"Train / Val    : {len(train_idx):,} / {len(val_idx):,} windows "
          f"(file-level split)")

    # All normalisation stats from TRAIN windows only (no val leakage).
    Xtr = X[train_idx]
    bin_mean = Xtr.mean(0)
    bin_std  = Xtr.std(0)
    bin_std[bin_std < 1e-6] = 1.0

    freq_mean = float(cf[train_idx].mean())
    freq_std  = float(cf[train_idx].std())
    span_mean = float(span[train_idx].mean())
    span_std  = float(span[train_idx].std())
    # Guard degenerate stddevs (single fixed band => std 0 => freq branch sees a
    # constant, which is fine; just avoid divide-by-zero).
    if freq_std < 1e-6: freq_std = 1.0
    if span_std < 1e-6: span_std = 1.0

    print(f"Freq mean/std  : {freq_mean/1e6:.3f} / {freq_std/1e6:.3f} MHz")
    print(f"Span mean/std  : {span_mean/1e6:.3f} / {span_std/1e6:.3f} MHz")

    def make_ds(idx, augment):
        return WindowDataset(
            X[idx], y[idx], cf[idx], span[idx], bin_mean, bin_std,
            freq_mean, freq_std, span_mean, span_std, augment=augment)

    train_ds = make_ds(train_idx, augment=not args.no_augment)
    val_ds   = make_ds(val_idx,   augment=False)

    # WeightedRandomSampler rebalances the gating-induced class imbalance.
    counts = np.bincount(y[train_idx], minlength=len(classes))
    w_cls  = 1.0 / (counts + 1e-6)
    sw     = torch.tensor([w_cls[l] for l in y[train_idx]], dtype=torch.float)
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers)

    model = SpectralCNNSlim(len(classes), base_channels=args.base_channels,
                            dropout=args.dropout).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Model params   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}")
    print("-" * 65)

    best_val_acc, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl = tc = tn = 0
        for xb, mb, yb in train_loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb, mb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(yb)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(yb)

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
                logits = model(xb, mb)
                vl += crit(logits, yb).item() * len(yb)
                vc += (logits.argmax(1) == yb).sum().item()
                vn += len(yb)

        sched.step()
        tr_acc, vl_acc = tc / tn, vc / vn
        marker = "  <- best" if vl_acc > best_val_acc else ""
        print(f"{epoch:>6}  {tl/tn:>10.4f}  {tr_acc:>9.2%}  "
              f"{vl/vn:>10.4f}  {vl_acc:>9.2%}  {sched.get_last_lr()[0]:>10.2e}{marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── save best model + metadata ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output)
    meta = {
        "classes"      : classes,
        "n_classes"    : len(classes),
        "fft_bins"     : FFT_BINS,
        "window_hop"   : WINDOW_HOP,
        "floor_db"     : FLOOR_DB,
        "min_burst_db" : args.min_burst_db,
        "psd_bin_mean" : bin_mean.tolist(),
        "psd_bin_std"  : bin_std.tolist(),
        "freq_mean"    : freq_mean,
        "freq_std"     : freq_std,
        "span_mean"    : span_mean,
        "span_std"     : span_std,
        "val_acc"      : best_val_acc,
        "model"        : "SpectralCNNSlim",
        "base_channels": args.base_channels,
        "freq_inputs"  : ["center_freq_norm", "span_norm"],
    }
    meta_path = args.output.replace(".pt", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[ok] Model -> {args.output}")
    print(f"[ok] Meta  -> {meta_path}")
    print(f"     Best val accuracy : {best_val_acc:.2%}")
    print(f"     Total time        : {(time.time() - t0) / 60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Single-file drone/noise trainer")
    p.add_argument("--data_dir",      default=DATA_DIR)
    p.add_argument("--output",        default=OUTPUT_MODEL)
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=LR)
    p.add_argument("--val_split",     type=float, default=VAL_SPLIT)
    p.add_argument("--seed",          type=int,   default=SEED)
    p.add_argument("--min_burst_db",  type=float, default=MIN_BURST_DB,
                   help="Drop drone windows whose peak-above-floor is below this. "
                        "0 disables gating.")
    p.add_argument("--base_channels", type=int,   default=BASE_CHANNELS)
    p.add_argument("--dropout",       type=float, default=DROPOUT)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--workers",       type=int,   default=0,
                   help="DataLoader workers (0 is safest on Windows)")
    p.add_argument("--no-augment",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
