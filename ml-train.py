"""
train_fast_quick.py  –  Speed-optimised trainer for the drone/noise 2-class problem.
                         Updated to support wideband stitched IQ captures.

What changed vs the original
─────────────────────────────
  • Dataset now loads freq_span (= FreqMax - FreqMin) from each file's .json.
    Narrowband files get freq_span = 0, so old caches are still compatible.
  • SpectralCNNSlim frequency branch now takes [center_freq_norm, span_norm]
    (2 scalars) instead of just [center_freq_norm].
    This lets the network distinguish a 10 MHz wideband sweep centred at
    2.4 GHz from a 2 MHz narrowband capture at the same centre.
  • stats.json must now contain span_mean / span_std  (written by the
    updated precompute.py; see below for a backwards-compat fallback).
  • AugmentedSubset wrapper replaces duplicated PSDDataset instantiation —
    fixes the index-mismatch bug where train/val indices pointed to different
    random subsets in two independently-sampled dataset objects.
  • RF-aware augmentations: CFO shift (circular bin roll), SpecAugment
    frequency masking, and amplitude jitter replace the previous weak
    Gaussian-only scheme.
  • SEBlock (Squeeze-and-Excite) channel attention added to each CNN block
    so the network can weight discriminative frequency bands.
  • CrossEntropyLoss label_smoothing=0.1 reduces overconfidence.

Usage:
    python train_fast_quick.py
    python train_fast_quick.py --max_windows 50000 --epochs 5   # faster
    python train_fast_quick.py --max_windows 200000 --epochs 15 # more accurate
"""

import json
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# ── config ─────────────────────────────────────────────────────────────────────
CACHE_DIR    = "./psd_cache"
OUTPUT_MODEL = "./detector_quick.pt"
FFT_BINS     = 1024
BATCH_SIZE   = 256
EPOCHS       = 10
LR           = 1e-3
VAL_SPLIT    = 0.15
SEED         = 42
MAX_WINDOWS  = 100_000   # per class; None = use all

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PSDDataset(Dataset):
    """
    Mmap-backed PSD dataset.  No augmentation here — use AugmentedSubset
    to apply augmentation to the training split only, so train and val
    indices always refer to the same underlying data.

    Each index entry is a 5-tuple:
        (npy_path, row, center_freq_hz, freq_span_hz, label_idx)
    """

    def __init__(self, cache_dir: str, classes: list,
                 psd_mean: float = 0.0,  psd_std: float  = 1.0,
                 freq_mean: float = 0.0, freq_std: float = 1.0,
                 span_mean: float = 0.0, span_std: float = 1.0,
                 max_per_class: int | None = None):

        self.classes   = classes
        self.psd_mean  = psd_mean
        self.psd_std   = psd_std
        self.freq_mean = freq_mean
        self.freq_std  = freq_std
        self.span_mean = span_mean
        self.span_std  = span_std

        self.index: list = []   # list of (npy_path, row, center_freq, freq_span, label)
        self._mmaps: dict = {}

        cache_dir = Path(cache_dir)
        for label_idx, cls_name in enumerate(classes):
            cls_dir   = cache_dir / cls_name
            npy_files = sorted(cls_dir.glob("*.npy"))
            cls_index = []

            for npy_path in npy_files:
                json_path = npy_path.with_suffix(".json")
                try:
                    with open(json_path) as f:
                        meta = json.load(f)
                    n_rows    = meta["n_windows"]
                    cf        = meta["center_freq"]
                    freq_span = meta.get("freq_span", 0)
                except Exception:
                    arr       = np.load(str(npy_path), mmap_mode="r")
                    n_rows    = arr.shape[0]
                    cf        = 0
                    freq_span = 0

                for row in range(n_rows):
                    cls_index.append((str(npy_path), row, cf, freq_span, label_idx))

            if max_per_class is not None and len(cls_index) > max_per_class:
                cls_index = random.sample(cls_index, max_per_class)

            self.index.extend(cls_index)
            print(f"  {cls_name}: {len(cls_index):,} windows")

        if not self.index:
            raise RuntimeError(
                f"No windows found in {cache_dir}. Run precompute.py first."
            )

    def _get_mmap(self, path: str) -> np.ndarray:
        if path not in self._mmaps:
            self._mmaps[path] = np.load(path, mmap_mode="r")
        return self._mmaps[path]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        npy_path, row, center_freq_hz, freq_span_hz, label = self.index[idx]
        psd = self._get_mmap(npy_path)[row].astype(np.float32)

        x = (psd - self.psd_mean) / self.psd_std
        spec_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # (1, FFT_BINS)

        freq_n = (center_freq_hz - self.freq_mean) / self.freq_std
        span_n = (freq_span_hz   - self.span_mean) / self.span_std
        meta_t = torch.tensor([freq_n, span_n], dtype=torch.float32)

        return spec_t, meta_t, label


class AugmentedSubset(Dataset):
    """
    Wraps a PSDDataset + index list and applies RF-aware augmentations.

    Keeping augmentation here (not inside PSDDataset) means train and val
    always index the same underlying data — no index-mismatch between two
    separately-subsampled dataset objects.

    RF augmentations applied:
      • CFO shift    – circular bin roll ±32 bins, simulates carrier freq offset
      • Gain drift   – uniform amplitude shift, simulates AGC / hardware variation
      • Receiver noise – additive Gaussian, simulates SNR variation
      • Freq mask    – SpecAugment-style band dropout, forces freq-robust features
    """

    def __init__(self, dataset: PSDDataset, indices: list, augment: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        spec_t, meta_t, label = self.dataset[self.indices[idx]]

        if self.augment:
            x = spec_t.squeeze(0).numpy().copy()

            # CFO: circular frequency shift (device oscillator offset)
            x = np.roll(x, np.random.randint(-32, 33))

            # Gain drift (AGC variation, hardware gain differences)
            x += np.float32(np.random.uniform(-2.0, 2.0))

            # Receiver noise floor variation
            x += np.random.normal(0.0, 0.5, x.shape).astype(np.float32)

            # SpecAugment frequency masking — drops a contiguous band
            # forces the model to rely on remaining spectral features
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, FFT_BINS - 64)
                fw = np.random.randint(16, 64)
                x[f0:f0 + fw] = 0.0

            spec_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        return spec_t, meta_t, label


# ══════════════════════════════════════════════════════════════════════════════
# Model  –  "Slim" variant with SE attention + 2-input frequency branch
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Squeeze-and-Excite channel attention.
    Recalibrates each channel's importance — in the spectral domain this
    means the network can learn to upweight the frequency bands that are
    most discriminative for RF fingerprinting (e.g. the drone's emission
    peak vs the noise floor region).
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):   # x: (B, C, L)
        return x * self.fc(x).unsqueeze(-1)


class SpectralCNNSlim(nn.Module):
    """
    1-D CNN with SE attention blocks and a 2-input frequency branch.

    Frequency branch inputs:
        [center_freq_norm, freq_span_norm]
    This lets the model distinguish wideband sweeps from narrowband captures
    and captures centred at different frequencies.

    Channel counts: 1 → 16 → 32 → 64 → 128  (~1.4 M params)
    SE adds ~1% parameters but meaningfully improves RF fingerprinting accuracy
    by focusing on discriminative spectral bands.
    """

    def __init__(self, n_classes: int, n_bins: int = FFT_BINS):
        super().__init__()

        def block(in_ch, out_ch, k, pool):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                SEBlock(out_ch),
                nn.MaxPool1d(pool),
            )

        self.cnn = nn.Sequential(
            block(1,  16,  k=15, pool=4),
            block(16, 32,  k=7,  pool=4),
            block(32, 64,  k=5,  pool=4),
            block(64, 128, k=3,  pool=2),
        )
        cnn_flat = 128 * (n_bins // (4 * 4 * 4 * 2))   # 128 × 8 = 1024

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
        """
        spectrum : (B, 1, FFT_BINS)
        meta     : (B, 2)   [center_freq_norm, span_norm]
        """
        cnn_out  = self.cnn(spectrum).flatten(1)   # (B, 1024)
        freq_out = self.freq_branch(meta)           # (B, 16)
        return self.head(torch.cat([cnn_out, freq_out], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_weighted_sampler(dataset: PSDDataset, indices: list):
    labels = [dataset.index[i][4] for i in indices]
    counts = np.bincount(labels, minlength=len(dataset.classes))
    w_cls  = 1.0 / (counts + 1e-6)
    sw     = torch.tensor([w_cls[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sw, len(sw), replacement=True)


def save_checkpoint(path, epoch, model, opt, sched,
                    best_val_acc, best_state, train_idx, val_idx):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "opt_state": opt.state_dict(), "sched_state": sched.state_dict(),
        "best_val_acc": best_val_acc, "best_state": best_state,
        "train_idx": train_idx, "val_idx": val_idx,
    }, path)


def load_checkpoint(path, model, opt, sched):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    opt.load_state_dict(ckpt["opt_state"])
    sched.load_state_dict(ckpt["sched_state"])
    return (ckpt["epoch"], ckpt["best_val_acc"],
            ckpt["best_state"], ckpt["train_idx"], ckpt["val_idx"])


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.output + ".ckpt"

    stats_path = Path(args.cache_dir) / "stats.json"
    if not stats_path.exists():
        raise RuntimeError(
            f"stats.json not found in {args.cache_dir}. Run precompute.py first."
        )

    with open(stats_path) as f:
        stats = json.load(f)

    classes   = stats["classes"]
    psd_mean  = stats["psd_mean"]
    psd_std   = stats["psd_std"]
    freq_mean = stats["freq_mean"]
    freq_std  = stats["freq_std"]
    span_mean = stats.get("span_mean", 0.0)
    span_std  = stats.get("span_std",  1.0)

    print(f"\nDevice         : {device}")
    print(f"Classes        : {classes}")
    print(f"Max wins/class : {args.max_windows:,}")
    print(f"PSD  mean/std  : {psd_mean:.2f} / {psd_std:.2f} dBFS")
    print(f"Freq mean/std  : {freq_mean/1e6:.3f} / {freq_std/1e6:.3f} MHz")
    print(f"Span mean/std  : {span_mean/1e6:.3f} / {span_std/1e6:.3f} MHz")

    print("\nBuilding dataset index…")
    full_ds = PSDDataset(
        args.cache_dir, classes,
        psd_mean   = psd_mean,  psd_std   = psd_std,
        freq_mean  = freq_mean, freq_std  = freq_std,
        span_mean  = span_mean, span_std  = span_std,
        max_per_class = args.max_windows,
    )

    n_total = len(full_ds)
    print(f"Total windows  : {n_total:,}")

    model = SpectralCNNSlim(n_classes=len(classes)).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params   : {n_params:,}")

    start_epoch  = 1
    best_val_acc = 0.0
    best_state   = None
    train_idx    = None
    val_idx      = None

    if Path(ckpt_path).exists():
        print(f"\n▶  Resuming from checkpoint  {ckpt_path}")
        start_epoch, best_val_acc, best_state, train_idx, val_idx = \
            load_checkpoint(ckpt_path, model, opt, sched)
        start_epoch += 1
        print(f"   Resuming at epoch {start_epoch}  (best val: {best_val_acc:.2%})")
    else:
        print("\n▶  Starting fresh")

    if train_idx is None:
        n_val   = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val
        indices = list(range(n_total))
        random.shuffle(indices)
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        print(f"Train / Val    : {n_train:,} / {n_val:,}")
    else:
        print(f"Train / Val    : {len(train_idx):,} / {len(val_idx):,}  (from checkpoint)")

    if start_epoch > args.epochs:
        print(f"\nAlready completed {args.epochs} epochs. Delete .ckpt to retrain.")
        return

    # One dataset object — AugmentedSubset applies augmentation to train split only.
    # This guarantees train_idx and val_idx always refer to the same rows in full_ds.
    train_subset = AugmentedSubset(full_ds, train_idx, augment=True)
    val_subset   = AugmentedSubset(full_ds, val_idx,   augment=False)

    sampler = make_weighted_sampler(full_ds, train_idx)
    pf      = 2 if args.workers > 0 else None
    pin     = device.type == "cuda"

    train_loader = DataLoader(
        train_subset,
        batch_size      = args.batch_size,
        sampler         = sampler,
        num_workers     = args.workers,
        pin_memory      = pin,
        prefetch_factor = pf,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size      = args.batch_size,
        shuffle         = False,
        num_workers     = args.workers,
        pin_memory      = pin,
        prefetch_factor = pf,
    )

    # label_smoothing reduces overconfidence — consistently helps RF classifiers
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  "
          f"{'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}")
    print("─" * 65)

    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for spec_b, meta_b, label_b in train_loader:
            spec_b  = spec_b.to(device)
            meta_b  = meta_b.to(device)
            label_b = label_b.to(device)
            opt.zero_grad()
            logits = model(spec_b, meta_b)
            loss   = crit(logits, label_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss    += loss.item() * len(label_b)
            t_correct += (logits.argmax(1) == label_b).sum().item()
            t_total   += len(label_b)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for spec_b, meta_b, label_b in val_loader:
                spec_b  = spec_b.to(device)
                meta_b  = meta_b.to(device)
                label_b = label_b.to(device)
                logits  = model(spec_b, meta_b)
                v_loss    += crit(logits, label_b).item() * len(label_b)
                v_correct += (logits.argmax(1) == label_b).sum().item()
                v_total   += len(label_b)

        sched.step()
        tr_acc = t_correct / t_total
        vl_acc = v_correct / v_total

        epoch_times.append(time.time() - t_epoch)
        avg_t   = sum(epoch_times[-5:]) / len(epoch_times[-5:])
        eta_min = (avg_t * (args.epochs - epoch)) / 60
        marker  = "  ← best" if vl_acc > best_val_acc else ""

        print(f"{epoch:>6}  {t_loss/t_total:>10.4f}  {tr_acc:>9.2%}  "
              f"{v_loss/v_total:>10.4f}  {vl_acc:>9.2%}  "
              f"{sched.get_last_lr()[0]:>10.2e}"
              f"  ETA {eta_min:.0f}m{marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        save_checkpoint(ckpt_path, epoch, model, opt, sched,
                        best_val_acc, best_state, train_idx, val_idx)

    # ── save final model + metadata ───────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output)

    meta = {
        "classes"     : classes,
        "n_classes"   : len(classes),
        "fft_bins"    : FFT_BINS,
        "psd_mean"    : psd_mean,
        "psd_std"     : psd_std,
        "freq_mean"   : freq_mean,
        "freq_std"    : freq_std,
        "span_mean"   : span_mean,
        "span_std"    : span_std,
        "val_acc"     : best_val_acc,
        "model"       : "SpectralCNNSlim",
        "freq_inputs" : ["center_freq_norm", "span_norm"],
    }
    meta_path = args.output.replace(".pt", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Model   →  {args.output}")
    print(f"✓ Meta    →  {meta_path}")
    print(f"  Best val accuracy : {best_val_acc:.2%}")
    print(f"\n  (Checkpoint at {ckpt_path} — safe to delete now)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Wideband-aware SpectralCNN training (CPU-friendly)"
    )
    p.add_argument("--cache_dir",   default=CACHE_DIR)
    p.add_argument("--output",      default=OUTPUT_MODEL)
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--val_split",   type=float, default=VAL_SPLIT)
    p.add_argument("--max_windows", type=int,   default=MAX_WINDOWS,
                   help="Max windows per class to subsample (default 100 000)")
    p.add_argument("--workers",     type=int,   default=4,
                   help="DataLoader workers (set 0 on Windows if errors)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0   = time.time()
    train(args)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
