"""
train_fingerprint_spec.py — spectrogram fingerprint trainer (RFUAV-style).

Folder-driven: every top-level folder under --data_dir is one class, and session_*
subfolders drive a session-held-out split (train on early sessions, validate on a
session never seen in training; the only honest number). Representation and model:

  * representation: the paper's 256-point STFT spectrogram (fp_spectrogram.py)
  * model: SpecCNN, a compact from-scratch 2-D CNN — the right-sized stand-in for
    the paper's ViT-L-16, which needs ImageNet pretraining + huge data.

Reads the raw-IQ captures recorded by terminal-ml-v2.py (Focus / Wideband + Record),
so nothing needs re-recording. The GUI loads the resulting model via the SAME module.

    python train_fingerprint_spec.py
    python train_fingerprint_spec.py --seg_len 8192 --epochs 40
"""

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from fp_spectrogram import (iq_to_spectrogram, SpecCNN,
                            N_FFT, STFT_HOP, SEG_LEN, SEG_HOP)

DATA_DIR   = "./fingerprint_data"
OUTPUT     = "./fingerprint_spec_model.pt"
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 1e-3
BASE_CH    = 16
UNKNOWN_THRESH = 0.8
SEED       = 42


def file_to_specs(path: Path, seg_len: int, seg_hop: int) -> np.ndarray:
    """One .iq file -> (k, 1, N_FFT, frames) spectrograms (None if too short)."""
    raw = np.fromfile(str(path), dtype=np.complex64)
    if len(raw) < seg_len:
        return None
    k = (len(raw) - seg_len) // seg_hop + 1
    return np.stack([iq_to_spectrogram(raw[i * seg_hop:i * seg_hop + seg_len])
                     for i in range(k)])


def load_split(data_dir: Path, seg_len: int, seg_hop: int, rng):
    """Build train/val spectrogram tensors. Class = folder; split by session."""
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found under {data_dir}.")

    Xtr, ytr, Xva, yva = [], [], [], []
    for lab, cls in enumerate(classes):
        files_by_sess = defaultdict(list)
        for f in (data_dir / cls).rglob("*.iq"):
            files_by_sess[f.parent.name].append(f)
        sessions = sorted(files_by_sess)
        if not sessions:
            print(f"  {cls:<10}: no .iq files, skipped")
            continue

        if len(sessions) >= 2:
            val_sessions, random_val = {sessions[-1]}, False
        else:
            val_sessions, random_val = set(), True
            print(f"  [warn] class '{cls}' has a single session ({sessions[0]}); "
                  f"using a RANDOM split — that val accuracy is optimistic.")

        cls_tr = cls_va = 0
        for sess, files in files_by_sess.items():
            specs = [file_to_specs(f, seg_len, seg_hop) for f in files]
            specs = [s for s in specs if s is not None]
            if not specs:
                continue
            specs = np.concatenate(specs)

            if random_val:
                perm = rng.permutation(len(specs))
                n_va = max(1, int(0.2 * len(specs)))
                va, tr = specs[perm[:n_va]], specs[perm[n_va:]]
            elif sess in val_sessions:
                va, tr = specs, specs[:0]
            else:
                va, tr = specs[:0], specs

            if len(tr):
                Xtr.append(tr); ytr.append(np.full(len(tr), lab, np.int64)); cls_tr += len(tr)
            if len(va):
                Xva.append(va); yva.append(np.full(len(va), lab, np.int64)); cls_va += len(va)

        held = "random" if random_val else f"session {sorted(val_sessions)[0]}"
        print(f"  {cls:<10}: {cls_tr:,} train / {cls_va:,} val  (val = {held})")

    if not Xtr or not Xva:
        raise RuntimeError("Not enough data to form both a train and a val split.")
    return (classes,
            np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xva), np.concatenate(yva))


def print_confusion(model, loader, classes, device):
    model.eval()
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                cm[int(t), int(p)] += 1
    print("\nConfusion matrix (val — rows = true, cols = pred):")
    print(" " * 11 + "".join(f"{c:>12}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>10} " + "".join(f"{cm[i, j]:>12,}" for j in range(n)))
    print("\nPer-class (val):")
    for i, c in enumerate(classes):
        tp, col, row = int(cm[i, i]), int(cm[:, i].sum()), int(cm[i, :].sum())
        prec = tp / col if col else 0.0
        rec  = tp / row if row else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {c:>10}: precision {prec:6.1%}   recall {rec:6.1%}   f1 {f1:6.1%}")


def train(args):
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice : {device}")
    print(f"STFT   : {N_FFT}-pt, hop {STFT_HOP}   |   segment {args.seg_len}/{args.seg_hop}")
    print("\nLoading captures…")
    classes, Xtr, ytr, Xva, yva = load_split(
        Path(args.data_dir), args.seg_len, args.seg_hop, rng)
    print(f"\nClasses  : {classes}")
    print(f"Train/Val: {len(Xtr):,} / {len(Xva):,} spectrograms  shape {Xtr.shape[1:]}")

    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
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
            xb, yb = xb.to(device), yb.to(device)
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
                xb, yb = xb.to(device), yb.to(device)
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

    model.load_state_dict(best_state)
    print_confusion(model, va_loader, classes, device)

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
        "val_acc"       : best_acc,
        "model"         : "SpecCNN",
        "representation": "stft256_logmag_1ch",
    }
    with open(args.out.replace(".pt", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[ok] Model -> {args.out}")
    print(f"     Best val accuracy : {best_acc:.2%}")
    print(f"     Total time        : {(time.time() - t0)/60:.1f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Spectrogram RF-fingerprint trainer")
    p.add_argument("--data_dir",       default=DATA_DIR)
    p.add_argument("--out",            default=OUTPUT)
    p.add_argument("--seg_len",        type=int,   default=SEG_LEN)
    p.add_argument("--seg_hop",        type=int,   default=SEG_HOP)
    p.add_argument("--epochs",         type=int,   default=EPOCHS)
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--base_ch",        type=int,   default=BASE_CH)
    p.add_argument("--unknown_thresh", type=float, default=UNKNOWN_THRESH)
    p.add_argument("--seed",           type=int,   default=SEED)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
