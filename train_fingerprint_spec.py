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
    python train_fingerprint_spec.py --quick                      # fast sanity check
    python train_fingerprint_spec.py --max_segs_per_class 4000    # cap RAM / balance noise
    python train_fingerprint_spec.py --seg_len 8192 --epochs 40

Memory/time is bounded by the --max_* caps and --store_dtype (float16 by default);
--quick is the "is this going the right way?" preset.
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

# ── resource / time budget — edit here, or override on the CLI ────────────────
QUICK               = True       # True = fast sanity run (few epochs + small caps)
MAX_FILES_PER_CLASS = 0          # 0 = all; cap .iq files per class per session
MAX_SEGS_PER_FILE   = 0          # 0 = all; cap spectrogram segments per file
MAX_SEGS_PER_CLASS  = 0          # 0 = all; cap segments per class per split (reins in noise)
STORE_DTYPE         = "float16"  # in-RAM cache: "float16" (half the RAM) or "float32"
FORCE_CPU           = False      # True = force CPU even if CUDA is available

# ── energy gate: drop silent (noise-level) segments from non-noise classes ────
GATE_DEVICE_SEGS = True          # True = keep only device segments above the noise floor
GATE_MARGIN_DB   = 3.0           # a device segment must beat the noise 95th-pct by this
NOISE_CLASS      = "noise"       # class treated as the noise floor / kept un-gated


def _seg_powers_db(raw, seg_len, seg_hop):
    """Per-segment mean power (dB) across one IQ buffer."""
    k = (len(raw) - seg_len) // seg_hop + 1
    if k <= 0:
        return np.empty(0, np.float64)
    p = np.array([np.mean(np.abs(raw[i * seg_hop:i * seg_hop + seg_len]) ** 2)
                  for i in range(k)], dtype=np.float64)
    return 10.0 * np.log10(p + 1e-12)


def file_to_specs(path: Path, seg_len: int, seg_hop: int, max_segs: int = 0,
                  store_dtype=np.float32, min_power_db=None) -> np.ndarray:
    """One .iq file -> (k, 1, N_FFT, frames) spectrograms (None if nothing kept).

    max_segs > 0 evenly subsamples to at most that many segments per file (less RAM
    and time, keeps the temporal spread); store_dtype=float16 halves the cache;
    min_power_db drops segments quieter than that — the energy gate that removes the
    silent gaps between bursts in a parked device capture (those are really noise)."""
    raw = np.fromfile(str(path), dtype=np.complex64)
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
    return np.stack([iq_to_spectrogram(raw[i * seg_hop:i * seg_hop + seg_len])
                     for i in sel]).astype(store_dtype, copy=False)


def load_split(data_dir: Path, seg_len: int, seg_hop: int, rng, *,
               max_files: int = 0, max_segs_file: int = 0, max_segs_class: int = 0,
               store_dtype=np.float32, gate: bool = False, gate_margin_db: float = 3.0,
               noise_class: str = "noise"):
    """Build train/val spectrogram tensors. Class = folder; split by session.

    Works for any class folder, `noise` included. Optional caps bound memory/time:
      * max_files      — .iq files kept per class per session
      * max_segs_file  — segments per file
      * max_segs_class — segments per class per split (also reins in a huge `noise`
                         class so it can't dominate RAM or the loss)
    store_dtype=float16 halves the cache footprint.

    gate=True drops, from every NON-noise class, segments quieter than the noise
    floor (`noise` 95th-pct power + gate_margin_db). A parked device capture is mostly
    noise between bursts; without this those silent segments are mislabelled as the
    device and the classifier collapses."""
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found under {data_dir}.")

    gate_thr = None
    if gate and noise_class in classes:
        npw = []
        for f in sorted((data_dir / noise_class).rglob("*.iq"))[:80]:
            raw = np.fromfile(str(f), dtype=np.complex64)
            if len(raw) >= seg_len:
                npw.append(_seg_powers_db(raw, seg_len, seg_hop))
        if npw:
            ceiling = float(np.percentile(np.concatenate(npw), 95))
            gate_thr = ceiling + gate_margin_db
            print(f"  [gate] noise floor 95th-pct {ceiling:.1f} dB -> keep non-noise "
                  f"segments >= {gate_thr:.1f} dB  (margin +{gate_margin_db:.0f} dB)")
        else:
            print("  [gate] noise class has no usable files — gate disabled")
    elif gate:
        print(f"  [gate] no '{noise_class}' class found — energy gate disabled")

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

        min_pdb = None if (gate_thr is None or cls == noise_class) else gate_thr
        tr_parts, va_parts = [], []
        for sess, files in files_by_sess.items():
            if max_files and len(files) > max_files:        # random subset of files
                files = [files[i] for i in rng.permutation(len(files))[:max_files]]
            specs = [file_to_specs(f, seg_len, seg_hop, max_segs_file, store_dtype, min_pdb)
                     for f in files]
            specs = [s for s in specs if s is not None]
            if not specs:
                continue
            specs = np.concatenate(specs)

            if random_val:
                perm = rng.permutation(len(specs))
                n_va = max(1, int(0.2 * len(specs)))
                va_parts.append(specs[perm[:n_va]]); tr_parts.append(specs[perm[n_va:]])
            elif sess in val_sessions:
                va_parts.append(specs)
            else:
                tr_parts.append(specs)

        tr = np.concatenate(tr_parts) if tr_parts else None
        va = np.concatenate(va_parts) if va_parts else None
        if max_segs_class and tr is not None and len(tr) > max_segs_class:
            tr = tr[rng.permutation(len(tr))[:max_segs_class]]
        if max_segs_class and va is not None and len(va) > max_segs_class:
            va = va[rng.permutation(len(va))[:max_segs_class]]

        n_tr = 0 if tr is None else len(tr)
        n_va = 0 if va is None else len(va)
        if n_tr:
            Xtr.append(tr); ytr.append(np.full(n_tr, lab, np.int64))
        if n_va:
            Xva.append(va); yva.append(np.full(n_va, lab, np.int64))

        held = "random" if random_val else f"session {sorted(val_sessions)[0]}"
        print(f"  {cls:<10}: {n_tr:,} train / {n_va:,} val  (val = {held})")

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
            pred = model(xb.to(device).float()).argmax(1).cpu().numpy()
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
    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available())
                          else "cuda")
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32

    if args.quick:                       # fast "is this going the right way?" run
        args.epochs = min(args.epochs, 5)
        if not args.max_files_per_class: args.max_files_per_class = 15
        if not args.max_segs_per_file:   args.max_segs_per_file   = 40
        print("\n*** QUICK MODE — small caps + few epochs. Sanity check only; "
              "the val number here is NOT final. ***")

    print(f"\nDevice : {device}   |   cache dtype {store_dtype.__name__}")
    print(f"STFT   : {N_FFT}-pt, hop {STFT_HOP}   |   segment {args.seg_len}/{args.seg_hop}")
    caps = []
    if args.max_files_per_class: caps.append(f"<={args.max_files_per_class} files/class/session")
    if args.max_segs_per_file:   caps.append(f"<={args.max_segs_per_file} segs/file")
    if args.max_segs_per_class:  caps.append(f"<={args.max_segs_per_class} segs/class/split")
    print(f"Caps   : {', '.join(caps) if caps else 'none (full dataset)'}")
    print("\nLoading captures…")
    classes, Xtr, ytr, Xva, yva = load_split(
        Path(args.data_dir), args.seg_len, args.seg_hop, rng,
        max_files=args.max_files_per_class, max_segs_file=args.max_segs_per_file,
        max_segs_class=args.max_segs_per_class, store_dtype=store_dtype,
        gate=GATE_DEVICE_SEGS, gate_margin_db=args.gate_margin_db,
        noise_class=NOISE_CLASS)
    mem = (Xtr.nbytes + Xva.nbytes) / 1e6
    print(f"\nClasses  : {classes}")
    print(f"Train/Val: {len(Xtr):,} / {len(Xva):,} spectrograms  shape {Xtr.shape[1:]}"
          f"   (~{mem:.0f} MB cached)")

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
            xb, yb = xb.to(device).float(), yb.to(device)
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
    # ── resource / time budget (defaults come from the constants above) ────────
    p.add_argument("--max_files_per_class", type=int, default=MAX_FILES_PER_CLASS,
                   help="cap .iq files loaded per class per session (0 = all)")
    p.add_argument("--max_segs_per_file",   type=int, default=MAX_SEGS_PER_FILE,
                   help="cap spectrogram segments per file (0 = all)")
    p.add_argument("--max_segs_per_class",  type=int, default=MAX_SEGS_PER_CLASS,
                   help="cap segments per class per split (0 = all; reins in noise)")
    p.add_argument("--store_dtype", choices=["float16", "float32"], default=STORE_DTYPE,
                   help="in-RAM spectrogram cache dtype (float16 halves memory)")
    p.add_argument("--gate_margin_db", type=float, default=GATE_MARGIN_DB,
                   help="dB a device segment must beat the noise floor by to be kept "
                        "(energy gate; toggle with GATE_DEVICE_SEGS in code)")
    p.add_argument("--quick", action="store_true", default=QUICK,
                   help="fast sanity run: few epochs + small caps")
    p.add_argument("--cpu", action="store_true", default=FORCE_CPU,
                   help="force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
