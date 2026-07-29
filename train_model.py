"""
train_model.py — spectrogram fingerprint trainer (RFUAV-style).

Folder-driven: every top-level folder under --data_dir is one class, and session_*
subfolders drive a session-held-out split (train on early sessions, validate on a
session never seen in training; the only honest number). Representation and model:

  * representation: the paper's 256-point STFT spectrogram (fp_spectrogram.py)
  * model: SpecCNN, a compact from-scratch 2-D CNN — the right-sized stand-in for
    the paper's ViT-L-16, which needs ImageNet pretraining + huge data.

Reads the raw-IQ captures recorded by terminal_v2.py (Device / Noise recordings),
so nothing needs re-recording. The GUI loads the resulting model via the SAME module.

    python train_model.py                              # balanced (default)
    python train_model.py --preset fast                # ~1 min sanity check
    python train_model.py --preset best                # slow, most accurate
    python train_model.py --preset best --epochs 15    # explicit flags beat the preset
    python train_model.py --max_segs_per_class 4000    # cap RAM / balance noise

Memory/time is bounded by the --max_* caps and --store_dtype (float16 by default);
--quick is the "is this going the right way?" preset.
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
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from fp_spectrogram import (iq_to_spectrogram, SpecCNN,
                            N_FFT, STFT_HOP, SEG_LEN, SEG_HOP)

DATA_DIR   = "./fingerprint_data"
OUTPUT     = "./fast_demo_model.pt"
BATCH_SIZE = 512
EPOCHS     = 8
LR         = 1e-3
BASE_CH    = 16
UNKNOWN_THRESH = 0.8
SEED       = 42

# ── resource / time budget — edit here, or override on the CLI ────────────────
PRESET              = "fast"     # "fast" | "balanced" | "best" — the tier used when
                                 # run without --preset (e.g. straight from the IDE)
QUICK               = False      # True = fast sanity run (alias for --preset fast)
MAX_FILES_PER_CLASS = 5000       # 0 = all; cap .iq files per class per session
MAX_SEGS_PER_FILE   = 150        # 0 = all; cap spectrogram segments per file
MAX_SEGS_PER_CLASS  = 20000      # 0 = all; cap segments per class per split (reins in noise / RAM)
STORE_DTYPE         = "float16"  # in-RAM cache: "float16" (half the RAM) or "float32"
FORCE_CPU           = False      # True = force CPU even if CUDA is available

# ── speed/quality presets (--preset). Each sets the knobs below; any flag given
# explicitly on the CLI still wins over the preset. Times are CPU ballparks and
# scale with how much you've recorded. ─────────────────────────────────────────
PRESETS = {
    # sanity check (~1-2 min): tiny data caps, few epochs. Val number NOT final.
    "fast":     dict(epochs=5,  max_files_per_class=15, max_segs_per_file=40,
                     max_segs_per_class=8000,  base_ch=16),
    # the everyday run (~minutes-tens of minutes): all files, capped per class.
    "balanced": dict(epochs=EPOCHS, max_files_per_class=MAX_FILES_PER_CLASS,
                     max_segs_per_file=MAX_SEGS_PER_FILE,
                     max_segs_per_class=MAX_SEGS_PER_CLASS, base_ch=BASE_CH),
    # squeeze the most out of the data (hours on CPU): more data, bigger net,
    # long cosine schedule — best-epoch checkpointing means extra epochs only
    # cost time, never accuracy.
    "best":     dict(epochs=30, max_files_per_class=0, max_segs_per_file=0,
                     max_segs_per_class=40000, base_ch=24),
}

# ── energy gate: drop silent (noise-level) segments from non-noise classes ────
GATE_DEVICE_SEGS = True          # True = keep only device segments above the noise floor
GATE_MARGIN_DB   = 3.0           # a device segment must beat the noise 95th-pct by this
NOISE_CLASS      = "noise"       # class treated as the noise floor / kept un-gated

# ── weak-signal SNR augmentation: captures are recorded strong (close/high SNR),
# so the model never sees the faint version of the same device and misses it live.
# Fix: re-embed a fraction of train device segments into REAL recorded noise at a
# random SNR, teaching the fingerprint across the whole strong->faint range. ─────
SNR_AUG_P      = 0.5             # fraction of train device segments re-embedded (0 = off)
SNR_AUG_DB     = (0.0, 20.0)     # target SNR range in dB above the added noise's power
NOISE_POOL_MAX = 2000            # noise segments cached for mixing (~64 MB at seg 4096)

# ── frequency-shift augmentation: recordings park the LO on the drone's channel,
# so its energy always sits at the SAME spectrogram bins — the CNN learns position
# instead of fingerprint. Live locks center differently and drone links channel-hop,
# so train device segments get randomly retuned. ───────────────────────────────
FREQ_SHIFT_FRAC = 0.10           # random retune of ± this fraction of fs (0 = off)

# ── weak-signal val metric: the val split only holds the strong segments that were
# recorded, so plain ValAcc can't show whether faint drones are still recognised.
# Val device segments are ALSO re-embedded in noise at low SNR and scored separately.
WEAK_VAL_DB = (0.0, 10.0)        # SNR range for the weak-val copies

# ── SpecAugment-lite: 2.4 GHz is shared with WiFi/BT, so live spectrograms carry
# foreign bursts over the drone. Masking a random freq band + time stripe per train
# batch teaches classification from partial evidence. ──────────────────────────
SPEC_MASK_P    = 0.5             # probability a train batch gets masked (0 = off)
SPEC_MASK_FREQ = 32              # max masked frequency bins (of N_FFT=256)
SPEC_MASK_TIME = 12              # max masked time frames (of ~61)


def _natkey(s):
    """Natural sort key so session_10 sorts after session_9, not after session_1."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _seg_powers_db(raw, seg_len, seg_hop):
    """Per-segment mean power (dB) across one IQ buffer."""
    k = (len(raw) - seg_len) // seg_hop + 1
    if k <= 0:
        return np.empty(0, np.float64)
    p = np.array([np.mean(np.abs(raw[i * seg_hop:i * seg_hop + seg_len]) ** 2)
                  for i in range(k)], dtype=np.float64)
    return 10.0 * np.log10(p + 1e-12)


def _mix_noise(seg, noise_pool, rng, snr_lo, snr_hi):
    """Re-embed a device segment in real recorded noise at a random SNR (dB).

    Scales the (strong) capture so its power sits `snr` dB above a random noise
    segment's, then adds that noise — a physically honest weak-signal simulation
    (the capture's own residual floor scales down with it, the added noise becomes
    the new floor)."""
    noise = noise_pool[rng.randint(len(noise_pool))]
    ps = float(np.mean(np.abs(seg) ** 2))
    pn = float(np.mean(np.abs(noise) ** 2))
    if ps <= 0.0 or pn <= 0.0:
        return seg
    g = np.sqrt(pn / ps * 10.0 ** (rng.uniform(snr_lo, snr_hi) / 10.0))
    return (seg * np.complex64(g) + noise).astype(np.complex64, copy=False)


def _freq_shift(seg, rng, max_frac):
    """Retune a segment by a random frequency offset, ± max_frac of the sample rate.

    An exact IQ-domain shift (multiply by a complex ramp). Teaches the model the
    fingerprint is the same wherever the signal sits in the band — otherwise it
    keys on 'energy at these exact bins', which breaks the moment a live lock
    centers differently than the recordings or the drone changes channel."""
    nu = rng.uniform(-max_frac, max_frac)               # cycles/sample
    ramp = np.exp((2j * np.pi * nu) * np.arange(len(seg)))
    return (seg * ramp).astype(np.complex64, copy=False)


def file_to_specs(path: Path, seg_len: int, seg_hop: int, max_segs: int = 0,
                  store_dtype=np.float32, min_power_db=None,
                  noise_pool=None, aug_p=SNR_AUG_P, snr_db=SNR_AUG_DB,
                  f_shift=0.0, rng=None) -> np.ndarray:
    """One .iq file -> (k, 1, N_FFT, frames) spectrograms (None if nothing kept).

    max_segs > 0 evenly subsamples to at most that many segments per file (less RAM
    and time, keeps the temporal spread); store_dtype=float16 halves the cache;
    min_power_db drops segments quieter than that — the energy gate that removes the
    silent gaps between bursts in a parked device capture (those are really noise).

    noise_pool + rng turn on SNR augmentation: each kept segment is, with
    probability aug_p, re-embedded in real noise at a random snr_db level.
    f_shift > 0 first retunes every kept segment by a random ±f_shift × fs
    (shift THEN mix, so the added noise floor stays where a real one would).
    The gate runs on ORIGINAL powers first, so only genuine device bursts get
    attenuated — never near-silence relabelled as device."""
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
    segs = [raw[i * seg_hop:i * seg_hop + seg_len] for i in sel]
    if rng is not None and f_shift > 0:
        segs = [_freq_shift(s, rng, f_shift) for s in segs]
    if rng is not None and noise_pool:
        segs = [_mix_noise(s, noise_pool, rng, *snr_db) if rng.rand() < aug_p else s
                for s in segs]
    return np.stack([iq_to_spectrogram(s)
                     for s in segs]).astype(store_dtype, copy=False)


def load_split(data_dir: Path, seg_len: int, seg_hop: int, rng, *,
               max_files: int = 0, max_segs_file: int = 0, max_segs_class: int = 0,
               store_dtype=np.float32, gate: bool = False, gate_margin_db: float = 3.0,
               noise_class: str = "noise", snr_aug_p: float = 0.0,
               f_shift: float = 0.0):
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
    device and the classifier collapses.

    snr_aug_p > 0 re-embeds that fraction of TRAIN device segments into real noise
    at a random SNR (SNR_AUG_DB), and f_shift > 0 randomly retunes them, so a device
    recorded strong on one channel is still recognised weak on another. Val segments
    and the noise class itself are never augmented — but device val segments get a
    SEPARATE weak copy (re-embedded at WEAK_VAL_DB) returned as (Xwk, ywk) so the
    faint-signal case is measured, not assumed."""
    classes = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"No class folders found under {data_dir}.")

    # One pass over the noise class's TRAIN sessions feeds both the energy-gate
    # threshold and the SNR-augmentation pool (val-session noise stays out of both).
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
            raw = np.fromfile(str(f), dtype=np.complex64)
            if len(raw) < seg_len:
                continue
            if gate:
                npw.append(_seg_powers_db(raw, seg_len, seg_hop))
            if snr_aug_p > 0:
                k = (len(raw) - seg_len) // seg_hop + 1
                for i in rng.permutation(k)[:per_file]:
                    noise_pool.append(raw[i * seg_hop:i * seg_hop + seg_len].copy())
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

    Xtr, ytr, Xva, yva, Xwk, ywk = [], [], [], [], [], []
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
        # Augment only non-noise TRAIN sessions; a random split can't separate the
        # two before conversion, so single-session classes stay un-augmented.
        aug_dev = cls != noise_class and not random_val
        tr_parts, va_parts, wk_parts = [], [], []
        for sess, files in files_by_sess.items():
            if max_files and len(files) > max_files:        # random subset of files
                files = [files[i] for i in rng.permutation(len(files))[:max_files]]
            is_train = aug_dev and sess not in val_sessions
            pool  = noise_pool if (is_train and noise_pool) else None
            shift = f_shift if is_train else 0.0
            specs = [file_to_specs(f, seg_len, seg_hop, max_segs_file, store_dtype,
                                   min_pdb, noise_pool=pool, aug_p=snr_aug_p,
                                   f_shift=shift, rng=rng)
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
                # Weak-val twin: same val files re-embedded at low SNR. The noise
                # comes from the TRAIN pool — a leak in the hard direction only
                # (it can't inflate device recognition), so the number stays honest.
                if aug_dev and noise_pool:
                    wk = [file_to_specs(f, seg_len, seg_hop, max_segs_file,
                                        store_dtype, min_pdb, noise_pool=noise_pool,
                                        aug_p=1.0, snr_db=WEAK_VAL_DB, rng=rng)
                          for f in files]
                    wk = [s for s in wk if s is not None]
                    if wk:
                        wk_parts.append(np.concatenate(wk))
            else:
                tr_parts.append(specs)

        tr = np.concatenate(tr_parts) if tr_parts else None
        va = np.concatenate(va_parts) if va_parts else None
        wk = np.concatenate(wk_parts) if wk_parts else None
        if max_segs_class and tr is not None and len(tr) > max_segs_class:
            tr = tr[rng.permutation(len(tr))[:max_segs_class]]
        if max_segs_class and va is not None and len(va) > max_segs_class:
            va = va[rng.permutation(len(va))[:max_segs_class]]
        if max_segs_class and wk is not None and len(wk) > max_segs_class:
            wk = wk[rng.permutation(len(wk))[:max_segs_class]]

        n_tr = 0 if tr is None else len(tr)
        n_va = 0 if va is None else len(va)
        n_wk = 0 if wk is None else len(wk)
        if n_tr:
            Xtr.append(tr); ytr.append(np.full(n_tr, lab, np.int64))
        if n_va:
            Xva.append(va); yva.append(np.full(n_va, lab, np.int64))
        if n_wk:
            Xwk.append(wk); ywk.append(np.full(n_wk, lab, np.int64))

        held = "random" if random_val else f"session {sorted(val_sessions)[0]}"
        weak = f" + {n_wk:,} weak" if n_wk else ""
        print(f"  {cls:<10}: {n_tr:,} train / {n_va:,} val{weak}  (val = {held})")

    if not Xtr or not Xva:
        raise RuntimeError("Not enough data to form both a train and a val split.")
    return (classes,
            np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xva), np.concatenate(yva),
            np.concatenate(Xwk) if Xwk else None,
            np.concatenate(ywk) if ywk else None)


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
    classes, Xtr, ytr, Xva, yva, Xwk, ywk = load_split(
        Path(args.data_dir), args.seg_len, args.seg_hop, rng,
        max_files=args.max_files_per_class, max_segs_file=args.max_segs_per_file,
        max_segs_class=args.max_segs_per_class, store_dtype=store_dtype,
        gate=GATE_DEVICE_SEGS, gate_margin_db=args.gate_margin_db,
        noise_class=NOISE_CLASS, snr_aug_p=args.snr_aug_p,
        f_shift=args.freq_shift_frac)
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
            if SPEC_MASK_P and torch.rand(()).item() < SPEC_MASK_P:
                # SpecAugment-lite: zero a random freq band + time stripe (0 = the
                # per-image mean, since specs are standardised) so the model learns
                # from partial evidence — WiFi/BT bursts will sit over live drones.
                # Multiply (not in-place) — xb may share storage with the cache.
                # ponytail: one mask per batch; go per-sample if gains stall.
                m = torch.ones(1, 1, xb.shape[2], xb.shape[3], device=xb.device)
                fw = int(torch.randint(1, SPEC_MASK_FREQ + 1, ()).item())
                fs = int(torch.randint(0, xb.shape[2] - fw + 1, ()).item())
                tw = int(torch.randint(1, SPEC_MASK_TIME + 1, ()).item())
                ts = int(torch.randint(0, xb.shape[3] - tw + 1, ()).item())
                m[..., fs:fs + fw, :] = 0.0
                m[..., ts:ts + tw] = 0.0
                xb = xb * m
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

    if best_state is not None:              # val acc can stay 0.0 on a broken run
        model.load_state_dict(best_state)
    print_confusion(model, va_loader, classes, device)

    # Weak-signal val: the same held-out device segments, re-embedded at low SNR.
    # This is the number that moves when faint-drone recognition regresses.
    weak_acc = None
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
        per = "   ".join(f"{classes[i]} {hit[i]/tot[i]:.1%}"
                         for i in range(len(classes)) if tot[i])
        print(f"\nWeak-signal val ({WEAK_VAL_DB[0]:.0f}-{WEAK_VAL_DB[1]:.0f} dB SNR): "
              f"{weak_acc:.2%}   ({per})")

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
        "preset"        : args.preset,
        "snr_aug_p"     : args.snr_aug_p,
        "snr_aug_db"    : list(SNR_AUG_DB),
        "freq_shift_frac": args.freq_shift_frac,
        "weak_val_acc"  : weak_acc,
    }
    # with_suffix (not str.replace) so this always matches how FingerprintModel
    # derives the meta path (splitext), whatever --out is called.
    with open(Path(args.out).with_suffix(".meta.json"), "w") as f:
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
    p.add_argument("--batch_size",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",             type=float, default=LR)
    p.add_argument("--unknown_thresh", type=float, default=UNKNOWN_THRESH)
    p.add_argument("--seed",           type=int,   default=SEED)
    # ── speed/quality preset. The flags below default to None so an explicitly
    # given value can be told apart from "let the preset decide". ───────────────
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="fast = ~1-2 min sanity check | balanced = everyday run "
                        "(default) | best = slowest, most accurate")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--max_files_per_class", type=int, default=None,
                   help="cap .iq files loaded per class per session (0 = all)")
    p.add_argument("--max_segs_per_file",   type=int, default=None,
                   help="cap spectrogram segments per file (0 = all)")
    p.add_argument("--max_segs_per_class",  type=int, default=None,
                   help="cap segments per class per split (0 = all; reins in noise)")
    p.add_argument("--store_dtype", choices=["float16", "float32"], default=STORE_DTYPE,
                   help="in-RAM spectrogram cache dtype (float16 halves memory)")
    p.add_argument("--gate_margin_db", type=float, default=GATE_MARGIN_DB,
                   help="dB a device segment must beat the noise floor by to be kept "
                        "(energy gate; toggle with GATE_DEVICE_SEGS in code)")
    p.add_argument("--snr_aug_p", type=float, default=SNR_AUG_P,
                   help="fraction of train device segments re-embedded in real noise "
                        "at a random SNR so weak signals are recognised (0 = off; "
                        "range set by SNR_AUG_DB in code)")
    p.add_argument("--freq_shift_frac", type=float, default=FREQ_SHIFT_FRAC,
                   help="randomly retune train device segments by ± this fraction of "
                        "the sample rate so off-center / channel-hopped signals are "
                        "recognised (0 = off)")
    p.add_argument("--quick", action="store_true", default=QUICK,
                   help="alias for --preset fast")
    p.add_argument("--cpu", action="store_true", default=FORCE_CPU,
                   help="force CPU even if CUDA is available")
    args = p.parse_args()
    # resolve: explicit CLI flag > preset > the PRESET constant up top (IDE runs)
    if args.preset is None:
        args.preset = "fast" if args.quick else PRESET
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


if __name__ == "__main__":
    train(parse_args())
