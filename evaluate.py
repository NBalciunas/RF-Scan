"""Measure a trained model on captures, at the level that the user sees.

The trainer reports a per-segment accuracy over the segments that pass the energy
gate. That number is correct and it is not what the program shows: the GUI reads a
whole capture and gives one badge. The defect #29 lived in that difference for as
long as nobody measured the second one. This program measures the second one.

    python evaluate.py trained_model.pt
    python evaluate.py trained_model.pt --data_dir ./heldout_data
    python evaluate.py trained_model.pt --session session_3 --sweep

Three tables come out.

  * Each class and session: the captures that got the correct name, the captures
    that read `clear`, and the captures that got another name.
  * The confusion of the captures, where `clear` is a column of its own. A `clear`
    on a drone capture is a miss and it is not a wrong name, thus the two must not
    go in one cell.
  * The same captures at the level of the segments, to compare against the trainer.

A capture of a hopping link often holds no hop, thus `clear` is the correct answer
for it. The column "holds signal" gives the captures that have at least one segment
above the noise floor, and the accuracy beside it counts those only.

--sweep varies MIN_SEG_SHARE, which decides how often a bursty link is seen. Run it
on the validation session and never on the held-out captures.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from fp_spectrogram import FingerprintModel, iq_segments_to_specs, remove_dc
from train_model import _seg_powers_db, GATE_MARGIN_DB, NOISE_CLASS
# badge_for is the rule the GUI uses. This program must never hold a second copy
# of it, or the report and the program drift apart. See the defect #29.
from terminal import badge_for

SWEEP = (0.30, 0.20, 0.15, 0.10, 0.05, 0.02)


def gate_threshold(data_dir: Path, seg_len, seg_hop, max_files=20):
    """The energy gate of the trainer: the 95th percentile of noise plus a margin.

    The trainer takes it from the train sessions of the noise class. The same rule
    runs here, thus a capture is called empty by the same limit that decided which
    segments trained the model.
    """
    root = data_dir / NOISE_CLASS
    if not root.is_dir():
        return None
    sess = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not sess:
        return None
    train_sess = sess[:-1] if len(sess) >= 2 else sess
    pw = []
    for s in train_sess:
        for f in sorted(s.glob("*.iq"))[:max_files]:
            raw = remove_dc(np.fromfile(str(f), dtype=np.complex64))
            if len(raw) >= seg_len:
                pw.append(_seg_powers_db(raw, seg_len, seg_hop))
    if not pw:
        return None
    return float(np.percentile(np.concatenate(pw), 95)) + GATE_MARGIN_DB


def capture_rows(fp, data_dir: Path, session=None, max_files=0, gate=None):
    """Give one row for each capture: (class, session, badge kind, name, ...)."""
    rows = []
    for cls_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        for sess in sorted(d for d in cls_dir.iterdir() if d.is_dir()):
            if session and sess.name != session:
                continue
            files = sorted(sess.glob("*.iq"))
            if max_files:
                files = files[:max_files]
            for f in files:
                iq = np.fromfile(str(f), dtype=np.complex64)
                specs = iq_segments_to_specs(iq, fp.seg_len, fp.seg_hop, fp.n_fft,
                                             fp.stft_hop, max_segs=fp.infer_max_segs)
                with torch.no_grad():
                    x = torch.from_numpy(specs.astype(np.float32))
                    seg_probs = torch.softmax(fp.net(x), dim=1).numpy()
                signal = None
                if gate is not None:
                    raw = remove_dc(iq)
                    if len(raw) >= fp.seg_len:
                        signal = bool((_seg_powers_db(raw, fp.seg_len, fp.seg_hop)
                                       >= gate).any())
                rows.append({"cls": cls_dir.name, "sess": sess.name,
                             "seg_probs": seg_probs, "signal": signal})
    return rows


def badge_of(fp, seg_probs, min_share):
    """The badge that the GUI would show for these segment probabilities."""
    from fp_spectrogram import segment_vote
    probs = seg_probs.mean(0)
    idx = int(probs.argmax())
    res = {"label": fp.classes[idx], "confidence": float(probs[idx]),
           "probs": {c: float(p) for c, p in zip(fp.classes, probs)},
           "detections": segment_vote(seg_probs, fp.classes, fp.vote_thresh,
                                      min_share)}
    text, kind = badge_for(res)
    if kind == "none":
        return "clear"
    for c in fp.classes:
        if text.startswith(c):
            return c
    return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description="Measure a model on whole captures")
    p.add_argument("model")
    p.add_argument("--data_dir", default="./fingerprint_data")
    p.add_argument("--session", default=None,
                   help="one session name only, for example session_3")
    p.add_argument("--max_files", type=int, default=0,
                   help="captures for each session (0 = all)")
    p.add_argument("--min_share", type=float, default=None,
                   help="override MIN_SEG_SHARE for the report")
    p.add_argument("--sweep", action="store_true",
                   help="vary MIN_SEG_SHARE. Use the validation session only.")
    args = p.parse_args(argv)

    from fp_spectrogram import MIN_SEG_SHARE
    share = MIN_SEG_SHARE if args.min_share is None else args.min_share
    data_dir = Path(args.data_dir)
    fp = FingerprintModel(args.model)
    print(f"\nModel   : {args.model}")
    print(f"Classes : {', '.join(fp.classes)}")
    print(f"Data    : {data_dir.resolve()}"
          + (f"   session {args.session}" if args.session else ""))
    print(f"Rule    : vote_thresh {fp.vote_thresh}, min_share {share}, "
          f"unknown_thresh {fp.unknown_thresh}")

    gate = gate_threshold(data_dir, fp.seg_len, fp.seg_hop)
    print(f"Gate    : {'none' if gate is None else f'{gate:.2f} dB'}"
          "   (a capture with no segment above it holds no signal)\n")

    rows = capture_rows(fp, data_dir, args.session, args.max_files, gate)
    if not rows:
        print("[error] no captures found.")
        return 1

    devices = [c for c in fp.classes if c != NOISE_CLASS]

    # 1. each class and session
    hdr = (f"{'class':<20}{'session':<11}{'n':>5}{'named right':>12}{'clear':>7}"
           f"{'named wrong':>12}{'correct':>9}   {'holds signal':>13}"
           f"{'right of those':>16}")
    print(hdr)
    print("-" * len(hdr))
    print("The three count columns do not overlap. For `noise` the correct badge is "
          "`clear`,\nthus its 'named right' is 0 by construction.\n")
    for cls in fp.classes:
        for sess in sorted({r["sess"] for r in rows if r["cls"] == cls}):
            g = [r for r in rows if r["cls"] == cls and r["sess"] == sess]
            named = [badge_of(fp, r["seg_probs"], share) for r in g]
            right = sum(n == cls for n in named)
            clear = sum(n == "clear" for n in named)
            wrong = len(g) - right - clear
            ok = clear if cls == NOISE_CLASS else right
            sig = [i for i, r in enumerate(g) if r["signal"]]
            sig_txt = f"{len(sig)}" if gate is not None else "-"
            if gate is not None and sig and cls != NOISE_CLASS:
                acc_txt = f"{sum(named[i] == cls for i in sig) / len(sig):.1%}"
            else:
                acc_txt = "-"
            print(f"{cls:<20}{sess:<11}{len(g):>5}{right:>12}{clear:>7}{wrong:>12}"
                  f"{ok / len(g):>9.1%}   {sig_txt:>13}{acc_txt:>16}")

    # 2. the confusion of the captures
    cols = devices + ["unknown", "clear"]
    print(f"\nCaptures, rows are true and columns are the badge:\n")
    w = max(len(c) for c in fp.classes) + 2
    print(" " * w + "".join(f"{c[:12]:>14}" for c in cols))
    for cls in fp.classes:
        g = [r for r in rows if r["cls"] == cls]
        named = [badge_of(fp, r["seg_probs"], share) for r in g]
        line = f"{cls:<{w}}"
        for c in cols:
            line += f"{sum(n == c for n in named):>14}"
        print(line)

    # 3. the same captures at the level of the segments
    print("\nSegments, every one of them. The table of the trainer counts the "
          "segments that\npass the energy gate only, thus the two are different "
          "populations and the numbers\nbelow are the harder ones. See the "
          "defect #29.\n")
    print(" " * w + "".join(f"{c[:12]:>14}" for c in fp.classes))
    for cls in fp.classes:
        g = [r for r in rows if r["cls"] == cls]
        if not g:
            continue
        win = np.concatenate([r["seg_probs"].argmax(1) for r in g])
        line = f"{cls:<{w}}"
        for i in range(len(fp.classes)):
            line += f"{int((win == i).sum()):>14}"
        print(line)

    if args.sweep:
        print(f"\nMIN_SEG_SHARE sweep. Read the false alarm column first.\n")
        head = f"{'min_share':>10}" + "".join(f"{c[:14]:>16}" for c in devices)
        head += f"{'noise named':>13}"
        print(head)
        print("-" * len(head))
        for s in SWEEP:
            line = f"{s:>10.2f}"
            for cls in devices:
                g = [r for r in rows if r["cls"] == cls]
                if not g:
                    line += f"{'-':>16}"
                    continue
                hit = sum(badge_of(fp, r["seg_probs"], s) == cls for r in g)
                line += f"{hit / len(g):>15.1%} "
            g = [r for r in rows if r["cls"] == NOISE_CLASS]
            fa = sum(badge_of(fp, r["seg_probs"], s) != "clear" for r in g)
            line += f"{(fa / len(g) if g else 0):>12.1%} "
            print(line)
        print("\nA false alarm is a noise capture that the badge named. Pick the "
              "largest\ndetection rate whose false alarm rate you accept, on the "
              "validation session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
