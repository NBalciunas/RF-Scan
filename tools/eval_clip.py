"""Measure a trained model on a prepared clip, at a chosen signal-to-noise ratio.

`tools/evaluate.py` reads captures that came from the radio. This program reads a clip
that has not been on the air, thus it answers a question that no recording can answer
without a bench session: what does the model say about a signal that nobody recorded?

    python tools/eval_clip.py trained_model.pt transmitting/clips/dji_bw20_0-1s.iq \
        --expect DJI-MINI-3

Run it from the repository root. The paths are relative to the current directory.

A clip is clean and a capture is not. The program therefore puts each capture into
**recorded** noise at a stated SNR, with the same function that makes the weak-signal
copies of the trainer. A clip read with no noise is the `clean` row and it is the
condition that is furthest from the air. Read the whole column, not one number.

**Read the `floor` column before any other.** A clip carries the noise floor of the
source recording and not of this room, and the model has never met that floor. Measured
on 2026-08-17: a 50 ms window of `at9s_4-5s.iq` that holds no hop reads
`DJI-MINI-3` on 8 of 8, where a recorded capture of the room reads `clear` on 19 of 20.
The added noise covers the foreign floor only when it is louder than it, and
`_mix_noise` sets the level against the mean of the whole capture, thus a capture that
is mostly its own floor gets noise **below** that floor and nothing is fixed. The
`floor` column is how far the added noise sits above the quiet segments of the clip. A
row with a negative number measures the source recording and not the drone.

The result means nothing without a control. Give a clip of a class that the model was
trained on in the same command, and read the two rows together. A control of the right
class is not enough by itself: on 2026-08-17 a DJI control passed while the path named
an empty window `DJI-MINI-3` as well, thus the control agreed for the wrong reason. Add
a clip window that holds no transmitter whenever the answer matters.

The noise comes from the train sessions of the `noise` class, which is a leak. It makes
the task harder and it can not raise the result, thus it is the same trade that
WEAK_VAL_DB takes in the trainer.
"""

import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

# This file is in tools/ and it reads the model, the trainer and the badge rule from
# the root. See §2 of NOTES.md.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CAP_LEN = 500_000        # samples of one capture. 50 ms at 10 Msps, as the dataset.
NOISE_FILES = 40         # recorded captures that make the noise pool


def clip_captures(path, n, cap_len=CAP_LEN):
    """Give n captures of cap_len samples, at equal distances along the clip.

    A clip holds one second and a capture holds 50 ms, thus the program must choose
    where the captures come from. Equal distances cover the whole clip, and a hopping
    link needs that: one part of a second says nothing about the rest."""
    iq = np.fromfile(str(path), dtype=np.complex64)
    if len(iq) < cap_len:
        raise SystemExit(f"{path} holds {len(iq)} samples and one capture needs "
                         f"{cap_len}.")
    starts = np.unique(np.linspace(0, len(iq) - cap_len, n).round().astype(int))
    return np.stack([iq[s:s + cap_len] for s in starts])


def floor_margin_db(caps, pool, snr_db):
    """How far the added noise sits above the quiet segments of the clip, in dB.

    A clip holds the noise floor of the source recording. The model has never met that
    floor and it names it, thus the floor must be covered by a floor the model knows.
    `_mix_noise` scales the capture so its **mean** power sits snr_db above the noise,
    thus the added noise lands at (mean - snr_db). A capture that is mostly its own
    floor has mean equal to that floor, and the added noise then lands below it.

    Gives None for the clean condition, where nothing is added at all."""
    from train_model import _seg_powers_db
    from fp_spectrogram import remove_dc, SEG_LEN, SEG_HOP
    if snr_db is None:
        return None
    quiet, mean_p = [], []
    for cap in caps:
        p = _seg_powers_db(remove_dc(cap), SEG_LEN, SEG_HOP)
        if len(p):
            quiet.append(float(np.percentile(p, 5)))
            mean_p.append(10.0 * np.log10(float(np.mean(np.abs(cap) ** 2)) + 1e-30))
    if not quiet:
        return None
    pn = float(np.median([10.0 * np.log10(float(np.mean(np.abs(n) ** 2)) + 1e-30)
                          for n in pool]))
    # _mix_noise scales the capture and keeps the noise as it is, thus the floor that
    # arrives is the power of the noise pool itself, and the clip is moved instead.
    gain_db = pn + snr_db - float(np.median(mean_p))
    return pn - (float(np.median(quiet)) + gain_db)


def noise_pool(noise_dir, cap_len=CAP_LEN, max_files=NOISE_FILES):
    """Give recorded noise captures, cut to cap_len samples.

    The last session is the validation session of the trainer, thus it stays out. The
    pool is a floor to add and not an evaluation set, and the train sessions are the
    larger part."""
    root = Path(noise_dir)
    sess = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not sess:
        raise SystemExit(f"{root} holds no session folder.")
    train_sess = sess[:-1] if len(sess) >= 2 else sess
    out = []
    for s in train_sess:
        for f in sorted(s.glob("*.iq")):
            if len(out) >= max_files:
                return out
            iq = np.fromfile(str(f), dtype=np.complex64)
            if len(iq) >= cap_len:
                out.append(iq[:cap_len])
    if not out:
        raise SystemExit(f"{root} holds no capture of {cap_len} samples.")
    return out


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except Exception:
        return None


def _conditions(text):
    """Parse the SNR list. `clean` is no noise at all."""
    out = []
    for part in text.split(","):
        part = part.strip()
        out.append(None if part.lower() == "clean" else float(part))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Measure a model on a prepared clip, at a chosen SNR")
    p.add_argument("model")
    p.add_argument("clips", nargs="+", help="the .iq of the clips, complex64 at the "
                                            "rate of the model")
    p.add_argument("--expect", default=None,
                   help="the class that the badge should name. Without it the "
                        "program reports the names and counts nothing right.")
    p.add_argument("--noise_dir", default="./fingerprint_data/noise")
    p.add_argument("--snr", default="clean,30,20,10",
                   help="the conditions, in dB. `clean` adds no noise.")
    p.add_argument("--captures", type=int, default=100)
    p.add_argument("--cap_len", type=int, default=CAP_LEN)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default=None, help="also write the report to this file")
    args = p.parse_args(argv)

    import torch
    from fp_spectrogram import FingerprintModel, iq_segments_to_specs
    from train_model import _mix_noise
    # The badge rule and the mapping from a badge to a name live in one place only.
    from tools.evaluate import badge_of

    fp = FingerprintModel(args.model)
    if args.expect and args.expect not in fp.classes:
        raise SystemExit(f"the model has no class {args.expect!r}. It knows "
                         f"{', '.join(fp.classes)}.")
    share = fp.min_seg_share
    pool = noise_pool(args.noise_dir, args.cap_len)
    conds = _conditions(args.snr)

    print(f"\nModel   : {args.model}")
    print(f"Classes : {', '.join(fp.classes)}")
    print(f"Rule    : vote_thresh {fp.vote_thresh}, min_share {share}, "
          f"unknown_thresh {fp.unknown_thresh}")
    print(f"Noise   : {len(pool)} recorded captures from {args.noise_dir}")
    print(f"Truth   : {args.expect or 'not given'}\n")

    hdr = (f"{'clip':<22}{'snr':>7}{'floor':>8}{'n':>6}{'named right':>13}{'clear':>7}"
           f"{'named wrong':>13}{'correct':>9}{'share p50':>11}{'share p90':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for clip in args.clips:
        caps = clip_captures(clip, args.captures, args.cap_len)
        for snr in conds:
            margin = floor_margin_db(caps, pool, snr)
            # A new generator for each condition, thus every row of the table meets
            # the same noise captures in the same order and the rows compare.
            rng = np.random.RandomState(args.seed)
            names, shares = [], []
            for cap in caps:
                iq = cap if snr is None else _mix_noise(cap, pool, rng, snr, snr)
                specs = iq_segments_to_specs(iq, fp.seg_len, fp.seg_hop, fp.n_fft,
                                             fp.stft_hop, max_segs=fp.infer_max_segs)
                with torch.no_grad():
                    x = torch.from_numpy(specs.astype(np.float32))
                    seg_probs = torch.softmax(fp.net(x), dim=1).numpy()
                names.append(badge_of(fp, seg_probs, share))
                if args.expect:
                    win = seg_probs.argmax(1)
                    i = fp.classes.index(args.expect)
                    shares.append(float(((win == i) &
                                         (seg_probs.max(1) >= fp.vote_thresh)).mean()))
            n = len(names)
            right = sum(x == args.expect for x in names) if args.expect else 0
            clear = sum(x == "clear" for x in names)
            wrong = n - right - clear
            s50 = float(np.median(shares)) if shares else 0.0
            s90 = float(np.percentile(shares, 90)) if shares else 0.0
            txt = "clean" if snr is None else f"{snr:.0f} dB"
            m_txt = "-" if margin is None else f"{margin:+.0f} dB"
            print(f"{Path(clip).stem:<22}{txt:>7}{m_txt:>8}{n:>6}{right:>13}{clear:>7}"
                  f"{wrong:>13}{right / n:>9.1%}{s50:>11.3f}{s90:>11.3f}")
            rows.append({"clip": Path(clip).name, "snr_db": snr, "n": n,
                         "floor_margin_db": margin,
                         "named_right": right, "clear": clear, "named_wrong": wrong,
                         "share_p50": s50, "share_p90": s90,
                         "names": {c: names.count(c) for c in set(names)}})

    bad = [r for r in rows if r["floor_margin_db"] is not None
           and r["floor_margin_db"] < 10.0]
    print("\n`floor` is how far the added noise sits above the quiet segments of the "
          "clip.\nBelow about +10 dB the model reads the floor of the source recording "
          "and not the drone,\nand it names that floor. The `clean` rows have no added "
          "floor at all.")
    if bad:
        print(f"[warn] {len(bad)} row(s) have less than +10 dB of margin. Those rows "
              f"measure the\n       source recording. See §9 Phase 5b item 6 of "
              f"NOTES.md.")
    if args.expect:
        print("\n`clear` is a miss and `named wrong` is another class. The two do not "
              "overlap.\nRead the control clip first: a control that fails makes the "
              "other row meaningless. A\ncontrol of the right class is not enough — add "
              "a window that holds no transmitter.")

    if args.json:
        report = {
            "model": args.model, "expect": args.expect,
            "rule": {"vote_thresh": fp.vote_thresh, "min_share": share,
                     "unknown_thresh": fp.unknown_thresh},
            "noise_dir": str(Path(args.noise_dir).resolve()),
            "noise_files": len(pool), "captures": args.captures,
            "cap_len": args.cap_len, "seed": args.seed,
            "rows": rows, "git_commit": _git_commit(),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[ok] report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
