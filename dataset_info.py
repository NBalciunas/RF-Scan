"""Report what fingerprint_data/ holds. Type `python dataset_info.py`.

The trainer tells you nothing until it runs, and by then the recording session is
over. This tool reads the folders and the .json sidecars and it answers the questions
that decide whether a training run can mean anything:

  * How many captures, segments and seconds does each class hold?
  * Does each class have two sessions or more? With one session the split is random
    and the accuracy has no meaning.
  * Is there a noise class? Without it the energy gate and the Auto mode do nothing.
  * Is the gain the same in every session? The energy gate compares a device against
    a noise level in raw dB. A gain change makes that comparison false.

    python dataset_info.py
    python dataset_info.py --data_dir ./other_data
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from fp_spectrogram import SEG_LEN, SEG_HOP

BYTES_PER_SAMPLE = 8            # complex64
NOISE_CLASS = "noise"


def _natkey(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _sidecar(iq_path):
    p = iq_path.with_suffix(".json")
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def scan(data_dir: Path):
    """Give {class: {session: {...}}} plus the values that each session used."""
    out = {}
    for cls_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        sessions = defaultdict(lambda: {"files": 0, "bytes": 0, "segments": 0,
                                        "seconds": 0.0, "gain": set(),
                                        "sample_rate": set(), "freqs": set(),
                                        "no_sidecar": 0, "short": []})
        for f in cls_dir.rglob("*.iq"):
            s = sessions[f.parent.name]
            try:
                size = f.stat().st_size
            except OSError:
                continue
            n = size // BYTES_PER_SAMPLE
            s["files"] += 1
            s["bytes"] += size
            s["segments"] += max(0, (n - SEG_LEN) // SEG_HOP + 1)
            meta = _sidecar(f)
            if meta:
                sr = float(meta.get("sample_rate", 0)) or None
                if sr:
                    s["sample_rate"].add(sr)
                    s["seconds"] += n / sr
                if "gain_db" in meta:
                    s["gain"].add(int(meta["gain_db"]))
                if "center_freq" in meta:
                    s["freqs"].add(round(float(meta["center_freq"]) / 1e6, 3))
                # A short file means the write did not finish, and a full disk is the
                # usual cause. numpy reads such a file without a complaint, thus the
                # capture becomes shorter training data and nothing says so.
                want = int(meta.get("n_samples", 0))
                if want and n < want:
                    s["short"].append((f.name, n, want))
            else:
                s["no_sidecar"] += 1
        if sessions:
            out[cls_dir.name] = dict(sorted(sessions.items(),
                                            key=lambda kv: _natkey(kv[0])))
    return out


def _fmt_gb(b):
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def report(data_dir: Path):
    if not data_dir.is_dir():
        print(f"[error] {data_dir} does not exist. Record something first.")
        return 1
    tree = scan(data_dir)
    if not tree:
        print(f"[error] {data_dir} holds no .iq files.")
        return 1

    print(f"\nDataset: {data_dir.resolve()}")
    print(f"Segments counted at seg_len {SEG_LEN} / seg_hop {SEG_HOP}\n")
    head = f"{'class':<12}{'session':<12}{'files':>7}{'segments':>10}{'seconds':>9}{'size':>10}{'gain':>7}"
    print(head)
    print("-" * len(head))

    tot_files = tot_segs = tot_bytes = 0
    tot_secs = 0.0
    warnings, all_gains = [], set()
    for cls, sessions in tree.items():
        c_files = c_segs = c_bytes = 0
        c_secs = 0.0
        for name, s in sessions.items():
            gain = ",".join(str(g) for g in sorted(s["gain"])) or "?"
            all_gains |= s["gain"]
            print(f"{cls:<12}{name:<12}{s['files']:>7,}{s['segments']:>10,}"
                  f"{s['seconds']:>9.1f}{_fmt_gb(s['bytes']):>10}{gain:>7}")
            c_files += s["files"]; c_segs += s["segments"]
            c_bytes += s["bytes"]; c_secs += s["seconds"]
            if len(s["gain"]) > 1:
                warnings.append(f"{cls}/{name} mixes gains {sorted(s['gain'])}")
            if len(s["sample_rate"]) > 1:
                warnings.append(f"{cls}/{name} mixes sample rates "
                                f"{sorted(s['sample_rate'])}")
            if s["no_sidecar"]:
                warnings.append(f"{cls}/{name} has {s['no_sidecar']} captures with "
                                f"no .json sidecar")
            if s["short"]:
                worst = min(s["short"], key=lambda t: t[1] / max(1, t[2]))
                warnings.append(
                    f"{cls}/{name} has {len(s['short'])} truncated capture(s). The "
                    f"worst is {worst[0]}, {worst[1]:,} of {worst[2]:,} samples. A "
                    f"full disk during the record is the usual cause.")
        print(f"{cls:<12}{'TOTAL':<12}{c_files:>7,}{c_segs:>10,}"
              f"{c_secs:>9.1f}{_fmt_gb(c_bytes):>10}\n")
        tot_files += c_files; tot_segs += c_segs
        tot_bytes += c_bytes; tot_secs += c_secs
        if len(sessions) < 2:
            warnings.append(f"class '{cls}' has {len(sessions)} session. The trainer "
                            f"falls back to a random split, and that accuracy has no "
                            f"meaning. Record a second session with the antenna moved.")

    print(f"{'ALL':<24}{tot_files:>7,}{tot_segs:>10,}{tot_secs:>9.1f}"
          f"{_fmt_gb(tot_bytes):>10}")

    if not any(c.lower() == NOISE_CLASS for c in tree):
        warnings.append(f"there is no '{NOISE_CLASS}' class. The energy gate and the "
                        f"Auto mode both need it, and the badge can never say clear.")
    if len(all_gains) > 1:
        warnings.append(f"the dataset mixes RX gains {sorted(all_gains)}. The energy "
                        f"gate compares raw dB across classes, thus a gain change "
                        f"makes the gate threshold false.")
    devices = [c for c in tree if c.lower() != NOISE_CLASS]
    if len(devices) < 2:
        warnings.append(f"there is {len(devices)} device class. The goal needs two "
                        f"drones that the model separates.")

    print("\nFrequencies of each class (MHz):")
    for cls, sessions in tree.items():
        freqs = sorted({f for s in sessions.values() for f in s["freqs"]})
        shown = ", ".join(f"{f:.3f}" for f in freqs[:8])
        more = f" (+{len(freqs) - 8} more)" if len(freqs) > 8 else ""
        print(f"  {cls:<12}{shown or 'unknown'}{more}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nNo warnings. The dataset is ready to train.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Report what fingerprint_data/ holds")
    p.add_argument("--data_dir", default="./fingerprint_data")
    raise SystemExit(report(Path(p.parse_args().data_dir)))
