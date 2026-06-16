"""
precompute.py  –  One-time conversion of raw .iq files → pre-computed PSD cache.

Supports both:
  • Narrowband files  – single center_freq, standard sample_rate
  • Wideband stitched files  – FreqMin / FreqMax tags present (from pluto_monitor.py)
    The wideband synthetic sample rate spans the whole recorded band, so a
    1024-sample FFT window gives a full-span spectrum in one shot.

Run this ONCE before training:
    python precompute.py
    python precompute.py --data_dir ./training_data --cache_dir ./psd_cache

Output layout
─────────────
psd_cache/
    drone/
        capture_2400MHz_abc.npy    ← float32 array  (N_windows, FFT_BINS)
        capture_2400MHz_abc.json   ← center_freq, freq_min, freq_max,
                                      freq_span, scale_factor, n_windows,
                                      is_wideband
        ...
    noise/
        ...
    stats.json   ← psd_mean, psd_std, freq_mean, freq_std,
                   span_mean, span_std, classes

After this, run train_fast_quick.py.
"""

import os
import json
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── config (must match train_fast_quick.py) ────────────────────────────────────
DATA_DIR   = "./training_data"
CACHE_DIR  = "./psd_cache"
FFT_BINS   = 1024
WINDOW_HOP = 512          # samples between successive PSD windows

_BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# XML metadata loader
# ══════════════════════════════════════════════════════════════════════════════

def load_xml_meta(iq_path: Path) -> dict:
    """
    Parse the paired .xml file.  Returns a dict with all fields that may
    be present in either narrowband or wideband captures.

    Wideband files (written by pluto_monitor.py) additionally carry:
        FreqMin, FreqMax  →  full span edges in Hz
    These are used by the trainer as extra model inputs so the network
    knows what part of the spectrum it is looking at.
    """
    xml_path = iq_path.with_suffix(".xml")
    result = dict(
        center_freq  = 0,
        sample_rate  = 0,
        scale_factor = 1.0,
        freq_min     = None,   # None = not present (narrowband file)
        freq_max     = None,
        drone        = "",
        ref_snr      = float("nan"),
        xml_found    = False,
        is_wideband  = False,
    )
    if not xml_path.exists():
        return result
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        return result

    def get(tag, cast, default):
        el = root.find(tag)
        if el is None or not el.text:
            return default
        try:
            return cast(el.text.strip())
        except (ValueError, TypeError):
            return default

    result["center_freq"]  = get("CenterFrequency",  int,   0)
    result["sample_rate"]  = get("SampleRate",        int,   0)
    result["scale_factor"] = get("ScaleFactor",       float, 1.0)
    result["drone"]        = get("Drone",             str,   "")
    result["ref_snr"]      = get("ReferenceSNRLevel", float, float("nan"))
    result["xml_found"]    = True

    freq_min = get("FreqMin", int, None)
    freq_max = get("FreqMax", int, None)
    if freq_min is not None and freq_max is not None:
        result["freq_min"]    = freq_min
        result["freq_max"]    = freq_max
        result["is_wideband"] = True

    return result


# ══════════════════════════════════════════════════════════════════════════════
# IQ → PSD  (batched)
# ══════════════════════════════════════════════════════════════════════════════

def compute_psd_batch(raw: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Convert all overlapping FFT_BINS-sample windows of `raw` to
    Blackman-windowed, FFT-shifted PSDs in dBFS — in one batched FFT call.

    Uses as_strided for a zero-copy window view so only the windowed copy
    is allocated, not the raw data replicated N times.

    Returns float32 array of shape (N_windows, FFT_BINS).
    """
    n = (len(raw) - FFT_BINS) // WINDOW_HOP + 1
    if n <= 0:
        return np.empty((0, FFT_BINS), dtype=np.float32)

    strides = (raw.strides[0] * WINDOW_HOP, raw.strides[0])
    chunks  = np.lib.stride_tricks.as_strided(raw, shape=(n, FFT_BINS), strides=strides)

    windowed = chunks * np.float32(scale) * _BLACKMAN   # (n, FFT_BINS) complex64 copy

    psds = 20.0 * np.log10(
        np.abs(np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)) / FFT_BINS + 1e-10
    )
    return psds.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def precompute(args):
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class sub-folders found in {data_dir}")

    classes = [d.name for d in class_dirs]
    print(f"Classes: {classes}")

    # Accumulators for normalisation stats
    all_psds_for_stats: list[np.ndarray] = []   # list of (k, FFT_BINS) slices
    all_freqs: list[float] = []
    all_spans: list[float] = []
    stats_sample_limit = 300_000

    total_files   = sum(len(list(d.glob("*.iq"))) for d in class_dirs)
    files_done    = 0
    total_windows = 0
    t0            = time.time()

    for cls_dir in class_dirs:
        out_cls = cache_dir / cls_dir.name
        out_cls.mkdir(exist_ok=True)

        iq_files = sorted(cls_dir.glob("*.iq"))
        print(f"\n[{cls_dir.name}] {len(iq_files)} files")

        for fpath in iq_files:
            npy_path  = out_cls / fpath.with_suffix(".npy").name
            json_path = out_cls / fpath.with_suffix(".json").name

            # ── skip if already cached ────────────────────────────────────────
            if npy_path.exists() and json_path.exists() and not args.force:
                try:
                    with open(json_path) as f:
                        cached_meta = json.load(f)
                    files_done    += 1
                    total_windows += cached_meta["n_windows"]
                    all_freqs.append(cached_meta["center_freq"])
                    all_spans.append(cached_meta.get("freq_span", 0))
                    elapsed = time.time() - t0
                    eta = (elapsed / files_done) * (total_files - files_done)
                    wb_tag = "  [wideband]" if cached_meta.get("is_wideband") else ""
                    print(f"  skip (cached)  {fpath.name}{wb_tag}"
                          f"  [{files_done}/{total_files}  ETA {eta/60:.1f} min]")
                    continue
                except Exception:
                    pass   # corrupted entry — recompute

            # ── load ──────────────────────────────────────────────────────────
            try:
                meta = load_xml_meta(fpath)
                raw  = np.fromfile(str(fpath), dtype=np.complex64)
            except Exception as e:
                print(f"  ERROR {fpath.name}: {e}")
                files_done += 1
                continue

            # ── compute PSD windows (batched) ─────────────────────────────────
            windows = compute_psd_batch(raw, meta["scale_factor"])
            del raw

            if len(windows) == 0:
                print(f"  skip (too short)  {fpath.name}")
                files_done += 1
                continue

            # ── derive freq_span for wideband files ───────────────────────────
            freq_span = 0
            if meta["is_wideband"] and meta["freq_min"] is not None:
                freq_span = meta["freq_max"] - meta["freq_min"]

            # ── save .npy + .json ─────────────────────────────────────────────
            np.save(str(npy_path), windows)
            with open(json_path, "w") as f:
                json.dump({
                    "center_freq" : meta["center_freq"],
                    "freq_min"    : meta["freq_min"],
                    "freq_max"    : meta["freq_max"],
                    "freq_span"   : freq_span,
                    "scale_factor": meta["scale_factor"],
                    "n_windows"   : len(windows),
                    "xml_found"   : meta["xml_found"],
                    "is_wideband" : meta["is_wideband"],
                }, f)

            # ── accumulate stats sample ───────────────────────────────────────
            already = sum(len(a) for a in all_psds_for_stats)
            if already < stats_sample_limit:
                take = min(len(windows), stats_sample_limit - already)
                all_psds_for_stats.append(windows[:take])

            all_freqs.append(meta["center_freq"])
            all_spans.append(freq_span)
            total_windows += len(windows)
            files_done    += 1

            elapsed = time.time() - t0
            eta = (elapsed / files_done) * (total_files - files_done) if files_done else 0
            wb_tag = "  [wideband]" if meta["is_wideband"] else ""
            print(f"  {fpath.name}  {len(windows):>7,} windows{wb_tag}"
                  f"  [{files_done}/{total_files}  ETA {eta/60:.1f} min]")

    # ── normalisation stats ───────────────────────────────────────────────────
    print(f"\nComputing normalisation stats from {sum(len(a) for a in all_psds_for_stats):,} windows…")
    if not all_psds_for_stats:
        raise RuntimeError("No PSD windows were produced. Check that .iq files are valid and non-empty.")
    psd_arr  = np.concatenate(all_psds_for_stats, axis=0)
    freq_arr = np.array(all_freqs, dtype=np.float64)
    span_arr = np.array(all_spans, dtype=np.float64)

    psd_mean  = float(psd_arr.mean())
    psd_std   = float(psd_arr.std())
    freq_mean = float(freq_arr.mean())
    freq_std  = float(freq_arr.std())
    span_mean = float(span_arr.mean())
    span_std  = float(span_arr.std())

    # Guard against degenerate stddevs (e.g. only one frequency used)
    if psd_std  < 1e-6: psd_std  = 1.0
    if freq_std < 1e-6: freq_std = 1.0
    if span_std < 1e-6: span_std = 1.0

    n_wideband = sum(1 for s in all_spans if s > 0)
    stats = {
        "classes"       : classes,
        "n_classes"     : len(classes),
        "fft_bins"      : FFT_BINS,
        "window_hop"    : WINDOW_HOP,
        "total_windows" : total_windows,
        "n_wideband_files": n_wideband,
        "n_narrowband_files": total_files - n_wideband,
        "psd_mean"      : psd_mean,
        "psd_std"       : psd_std,
        "freq_mean"     : freq_mean,
        "freq_std"      : freq_std,
        "span_mean"     : span_mean,
        "span_std"      : span_std,
    }
    stats_path = cache_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    disk_gb = sum(p.stat().st_size for p in cache_dir.rglob("*.npy")) / 1e9

    print(f"\n✓ Pre-compute complete")
    print(f"  Total windows      : {total_windows:,}")
    print(f"  Wideband files     : {n_wideband}")
    print(f"  Narrowband files   : {total_files - n_wideband}")
    print(f"  Cache size         : {disk_gb:.2f} GB  ({cache_dir})")
    print(f"  Time elapsed       : {elapsed/60:.1f} min")
    print(f"  PSD mean/std       : {psd_mean:.2f} / {psd_std:.2f} dBFS")
    print(f"  Freq mean/std      : {freq_mean/1e6:.3f} / {freq_std/1e6:.3f} MHz")
    print(f"  Span mean/std      : {span_mean/1e6:.3f} / {span_std/1e6:.3f} MHz")
    print(f"\nNow run:  python train_fast_quick.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default=DATA_DIR)
    p.add_argument("--cache_dir", default=CACHE_DIR)
    p.add_argument("--force",     action="store_true",
                   help="Re-compute even if .npy already exists")
    return p.parse_args()


if __name__ == "__main__":
    precompute(parse_args())
