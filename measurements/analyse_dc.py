"""Measure the real LO leakage on both paths of the application.

The two paths remove the constant differently, thus each needs its own measurement.

  the sweep path       terminal._peak_hold_psd. One mean over the whole buffer,
                       1024 bins, a Blackman window, the maximum of all the windows.
                       The number that matters is the height of the 0 Hz bin above
                       the median bin, in dB, because the scanner looks for a peak.

  the classifier path  fp_spectrogram.iq_to_spectrogram. remove_dc() on each segment
                       of SEG_LEN samples, 256 bins, a Hanning window, and then the
                       image is standardized. The number that matters is the height
                       of the DC row above the median row, in sigma, because that is
                       what the network sees. test_spectrogram uses the same measure.

Each path is measured three ways: with the constant left in, with the removal that
the application does now, and with the removal done on each window, which is the
cheapest correction that could work.
"""

import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _support import stub_hardware                                   # noqa: E402
stub_hardware()

from fp_spectrogram import (iq_to_spectrogram, remove_dc,            # noqa: E402
                            N_FFT, SEG_LEN, SEG_HOP)
from terminal import FFT_BINS                                     # noqa: E402

IQ = Path(sys.argv[1] if len(sys.argv) > 1 else "iq")
_BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)


def psd_path(iq, mode):
    """Reproduce _peak_hold_psd and give the height of the 0 Hz bin, in dB."""
    nwin = len(iq) // FFT_BINS
    w = iq[:nwin * FFT_BINS].reshape(nwin, FFT_BINS)
    if mode == "buffer_mean":            # what the application does now
        w = w - iq[:nwin * FFT_BINS].mean()
    elif mode == "window_mean":          # the cheapest correction
        w = w - w.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.fft(w * _BLACKMAN, axis=1)).max(axis=0)
    dc = mag[0] ** 2
    others = np.delete(mag, [0, 1, FFT_BINS - 1]) ** 2
    return 10.0 * np.log10(dc / np.median(others) + 1e-30)


def spec_path(iq, mode, max_segs=32):
    """Give the height of the DC row above the median row, in sigma."""
    n = (len(iq) - SEG_LEN) // SEG_HOP + 1
    idx = np.linspace(0, n - 1, min(n, max_segs)).astype(int)
    out = []
    for i in idx:
        seg = iq[i * SEG_HOP:i * SEG_HOP + SEG_LEN]
        if mode == "raw":                # no removal at all
            seg = seg + 0.0
        elif mode == "segment_mean":     # what the application does now
            seg = remove_dc(seg)
        elif mode == "window_mean":
            seg = remove_dc(seg)
        spec = iq_to_spectrogram(seg, n_fft=N_FFT) if mode != "raw" else None
        if mode == "raw":
            # iq_to_spectrogram always removes the mean, thus build it by hand.
            s = np.asarray(seg, dtype=np.complex64)
            nf = (len(s) - N_FFT) // 64 + 1
            j = np.arange(N_FFT)[None, :] + 64 * np.arange(nf)[:, None]
            fr = s[j] * np.hanning(N_FFT).astype(np.float32)
            S = np.fft.fftshift(np.fft.fft(fr, axis=1), axes=1)
            mag = 20.0 * np.log10(np.abs(S) / N_FFT + 1e-10)
            spec = mag.T.astype(np.float32)
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
            spec = spec[None, :, :]
        img = spec[0]
        dc_row = img[N_FFT // 2]                     # fftshift puts 0 Hz at the middle
        out.append(float(dc_row.mean() - np.median(img)))
    return float(np.mean(out))


def main():
    meta = json.loads((IQ / "index.json").read_text(encoding="utf-8"))
    print(f"termination: {meta['termination']}")
    print(f"sample rate {meta['sample_rate']/1e6:.0f} Msps, "
          f"buffer {meta['buffer']} samples, "
          f"SEG_LEN {SEG_LEN}, N_FFT {N_FFT}, PSD bins {FFT_BINS}\n")

    head = (f"{'freq MHz':>9} {'gain':>5} {'rms':>7} | "
            f"{'PSD raw':>8} {'PSD now':>8} {'PSD win':>8} | "
            f"{'img raw':>8} {'img now':>8}")
    print(head)
    print("-" * len(head))

    rows = []
    for c in meta["captures"]:
        iq = np.load(IQ / c["file"])
        rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        r = {
            "freq_hz": c["freq_hz"], "gain_db": c["gain_db"], "rms": rms,
            "psd_raw":  psd_path(iq, "none"),
            "psd_now":  psd_path(iq, "buffer_mean"),
            "psd_win":  psd_path(iq, "window_mean"),
            "img_raw":  spec_path(iq, "raw"),
            "img_now":  spec_path(iq, "segment_mean"),
        }
        rows.append(r)
        print(f"{c['freq_hz']/1e6:9.0f} {c['gain_db']:5d} {rms:7.1f} | "
              f"{r['psd_raw']:8.2f} {r['psd_now']:8.2f} {r['psd_win']:8.2f} | "
              f"{r['img_raw']:8.3f} {r['img_now']:8.3f}")

    print("\nPSD columns are dB above the median bin. img columns are sigma above "
          "the median of the standardized image.")

    worst_now = max(rows, key=lambda r: r["psd_now"])
    worst_win = max(rows, key=lambda r: r["psd_win"])
    worst_img = max(rows, key=lambda r: abs(r["img_now"]))
    print(f"\nsweep path, the removal of today : worst {worst_now['psd_now']:.2f} dB "
          f"at {worst_now['freq_hz']/1e6:.0f} MHz gain {worst_now['gain_db']}")
    print(f"sweep path, a mean for each window: worst {worst_win['psd_win']:.2f} dB "
          f"at {worst_win['freq_hz']/1e6:.0f} MHz gain {worst_win['gain_db']}")
    print(f"classifier path, today           : worst {worst_img['img_now']:+.3f} sigma "
          f"at {worst_img['freq_hz']/1e6:.0f} MHz gain {worst_img['gain_db']}")

    (IQ / "dc_analysis.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten to {(IQ / 'dc_analysis.json').resolve()}")


if __name__ == "__main__":
    main()
