"""Measure the LO leakage of the PlutoSDR, and the tune latency.

This closes the one open item of the section 9 Phase 1 of NOTES.md. The DC blocker
`remove_dc()` is proven against a synthetic offset only. Nobody has measured the real
residual of this radio.

The program receives. It does not transmit.

It reports three numbers for each frequency and each gain:

  raw DC     the height of the 0 Hz bin above the median of the other bins, in dB.
             This is the leakage as `_peak_hold_psd` and the spectrogram see it.
  after DC   the same height after `remove_dc()`. A small value means the blocker
             works on this radio.
  offset     20*log10(|mean(iq)| / rms(iq)). The size of the constant, against the
             signal that carries it.

It also times a retune, against the HOP_SETTLE_MS of the application.

Usage:
    python measure_lo.py                      # the URI of PLUTO_URI, or the default
    python measure_lo.py --uri ip:192.168.3.1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: F401

from _support import stub_hardware                                   # noqa: E402

stubbed = stub_hardware()
if "adi" in stubbed:
    sys.exit("adi is not installed. Install libiio, then pip install pyadi-iio.")

import adi                                                           # noqa: E402
from fp_spectrogram import remove_dc                                 # noqa: E402
from terminal import (SAMPLE_RATE, RX_BW_HZ, GAIN,                   # noqa: E402
                         HOP_SETTLE_MS, HOP_DWELL_MS, FFT_BINS)

# The control frequencies are outside the 2.4 GHz band, where the air is quieter.
# The in-band ones are what the application really uses.
FREQS_HZ = [1_000_000_000, 1_500_000_000,
            2_400_000_000, 2_437_000_000, 2_480_000_000]
GAINS_DB = [0, 10, 20, 40, 60, 70]
N_BUFFERS = 4          # after the discard
N_DISCARD = 2          # buffers thrown away after a retune


def dc_height_db(iq, n_fft=FFT_BINS):
    """Give the height of the 0 Hz bin above the median of the other bins, in dB."""
    n = (len(iq) // n_fft) * n_fft
    if n == 0:
        return float("nan")
    frames = np.asarray(iq[:n], dtype=np.complex64).reshape(-1, n_fft)
    frames = frames * np.blackman(n_fft)
    psd = np.mean(np.abs(np.fft.fft(frames, axis=1)) ** 2, axis=0)
    dc = psd[0]
    others = np.delete(psd, [0, 1, n_fft - 1])          # the leakage of the window
    return 10.0 * np.log10(dc / np.median(others) + 1e-30)


def offset_db(iq):
    """Give the size of the constant against the signal, in dB."""
    iq = np.asarray(iq, dtype=np.complex64)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    return 20.0 * np.log10(abs(complex(iq.mean())) / (rms + 1e-30) + 1e-30)


def capture(sdr, freq_hz, gain_db):
    """Retune, discard, and give N_BUFFERS buffers joined into one."""
    sdr.rx_destroy_buffer()
    sdr.rx_hardwaregain_chan0 = int(gain_db)
    sdr.rx_lo = int(freq_hz)
    for _ in range(N_DISCARD):
        sdr.rx()
    return np.concatenate([np.asarray(sdr.rx()) for _ in range(N_BUFFERS)])


def time_retune(sdr, f_a, f_b, n=10):
    """Time a retune and a buffer, so the settle of the application can be judged."""
    lo_ms, rx_ms = [], []
    for i in range(n):
        f = f_a if i % 2 == 0 else f_b
        sdr.rx_destroy_buffer()
        t0 = time.perf_counter()
        sdr.rx_lo = int(f)
        t1 = time.perf_counter()
        sdr.rx()
        t2 = time.perf_counter()
        lo_ms.append((t1 - t0) * 1e3)
        rx_ms.append((t2 - t1) * 1e3)
    return float(np.median(lo_ms)), float(np.median(rx_ms))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default=os.environ.get("PLUTO_URI", "ip:192.168.2.1"))
    ap.add_argument("--out", default="lo_leakage.json")
    args = ap.parse_args()

    print(f"connecting to {args.uri} ...")
    try:
        sdr = adi.Pluto(args.uri)
    except Exception as exc:
        sys.exit(f"no radio at {args.uri}: {exc}\n"
                 f"Give the correct address with --uri, or set PLUTO_URI.")

    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(RX_BW_HZ)
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        print("  warning: the radio refused the manual gain mode.")
    sdr.rx_buffer_size = max(1024, int(SAMPLE_RATE * HOP_DWELL_MS / 1000.0))

    print(f"  sample rate {SAMPLE_RATE/1e6:.1f} Msps, RX bandwidth "
          f"{RX_BW_HZ/1e6:.1f} MHz, buffer {sdr.rx_buffer_size} samples")
    print(f"  the application uses gain {GAIN} dB and a settle of {HOP_SETTLE_MS} ms\n")

    head = f"{'freq MHz':>10} {'gain dB':>8} {'raw DC dB':>11} " \
           f"{'after DC dB':>12} {'offset dB':>10}"
    print(head)
    print("-" * len(head))

    rows = []
    for f in FREQS_HZ:
        for g in GAINS_DB:
            try:
                iq = capture(sdr, f, g)
            except Exception as exc:
                print(f"{f/1e6:10.1f} {g:8d}   capture failed: {exc}")
                continue
            raw = dc_height_db(iq)
            aft = dc_height_db(remove_dc(iq))
            off = offset_db(iq)
            rows.append({"freq_hz": f, "gain_db": g, "raw_dc_db": raw,
                         "after_dc_db": aft, "offset_db": off})
            mark = "  <- the app default" if g == GAIN else ""
            print(f"{f/1e6:10.1f} {g:8d} {raw:11.2f} {aft:12.2f} {off:10.2f}{mark}")

    lo_ms, rx_ms = time_retune(sdr, FREQS_HZ[2], FREQS_HZ[-1])
    print(f"\nretune: rx_lo takes {lo_ms:.1f} ms, the buffer takes {rx_ms:.1f} ms, "
          f"against a settle of {HOP_SETTLE_MS} ms and a dwell of {HOP_DWELL_MS} ms")

    if rows:
        worst = max(rows, key=lambda r: r["raw_dc_db"])
        best_after = max(rows, key=lambda r: r["after_dc_db"])
        print(f"\nthe worst raw leakage is {worst['raw_dc_db']:.2f} dB at "
              f"{worst['freq_hz']/1e6:.1f} MHz and gain {worst['gain_db']} dB")
        print(f"the worst residual after remove_dc is {best_after['after_dc_db']:.2f} dB "
              f"at {best_after['freq_hz']/1e6:.1f} MHz and gain "
              f"{best_after['gain_db']} dB")

    out = {"uri": args.uri, "sample_rate": SAMPLE_RATE, "rx_bw": RX_BW_HZ,
           "n_fft": FFT_BINS, "buffer": int(sdr.rx_buffer_size),
           "rx_lo_ms": lo_ms, "rx_buffer_ms": rx_ms, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
