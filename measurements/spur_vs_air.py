"""Tell an internal spur from a signal in the air.

These two discriminators separate a CONSTANT signal from a BURSTY one. They do not
separate the radio from the air: a transmitter that never stops looks exactly like a
spur of the receiver. Only a capture with a 50 ohm load can separate the two. The run
of 2026-08-10 called 2440 MHz a spur of the radio on antenna data, which the method
does not support.

  hold - mean   A spur of the radio is constant, thus the maximum of the windows and
                the mean of the windows agree. Traffic in the air is bursty, thus the
                maximum stands far above the mean.

  repeat        A spur does not move. The program measures each frequency twice, one
                second apart, and gives the difference. Traffic changes.

The program receives. It does not transmit.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _support import stub_hardware                                   # noqa: E402
if "adi" in stub_hardware():
    sys.exit("adi is not installed.")

import adi                                                           # noqa: E402
from terminal import SAMPLE_RATE, RX_BW_HZ                        # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in ("antenna", "load"):
    sys.exit("The first argument must be 'antenna' or 'load'. This program can not\n"
             "see the termination, and a wrong one invalidates every conclusion.")
TERMINATION = sys.argv[1]
URI = sys.argv[2] if len(sys.argv) > 2 else "ip:192.168.2.1"
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else f"spur_vs_air_{TERMINATION}.json")

NF = 1024
WIN = np.blackman(NF).astype(np.float32)
BUF = 262144
GAINS = [40, 60]
FREQS = [2_390_000_000, 2_400_000_000, 2_410_000_000, 2_437_000_000,
         2_440_000_000, 2_450_000_000, 2_460_000_000, 2_470_000_000,
         2_475_000_000, 2_480_000_000, 2_485_000_000, 2_490_000_000]


def stats(iq, half=16):
    """Give the DC excess of the peak hold and of the mean, in dB."""
    n = len(iq) // NF
    w = iq[:n * NF].reshape(n, NF)
    w = w - w.mean(axis=1, keepdims=True)
    p = np.abs(np.fft.fft(w * WIN, axis=1)) ** 2
    hold, mean = p.max(axis=0), p.mean(axis=0)
    far = [i % NF for i in range(-64, 65)]
    idx = [i % NF for i in range(-half, half + 1)]
    h = 10 * np.log10(hold[idx].mean() / np.median(np.delete(hold, far)) + 1e-30)
    m = 10 * np.log10(mean[idx].mean() / np.median(np.delete(mean, far)) + 1e-30)
    return float(h), float(m)


def main():
    sdr = adi.Pluto(URI)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(RX_BW_HZ)
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        pass
    sdr.rx_buffer_size = BUF

    def measure(f, g):
        sdr.rx_destroy_buffer()
        sdr.rx_hardwaregain_chan0 = int(g)
        sdr.rx_lo = int(f)
        sdr.rx()
        return stats(np.asarray(sdr.rx(), dtype=np.complex64))

    print(f"buffer {BUF} samples ({BUF/SAMPLE_RATE*1e3:.1f} ms), "
          f"{NF//1} bin FFT, termination: {TERMINATION}\n")
    head = (f"{'freq MHz':>9} {'gain':>5} {'hold dB':>8} {'mean dB':>8} "
            f"{'hold-mean':>10} {'repeat d':>9}  verdict")
    print(head); print("-" * len(head))

    rows = []
    for f in FREQS:
        for g in GAINS:
            h1, m1 = measure(f, g)
            time.sleep(1.0)
            h2, m2 = measure(f, g)
            hm = h1 - m1
            rep = abs(h2 - h1)
            if h1 < 6.0:
                v = "quiet"
            elif hm < 6.0 and rep < 3.0:
                v = "constant"
            elif hm > 10.0 or rep > 6.0:
                v = "bursty, thus the air"
            else:
                v = "unclear"
            rows.append({"freq_hz": f, "gain_db": g, "hold_db": h1, "mean_db": m1,
                         "hold_minus_mean": hm, "repeat_delta": rep, "verdict": v})
            print(f"{f/1e6:9.0f} {g:5d} {h1:8.2f} {m1:8.2f} {hm:10.2f} "
                  f"{rep:9.2f}  {v}")

    OUT.write_text(json.dumps({"termination": TERMINATION, "rows": rows},
                              indent=2, default=float), encoding="utf-8")
    spurs = sorted({r["freq_hz"] for r in rows if r["verdict"] == "constant"})
    print(f"\nfrequencies where the radio itself makes the feature: "
          f"{[f/1e6 for f in spurs] if spurs else 'none'}")
    print(f"written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
