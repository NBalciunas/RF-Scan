"""Step the LO across the band and measure what sits at the centre of each hop.

A zero-IF receiver puts LO leakage and a DC offset at 0 Hz, which is the middle of
the tuned band, at every frequency. The size of it changes with the tuning. This
program measures that size at each tune frequency, so the hop plan of §9 Phase 1
rests on data.

**Run it twice, once with the antenna and once with a 50 ohm load.** The difference
between the two runs is the signals in the room. What stays with the load belongs to
the radio. One run alone can not separate the two, and the run of 2026-08-10 was
believed to be a load run when it was an antenna run. That error is the reason the
termination is now a required argument: this program writes down what you tell it and
can not verify it, so state it and state it correctly.

The program receives. It does not transmit.

Usage:
    python spur_survey.py antenna  [uri] [out.json]
    python spur_survey.py load     [uri] [out.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _support import stub_hardware                                   # noqa: E402
if "adi" in stub_hardware():
    sys.exit("adi is not installed.")

import adi                                                           # noqa: E402
from terminal import SAMPLE_RATE, RX_BW_HZ                           # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in ("antenna", "load"):
    sys.exit(__doc__.strip().splitlines()[-2].strip() + "\n"
             "The first argument must be 'antenna' or 'load'.")
TERMINATION = sys.argv[1]
URI = sys.argv[2] if len(sys.argv) > 2 else "ip:192.168.2.1"
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else f"spur_survey_{TERMINATION}.json")

NF = 1024
WIN = np.blackman(NF).astype(np.float32)
BUF = 65536                    # 6.6 ms at 10 Msps. Enough, and fast over RNDIS.
GAINS = [10, 40]               # the app default, and a sensitive one



def dc_excess_db(iq):
    """Height of the 0 Hz bin above the median bin, after a mean for each window."""
    n = len(iq) // NF
    w = iq[:n * NF].reshape(n, NF)
    w = w - w.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.fft(w * WIN, axis=1)).max(axis=0) ** 2
    med = np.median(np.delete(mag, list(range(-3, 4))))
    return float(10.0 * np.log10(mag[0] / med + 1e-30))


def near_dc_excess_db(iq, half_bins=16):
    """The same, but over the bins within +-half_bins of 0 Hz.

    The feature has width, so a single bin understates it."""
    n = len(iq) // NF
    w = iq[:n * NF].reshape(n, NF)
    w = w - w.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.fft(w * WIN, axis=1)).max(axis=0) ** 2
    idx = [i % NF for i in range(-half_bins, half_bins + 1)]
    med = np.median(np.delete(mag, [i % NF for i in range(-64, 65)]))
    return float(10.0 * np.log10(np.mean(mag[idx]) / med + 1e-30))


def main():
    sdr = adi.Pluto(URI)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(RX_BW_HZ)
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        pass
    sdr.rx_buffer_size = BUF

    # 1 MHz across the ISM band and its edges, then 50 kHz around 2440, which is the
    # one place where the centre feature stands above its neighbours.
    coarse = list(range(2_380_000_000, 2_501_000_000, 1_000_000))
    fine = list(range(2_435_000_000, 2_445_050_000, 50_000))
    freqs = sorted(set(coarse + fine))

    print(f"{len(freqs)} tune frequencies, gains {GAINS}, "
          f"buffer {BUF} samples, termination: {TERMINATION}\n")
    print(f"{'freq MHz':>10} " +
          " ".join(f"{'g'+str(g)+' DC':>9} {'g'+str(g)+' ±160k':>10}" for g in GAINS))

    rows = []
    for f in freqs:
        rec = {"freq_hz": f}
        for g in GAINS:
            sdr.rx_destroy_buffer()
            sdr.rx_hardwaregain_chan0 = int(g)
            sdr.rx_lo = int(f)
            sdr.rx()
            iq = np.asarray(sdr.rx(), dtype=np.complex64)
            rec[f"dc_g{g}"] = dc_excess_db(iq)
            rec[f"near_g{g}"] = near_dc_excess_db(iq)
        rows.append(rec)
        cols = " ".join(f"{rec[f'dc_g{g}']:9.2f} {rec[f'near_g{g}']:10.2f}"
                        for g in GAINS)
        print(f"{f/1e6:10.3f} {cols}")

    OUT.write_text(json.dumps({"termination": TERMINATION, "rows": rows},
                          indent=2, default=float), encoding="utf-8")

    bad = [r for r in rows if max(r[f"near_g{g}"] for g in GAINS) > 6.0]
    print(f"\n{len(bad)} of {len(rows)} tune frequencies exceed 6 dB near 0 Hz")
    if bad:
        lo, hi = min(r["freq_hz"] for r in bad), max(r["freq_hz"] for r in bad)
        print(f"they span {lo/1e6:.1f} to {hi/1e6:.1f} MHz")
        print("each point above the limit:")
        for r in bad:
            print(f"  {r['freq_hz']/1e6:9.1f} MHz  "
                  f"near {max(r[f'near_g{g}'] for g in GAINS):6.2f} dB")
    print(f"\nwritten to {OUT.resolve()}")


if __name__ == "__main__":
    main()
