"""Capture raw IQ from the PlutoSDR to disk, for analysis without the radio.

The analysis of the LO leakage must follow the two paths of the application exactly,
and each path removes the constant differently. Thus the correct method is to capture
once and to analyse many times.

The program receives. It does not transmit.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401

from _support import stub_hardware                                   # noqa: E402

if "adi" in stub_hardware():
    sys.exit("adi is not installed.")

import adi                                                           # noqa: E402
from terminal import SAMPLE_RATE, RX_BW_HZ, HOP_DWELL_MS          # noqa: E402

URI = sys.argv[1] if len(sys.argv) > 1 else "ip:192.168.2.1"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "iq")
FREQS_HZ = [1_000_000_000, 2_400_000_000, 2_437_000_000, 2_480_000_000]
GAINS_DB = [0, 10, 20, 40, 60, 70]
N_DISCARD = 2

OUT.mkdir(parents=True, exist_ok=True)
sdr = adi.Pluto(URI)
sdr.sample_rate = int(SAMPLE_RATE)
sdr.rx_rf_bandwidth = int(RX_BW_HZ)
try:
    sdr.gain_control_mode_chan0 = "manual"
except Exception:
    print("warning: the radio refused the manual gain mode.")
sdr.rx_buffer_size = max(1024, int(SAMPLE_RATE * HOP_DWELL_MS / 1000.0))

index = []
for f in FREQS_HZ:
    for g in GAINS_DB:
        sdr.rx_destroy_buffer()
        sdr.rx_hardwaregain_chan0 = int(g)
        sdr.rx_lo = int(f)
        for _ in range(N_DISCARD):
            sdr.rx()
        iq = np.asarray(sdr.rx(), dtype=np.complex64)
        name = f"f{int(f/1e6)}_g{g}.npy"
        np.save(OUT / name, iq)
        index.append({"file": name, "freq_hz": f, "gain_db": g, "n": int(len(iq))})
        print(f"  {name}  {len(iq)} samples  rms {np.sqrt(np.mean(np.abs(iq)**2)):.1f}")

meta = {"uri": URI, "sample_rate": int(SAMPLE_RATE), "rx_bw": int(RX_BW_HZ),
        "buffer": int(sdr.rx_buffer_size), "termination": os.environ.get("RFSCAN_TERMINATION", "UNSTATED"),
        "captures": index}
(OUT / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"\n{len(index)} captures written to {OUT.resolve()}")
