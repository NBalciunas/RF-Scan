"""Call the real _peak_hold_psd on the real captures, before and after the change."""
import sys, json, numpy as np
from pathlib import Path
import _paths  # noqa: F401
from _support import stub_hardware; stub_hardware()
import terminal
from terminal import SweepWorker, FFT_BINS

def dc_excess(psd_db):
    """psd_db is fftshifted, thus 0 Hz is the middle bin."""
    k = FFT_BINS // 2
    others = np.delete(psd_db, [k-1, k, k+1])
    return float(psd_db[k] - np.median(others))

class Bare:                     # a throwaway holder, as test_dsp does
    pass

def run(iq):
    o = Bare()
    o._BLACKMAN = np.blackman(FFT_BINS).astype(np.float32)
    return SweepWorker._peak_hold_psd(o, iq)

IQ = Path(sys.argv[1] if len(sys.argv) > 1 else "iq"); meta = json.loads((IQ/"index.json").read_text())
print(f"{'capture':>12} {'0 Hz bin, dB above median':>28}")
vals = []
for c in meta["captures"]:
    iq = np.load(IQ/c["file"])
    v = dc_excess(run(iq)); vals.append(v)
    if c["gain_db"] in (10, 40, 70):
        print(f"{c['file'].replace('.npy',''):>12} {v:28.2f}")
print(f"\nmean over all 24: {np.mean(vals):.2f} dB, worst {np.max(vals):.2f} dB")
print("before the change the worst was 38.70 dB and the mean was 13.5 dB")
