"""Per-segment statistics of the real artifact, which is what the classifier sees."""
import sys, json, numpy as np
from pathlib import Path
import _paths  # noqa: F401
from _support import stub_hardware; stub_hardware()
from fp_spectrogram import SEG_LEN, SEG_HOP, N_FFT

IQ = Path(sys.argv[1] if len(sys.argv) > 1 else "iq"); meta = json.loads((IQ/"index.json").read_text())
print("per 4096-sample segment, on the real captures")
print(f"{'capture':>12} {'|mean|/rms dB':>14} {'0Hz bin dB raw':>15} {'after seg mean':>15}")
allm, allr, alla = [], [], []
for c in meta["captures"]:
    if c["gain_db"] not in (10, 40, 70): continue
    iq = np.load(IQ/c["file"])
    ms, raws, afts = [], [], []
    for i in range(24):
        s = iq[i*SEG_HOP:i*SEG_HOP+SEG_LEN]
        rms = np.sqrt(np.mean(np.abs(s)**2))
        ms.append(20*np.log10(abs(complex(s.mean()))/rms + 1e-30))
        for tag, x in (("raw", s), ("aft", s - s.mean())):
            fr = np.asarray(x).reshape(-1, N_FFT) * np.hanning(N_FFT)
            p = np.abs(np.fft.fft(fr, axis=1)).mean(axis=0)**2
            v = 10*np.log10(p[0]/np.median(p[1:]) + 1e-30)
            (raws if tag=="raw" else afts).append(v)
    allm += ms; allr += raws; alla += afts
    print(f"{c['file'].replace('.npy',''):>12} {np.mean(ms):14.1f} {np.mean(raws):15.2f} {np.mean(afts):15.2f}")
print(f"\n{'MEAN':>12} {np.mean(allm):14.1f} {np.mean(allr):15.2f} {np.mean(alla):15.2f}")
