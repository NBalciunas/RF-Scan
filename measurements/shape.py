"""Is the artifact a constant at 0 Hz, or a narrow tone sitting on it?"""
import numpy as np
from pathlib import Path

IQ = Path("iq")
NF = 1024
w = np.blackman(NF)

print("bin width at 10 Msps:", 10e6 / NF / 1e3, "kHz")
print("\nprofile of the bins around 0 Hz, dB above the median bin")
print("bin offset:      -4    -3    -2    -1     0    +1    +2    +3    +4   width")
for name in ["f2437_g70", "f2400_g10", "f2480_g70", "f2480_g40", "f1000_g70"]:
    iq = np.load(IQ / f"{name}.npy")
    n = len(iq) // NF
    fr = iq[:n*NF].reshape(n, NF)
    fr = fr - fr.mean(axis=1, keepdims=True)          # per-window mean removed
    mag = np.abs(np.fft.fft(fr * w, axis=1)).max(axis=0) ** 2
    med = np.median(np.delete(mag, range(-3, 4)))
    db = 10*np.log10(mag/med + 1e-30)
    prof = [db[i % NF] for i in range(-4, 5)]
    # how many bins stay above 3 dB, walking out from 0
    width = 1
    for k in range(1, 20):
        if db[k] > 3 or db[-k] > 3: width += 1
        else: break
    print(f"{name:>10}: " + " ".join(f"{v:5.1f}" for v in prof) + f"   {width} bins")

print("\nis the tune frequency a multiple of the 40 MHz reference?")
for f in (1000, 2400, 2437, 2480):
    q = f / 40.0
    print(f"  {f} MHz / 40 MHz = {q:8.3f}  {'exact harmonic' if q == int(q) else 'off the grid'}")
