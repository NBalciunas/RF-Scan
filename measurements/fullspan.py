"""Full-span shape of the noise, so a DC line can be told from a wideband hump."""
import numpy as np
from pathlib import Path
IQ = Path("iq"); NF = 1024; w = np.blackman(NF)

def prof(name):
    iq = np.load(IQ / f"{name}.npy"); n = len(iq)//NF
    fr = iq[:n*NF].reshape(n, NF); fr = fr - fr.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.fftshift(np.fft.fft(fr*w, axis=1), axes=1)).max(axis=0)**2
    return 10*np.log10(mag/np.median(mag) + 1e-30)

edges = np.linspace(-5.0, 5.0, 21)   # MHz
print("dB above the median bin, averaged in 0.5 MHz buckets across the 10 MHz span")
print("       MHz " + " ".join(f"{(edges[i]+edges[i+1])/2:5.1f}" for i in range(20)))
for name in ["f2437_g70","f2400_g10","f2400_g70","f2480_g40","f2480_g70","f1000_g70"]:
    d = prof(name); f = np.linspace(-5, 5, NF, endpoint=False)
    b = [d[(f>=edges[i])&(f<edges[i+1])].mean() for i in range(20)]
    print(f"{name:>10} " + " ".join(f"{v:5.1f}" for v in b))

print("\npeak bin location and height, per capture")
for name in ["f2437_g70","f2400_g10","f2400_g70","f2480_g40","f2480_g70","f1000_g70"]:
    d = prof(name); f = np.linspace(-5, 5, NF, endpoint=False)
    k = int(np.argmax(d))
    print(f"{name:>10}  peak {d[k]:6.1f} dB at {f[k]:+7.3f} MHz   "
          f"DC bin {d[NF//2]:6.1f} dB   bins>10dB: {(d>10).sum()}")
