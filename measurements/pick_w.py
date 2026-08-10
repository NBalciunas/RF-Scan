"""Choose the width of the high pass from the real captures."""
import sys, json, numpy as np
from pathlib import Path
import _paths  # noqa: F401
from _support import stub_hardware; stub_hardware(force=("pyqtgraph","PyQt5"))
from fp_spectrogram import N_FFT, SEG_LEN, SEG_HOP

HAN = np.hanning(N_FFT).astype(np.float32)

def hp(x, w):
    """Subtract a moving average of w samples. w=0 means the mean of the whole."""
    x = np.asarray(x, dtype=np.complex64)
    if w == 0:
        return x - x.mean()
    if w >= len(x):
        return x - x.mean()
    pad = np.concatenate([np.full(w//2, x[0]), x, np.full(w - w//2 - 1, x[-1])])
    c = np.cumsum(np.concatenate([[0], pad]))
    ma = (c[w:] - c[:-w]) / w
    return (x - ma[:len(x)]).astype(np.complex64)

def dc_sigma(seg):
    n = (len(seg) - N_FFT)//64 + 1
    j = np.arange(N_FFT)[None,:] + 64*np.arange(n)[:,None]
    S = np.fft.fftshift(np.fft.fft(seg[j]*HAN, axis=1), axes=1)
    m = 20*np.log10(np.abs(S)/N_FFT + 1e-10)
    img = m.T.astype(np.float32); img = (img - img.mean())/(img.std()+1e-6)
    return float(img[N_FFT//2].mean() - np.median(img))

IQ = Path(sys.argv[1] if len(sys.argv) > 1 else "iq"); meta = json.loads((IQ/"index.json").read_text())
WS = [0, 32, 64, 128, 256, 512]
print("DC row height in sigma, averaged over 24 segments of each capture")
print(f"{'capture':>12} " + " ".join(f"{('mean' if w==0 else 'w='+str(w)):>7}" for w in WS))
tot = {w: [] for w in WS}
for c in meta["captures"]:
    iq = np.load(IQ/c["file"]); segs = [iq[i*SEG_HOP:i*SEG_HOP+SEG_LEN] for i in range(0,24)]
    row = []
    for w in WS:
        v = float(np.mean([dc_sigma(hp(s, w)) for s in segs])); row.append(v); tot[w].append(v)
    name = c["file"].replace(".npy","")
    if c["gain_db"] in (10, 40, 70):
        print(f"{name:>12} " + " ".join(f"{v:7.3f}" for v in row))
print("\nmean over all 24 captures")
print(f"{'':>12} " + " ".join(f"{np.mean(tot[w]):7.3f}" for w in WS))

# what the high pass costs a real signal
print("\nattenuation of a tone, dB, by its offset from 0 Hz")
n = SEG_LEN; noise = (np.random.default_rng(0).normal(size=n) + 1j*np.random.default_rng(1).normal(size=n)).astype(np.complex64)
print(f"{'offset kHz':>11} " + " ".join(f"{('mean' if w==0 else 'w='+str(w)):>7}" for w in WS))
for khz in (0, 10, 20, 39, 78, 156, 500):
    t = np.exp(2j*np.pi*(khz*1e3/10e6)*np.arange(n)).astype(np.complex64)
    row = []
    for w in WS:
        y = hp(t, w)
        row.append(20*np.log10(np.abs(y).mean()/np.abs(t).mean() + 1e-12))
    print(f"{khz:11d} " + " ".join(f"{v:7.2f}" for v in row))
