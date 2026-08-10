import sys, numpy as np
import _paths  # noqa: F401
from fp_spectrogram import remove_dc
src = open("measure_lo.py", encoding="utf-8").read()
ns = {"FFT_BINS": 1024, "np": np}
for fn in ("def dc_height_db", "def offset_db"):
    i = src.index(fn); j = src.index("\n\n\n", i)
    exec(src[i:j], ns)
dc_height_db, offset_db = ns["dc_height_db"], ns["offset_db"]
rng = np.random.default_rng(0)
n = 1024 * 40
noise = (rng.normal(size=n) + 1j*rng.normal(size=n)).astype(np.complex64) / np.sqrt(2)
for k in (0.0, 0.1, 1.0, 10.0):
    x = noise + np.complex64(k)
    print(f"offset {k:5.1f}x  raw DC {dc_height_db(x):8.2f} dB   "
          f"after {dc_height_db(remove_dc(x)):7.2f} dB   |mean|/rms {offset_db(x):7.2f} dB")
t = np.exp(2j*np.pi*0.05*np.arange(n)).astype(np.complex64)
print(f"tone off DC, no offset:  raw DC {dc_height_db(t+noise):.2f} dB (must stay low)")
