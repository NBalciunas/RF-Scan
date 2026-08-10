import sys, numpy as np
import _paths  # noqa: F401
from _support import stub_hardware; stub_hardware()
from fp_spectrogram import SEG_LEN, N_FFT, remove_dc

def art(n, strength, win, seed=7):
    r = np.random.RandomState(seed)
    w = (r.randn(n+win) + 1j*r.randn(n+win))/np.sqrt(2)
    cs = np.cumsum(np.concatenate([[0j], w]))
    ma = (cs[win:] - cs[:-win])/win
    return (ma[:n]*strength*np.sqrt(win)).astype(np.complex64)

def dcbin(x):
    fr = np.asarray(x).reshape(-1, N_FFT)*np.hanning(N_FFT)
    p = np.abs(np.fft.fft(fr, axis=1)).mean(axis=0)**2
    return 10*np.log10(p[0]/np.median(p[1:]) + 1e-30)

print("target from the radio: |mean|/rms -24.6 dB, raw 0Hz +10.5 dB, mean gains 1.1 dB")
print(f"{'win':>5} {'str':>5} {'|mean|/rms':>11} {'raw':>7} {'aft mean':>9} {'gain':>6} {'aft hp':>7}")
r = np.random.RandomState(3)
noise = ((r.randn(SEG_LEN)+1j*r.randn(SEG_LEN))*0.1/np.sqrt(2)).astype(np.complex64)
for win in (16, 32, 64, 128):
    for strength in (0.05, 0.1, 0.2):
        a = art(SEG_LEN, strength, win)
        iq = (noise + a).astype(np.complex64)
        rms = np.sqrt(np.mean(np.abs(iq)**2))
        m = 20*np.log10(abs(complex(iq.mean()))/rms + 1e-30)
        raw, aft, hp = dcbin(iq), dcbin(iq - iq.mean()), dcbin(remove_dc(iq))
        print(f"{win:5d} {strength:5.2f} {m:11.1f} {raw:7.2f} {aft:9.2f} {raw-aft:6.2f} {hp:7.2f}")
