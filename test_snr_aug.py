"""Self-check for the training augmentations (run: python test_snr_aug.py).

_mix_noise : embeds a known strong tone into unit noise at fixed SNRs and asserts
the scaled signal's power really sits that many dB above the noise's.
_freq_shift: asserts the tone's FFT peak moves by exactly the drawn offset and
that total power is preserved. Fails if either aug's math ever drifts.
"""
import numpy as np

from train_model import _mix_noise, _freq_shift

rng  = np.random.RandomState(0)
tone = (100.0 * np.exp(2j * np.pi * 0.1 * np.arange(4096))).astype(np.complex64)
pool = [((np.random.RandomState(1).randn(4096)
          + 1j * np.random.RandomState(2).randn(4096)) / np.sqrt(2)).astype(np.complex64)]
pn = float(np.mean(np.abs(pool[0]) ** 2))

for snr in (0.0, 10.0, 20.0):
    mix = _mix_noise(tone, pool, rng, snr, snr)     # lo == hi pins the target
    assert mix.dtype == np.complex64
    ps  = float(np.mean(np.abs(mix - pool[0]) ** 2))    # recover the scaled tone
    got = 10.0 * np.log10(ps / pn)
    assert abs(got - snr) < 0.1, f"target {snr} dB, got {got:.2f} dB"

silent = np.zeros(4096, np.complex64)
assert _mix_noise(silent, pool, rng, 0, 20) is silent   # zero-power seg: no-op

# freq shift: peak must land at round((0.1 + nu) * N), power unchanged
sh_rng = np.random.RandomState(3)
nu     = np.random.RandomState(3).uniform(-0.1, 0.1)    # the value _freq_shift draws
shifted = _freq_shift(tone, sh_rng, 0.1)
assert shifted.dtype == np.complex64
pk = int(np.argmax(np.abs(np.fft.fft(shifted))))
assert abs(pk - round((0.1 + nu) * 4096)) <= 1, (pk, nu)
assert abs(np.mean(np.abs(shifted) ** 2) / np.mean(np.abs(tone) ** 2) - 1) < 1e-5

print("aug self-checks OK")
