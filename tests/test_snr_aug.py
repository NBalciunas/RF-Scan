"""Self-check for the training augmentation. Type `python tests/test_snr_aug.py`.

_mix_noise puts a strong capture into recorded noise at a given signal-to-noise ratio.
The power of the capture must then be that many dB above the power of the noise, and
the added noise must become the new floor.
_freq_shift moves a segment in frequency. The FFT peak must move the same quantity,
and the total power must not change.

The augmentation is the reason that a model trained on strong captures can identify a
distant drone. If the mathematics here is wrong, the model learns the wrong thing and
nothing reports an error.
"""

import numpy as np

from _support import Checks, run
from train_model import _mix_noise, _freq_shift, SNR_AUG_DB, FREQ_SHIFT_FRAC

N = 4096


def _tone(amp=100.0, f=0.1, n=N):
    return (amp * np.exp(2j * np.pi * f * np.arange(n))).astype(np.complex64)


def _noise_seg(seed, n=N):
    r = np.random.RandomState(seed)
    return ((r.randn(n) + 1j * r.randn(n)) / np.sqrt(2)).astype(np.complex64)


def _power(x):
    return float(np.mean(np.abs(x) ** 2))


def main():
    c = Checks("Training augmentation (train_model.py)")
    tone = _tone()
    pool = [_noise_seg(1)]
    pn = _power(pool[0])

    # ── The SNR mix ───────────────────────────────────────────────────────────

    @c.check("the mix puts the capture at the exact signal-to-noise ratio")
    def _():
        rng = np.random.RandomState(0)
        for snr in (0.0, 6.0, 10.0, 20.0, 30.0):
            mix = _mix_noise(tone, pool, rng, snr, snr)   # lo = hi is an exact target
            ps = _power(mix - pool[0])                    # the tone without the noise
            got = 10.0 * np.log10(ps / pn)
            assert abs(got - snr) < 0.1, f"target {snr} dB, got {got:.2f} dB"

    @c.check("the mix keeps complex64, thus the cache does not double in size")
    def _():
        rng = np.random.RandomState(0)
        assert _mix_noise(tone, pool, rng, 10, 10).dtype == np.complex64

    @c.check("the added noise becomes the new floor")
    def _():
        # This is the reason that the mix is a correct simulation of a weak signal.
        # A capture that is only scaled down keeps its own quiet original floor.
        rng = np.random.RandomState(0)
        mix = _mix_noise(tone, pool, rng, 0.0, 0.0)
        assert abs(_power(mix) / (2.0 * pn) - 1.0) < 0.05, _power(mix) / pn

    @c.check("a segment with no power is given back unchanged")
    def _():
        silent = np.zeros(N, np.complex64)
        rng = np.random.RandomState(0)
        assert _mix_noise(silent, pool, rng, 0, 20) is silent

    @c.check("the mix draws from the whole noise pool, not from one segment")
    def _():
        big = [_noise_seg(s) for s in range(8)]
        rng = np.random.RandomState(0)
        outs = [_mix_noise(tone, big, rng, 0, 0) for _ in range(20)]
        distinct = len({o.tobytes() for o in outs})
        assert distinct > 1, "every draw gave the same noise segment"
        c.note(f"{distinct} different noise segments in 20 draws of a pool of 8")

    @c.check(f"the shipped SNR range {SNR_AUG_DB} covers a weak signal")
    def _():
        assert SNR_AUG_DB[0] <= 0.0 and SNR_AUG_DB[1] >= 10.0, SNR_AUG_DB

    # ── The frequency shift ───────────────────────────────────────────────────

    @c.check("the shift moves the FFT peak by the correct quantity")
    def _():
        for seed in (3, 4, 5):
            nu = np.random.RandomState(seed).uniform(-0.1, 0.1)   # the value used
            shifted = _freq_shift(tone, np.random.RandomState(seed), 0.1)
            pk = int(np.argmax(np.abs(np.fft.fft(shifted))))
            assert abs(pk - round((0.1 + nu) * N)) <= 1, (seed, pk, nu)

    @c.check("the shift keeps complex64 and does not change the total power")
    def _():
        shifted = _freq_shift(tone, np.random.RandomState(3), 0.1)
        assert shifted.dtype == np.complex64
        assert abs(_power(shifted) / _power(tone) - 1.0) < 1e-5

    @c.check("the shift is a true retune: it changes the phase of each sample only")
    def _():
        # One multiplication by a complex ramp keeps the magnitude of every sample.
        # A resample or a filter would not. This is what makes the shift the same
        # operation that the radio does when it tunes to another frequency.
        for seed in (3, 4, 5):
            shifted = _freq_shift(tone, np.random.RandomState(seed), 0.2)
            worst = float(np.abs(np.abs(shifted) - np.abs(tone)).max())
            assert worst < 1e-2, f"the magnitude moved by {worst:.4f}"

    @c.check("a shift of zero changes nothing")
    def _():
        same = _freq_shift(tone, np.random.RandomState(3), 0.0)
        assert np.allclose(same, tone, atol=1e-4)

    @c.check(f"the shipped shift of {FREQ_SHIFT_FRAC} keeps the signal in the band")
    def _():
        assert 0.0 < FREQ_SHIFT_FRAC < 0.5, FREQ_SHIFT_FRAC

    return c.report()


if __name__ == "__main__":
    run(main)
