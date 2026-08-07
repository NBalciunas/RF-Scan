"""Self-check for the shared spectrogram front end. Type `python tests/test_spectrogram.py`.

fp_spectrogram.py is the one module that both programs use to make an image from IQ
data. If it changes, the recordings and the live detection change together. These
checks hold the geometry, the normalization and the frequency axis of that image.
"""

import numpy as np

from _support import Checks, run
from fp_spectrogram import (iq_to_spectrogram, iq_segments_to_specs,
                            N_FFT, STFT_HOP, SEG_LEN, SEG_HOP)


def _noise(n, sigma=1.0, seed=0):
    r = np.random.RandomState(seed)
    return ((r.randn(n) + 1j * r.randn(n)) * (sigma / np.sqrt(2))).astype(np.complex64)


def _tone(n, f_norm, amp=1.0):
    return (amp * np.exp(2j * np.pi * f_norm * np.arange(n))).astype(np.complex64)


def main():
    c = Checks("Spectrogram front end (fp_spectrogram.py)")
    frames = (SEG_LEN - N_FFT) // STFT_HOP + 1

    @c.check(f"one segment gives (1, {N_FFT}, {frames})")
    def _():
        s = iq_to_spectrogram(_noise(SEG_LEN))
        assert s.shape == (1, N_FFT, frames), s.shape
        assert s.dtype == np.float32, s.dtype

    @c.check("the image is standardized: mean 0, standard deviation 1")
    def _():
        s = iq_to_spectrogram(_noise(SEG_LEN))
        assert abs(float(s.mean())) < 1e-4, float(s.mean())
        assert abs(float(s.std()) - 1.0) < 1e-3, float(s.std())

    @c.check("the image does not change with the gain (this is why the AGC is off)")
    def _():
        iq = _noise(SEG_LEN) + _tone(SEG_LEN, 0.1, 3.0)
        a = iq_to_spectrogram(iq)
        b = iq_to_spectrogram((iq * 1000.0).astype(np.complex64))
        worst = float(np.abs(a - b).max())
        assert worst < 0.01, f"a gain of 60 dB moved the image by {worst:.4f} sigma"

    @c.check("the same input always gives the same image")
    def _():
        iq = _noise(SEG_LEN)
        assert np.array_equal(iq_to_spectrogram(iq), iq_to_spectrogram(iq))

    @c.check("a tone lands on the row that the frequency axis predicts")
    def _():
        # After fftshift, row n_fft//2 is DC. A tone at f cycles for each sample is
        # at row n_fft//2 + f*n_fft. A tone at f = 0 is not tested, because the DC
        # blocker removes it on purpose.
        for f in (0.1, -0.25, 0.35):
            s = iq_to_spectrogram(_tone(SEG_LEN, f))
            row = int(s[0].mean(axis=1).argmax())
            want = N_FFT // 2 + int(round(f * N_FFT))
            assert abs(row - want) <= 1, f"tone {f}: row {row}, expected {want}"

    @c.check("a stationary signal stays on one row in every time frame")
    def _():
        # Only the strong part of the image is compared. The deep nulls between the
        # sidelobes move many dB for a change of one bit in float32, thus they are
        # not a measure of stationarity.
        s = iq_to_spectrogram(_tone(SEG_LEN, 0.1))[0]
        rows = s.argmax(axis=0)
        assert len(set(rows.tolist())) == 1, f"the peak moved between rows {set(rows)}"
        top = float(s.max(axis=0).std())
        assert top < 0.01, f"the peak level moves by {top:.4f} sigma"

    @c.check("a buffer shorter than the FFT is padded, not refused")
    def _():
        s = iq_to_spectrogram(_noise(100))
        assert s.shape == (1, N_FFT, 1), s.shape

    @c.check("an empty channel gives no NaN and no infinity")
    def _():
        s = iq_to_spectrogram(np.zeros(SEG_LEN, np.complex64))
        assert np.isfinite(s).all()

    @c.check("the segment count follows (len - seg_len) // seg_hop + 1")
    def _():
        for n, want in ((SEG_LEN, 1), (SEG_LEN + SEG_HOP, 2), (10240, 4)):
            k = len(iq_segments_to_specs(_noise(n)))
            assert k == want, f"{n} samples gave {k} segments, expected {want}"

    @c.check("max_segs keeps the first and the last segment, thus it covers the buffer")
    def _():
        iq = np.concatenate([_tone(SEG_LEN, 0.2, 5.0),          # a marker at the start
                             _noise(SEG_LEN * 6),
                             _tone(SEG_LEN, -0.2, 5.0)])        # a marker at the end
        specs = iq_segments_to_specs(iq.astype(np.complex64), max_segs=3)
        assert len(specs) == 3, len(specs)
        first_row = int(specs[0, 0].mean(axis=1).argmax())
        last_row  = int(specs[-1, 0].mean(axis=1).argmax())
        assert abs(first_row - (N_FFT // 2 + 51)) <= 2, first_row
        assert abs(last_row - (N_FFT // 2 - 51)) <= 2, last_row

    @c.check("a buffer shorter than one segment still gives one image")
    def _():
        s = iq_segments_to_specs(_noise(1000))
        assert s.shape[:2] == (1, 1), s.shape

    @c.check("the live inference cap gives at most that many segments")
    def _():
        s = iq_segments_to_specs(_noise(500_000), max_segs=24)
        assert len(s) <= 24, len(s)

    @c.check("iq_to_spectrogram honours its own n_fft argument")
    def _():
        # The window is built from the argument, thus a model whose meta records
        # another n_fft still runs.
        for n_fft, hop in ((128, 32), (256, 64), (512, 128)):
            s = iq_to_spectrogram(_noise(SEG_LEN), n_fft=n_fft, hop=hop)
            assert s.shape[1] == n_fft, s.shape
            assert s.shape[2] == (SEG_LEN - n_fft) // hop + 1, s.shape

    @c.check("the receiver DC artifact does not dominate the image")
    def _():
        # The LO leakage lands on the middle row of every capture of every class.
        # _freq_shift then moves it for the device class only, which is a difference
        # that has nothing to do with a drone.
        iq = (_noise(SEG_LEN, 0.1) + 10.0).astype(np.complex64)   # noise plus DC
        rows = iq_to_spectrogram(iq)[0].mean(axis=1)
        excess = float(rows[N_FFT // 2] - np.median(rows))
        c.note(f"a DC offset 100x the noise leaves {excess:+.2f} sigma on the DC row")
        assert excess < 1.0, f"the DC row is {excess:.1f} sigma above the rest"

    @c.check("the DC blocker keeps a real signal that sits next to DC")
    def _():
        # Only the 0 Hz component goes. A modulated signal near the LO must stay.
        iq = (_noise(SEG_LEN, 0.1) + _tone(SEG_LEN, 0.02, 5.0) + 10.0
              ).astype(np.complex64)
        rows = iq_to_spectrogram(iq)[0].mean(axis=1)
        assert int(rows.argmax()) == N_FFT // 2 + 5, int(rows.argmax())

    return c.report()


if __name__ == "__main__":
    run(main)
