"""Self-check for the shared spectrogram front end. Type `python tests/test_spectrogram.py`.

fp_spectrogram.py is the one module that both programs use to make an image from IQ
data. If it changes, the recordings and the live detection change together. These
checks hold the geometry, the normalization and the frequency axis of that image.
"""

import numpy as np

from _support import Checks, run
from fp_spectrogram import (iq_to_spectrogram, iq_segments_to_specs, remove_dc,
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

    # ── The artifact that the radio really makes ──────────────────────────────
    # The two checks above use a constant. The PlutoSDR does not make a constant.
    # Measured on 2026-08-10 with a 50 ohm load: |mean| / rms is -55 to -87 dB, and
    # the 0 Hz bin still stands 9 to 39 dB above the median. The artifact is a narrow
    # band of energy around 0 Hz. A check that uses a constant passes against an
    # artifact that no radio produces, and that is why §8 #1 stayed open while the
    # suite was green.

    # ART_WIN and ART_STRENGTH are calibrated against the radio, not chosen. On a
    # 4096-sample segment of the 24 real captures the artifact gives |mean| / rms
    # -24.6 dB and a 0 Hz bin 10.5 dB above the median, and a mean takes 1.1 dB of
    # that away. The model below gives -24.3 dB, 10.5 dB and 0.1 dB.
    ART_WIN, ART_STRENGTH = 32, 0.05

    def _lo_artifact(n, strength=ART_STRENGTH, win=ART_WIN, seed=7):
        """A model of the artifact of the receiver: noise held near 0 Hz."""
        r = np.random.RandomState(seed)
        w = (r.randn(n + win) + 1j * r.randn(n + win)) / np.sqrt(2)
        cs = np.cumsum(np.concatenate([[0.0 + 0.0j], w]))
        ma = (cs[win:] - cs[:-win]) / win
        return (ma[:n] * strength * np.sqrt(win)).astype(np.complex64)

    def _dc_bin_db(x):
        fr = np.asarray(x).reshape(-1, N_FFT) * np.hanning(N_FFT)
        p = np.abs(np.fft.fft(fr, axis=1)).mean(axis=0) ** 2
        return float(10 * np.log10(p[0] / np.median(p[1:]) + 1e-30))

    @c.check("the model of the artifact matches the radio that was measured")
    def _():
        # If this fails, the model stopped being realistic. Fix the model, not the
        # code, and do not weaken the two checks that follow.
        iq = (_noise(SEG_LEN, 0.1) + _lo_artifact(SEG_LEN)).astype(np.complex64)
        ratio = 20 * np.log10(abs(complex(iq.mean()))
                              / np.sqrt(np.mean(np.abs(iq) ** 2)) + 1e-30)
        raw = _dc_bin_db(iq)
        c.note(f"model: |mean|/rms {ratio:.1f} dB, 0 Hz bin {raw:+.2f} dB "
               f"(the radio gave -24.6 dB and +10.5 dB)")
        assert -28.0 < ratio < -21.0, ratio
        assert 8.0 < raw < 13.0, raw

    @c.check("remove_dc suppresses the real artifact, and a mean does not")
    def _():
        iq = (_noise(SEG_LEN, 0.1) + _lo_artifact(SEG_LEN)).astype(np.complex64)
        raw = _dc_bin_db(iq)
        by_mean = _dc_bin_db(iq - iq.mean())         # what the code did before
        by_hp = _dc_bin_db(remove_dc(iq))            # what it does now
        c.note(f"0 Hz bin: {raw:+.2f} dB raw, {by_mean:+.2f} dB after a mean, "
               f"{by_hp:+.2f} dB after remove_dc")
        assert by_mean > raw - 1.5, "a mean achieves nothing here, and it must not"
        assert by_hp < raw - 6.0, f"remove_dc left {by_hp:.2f} dB"

    @c.check("the 0 Hz row of the image carries nothing, whatever the input")
    def _():
        # iq_to_spectrogram replaces that row. Thus no class can be told by it.
        for name, iq in (
                ("quiet",    _noise(SEG_LEN, 0.1)),
                ("artifact", _noise(SEG_LEN, 0.1) + _lo_artifact(SEG_LEN)),
                ("constant", _noise(SEG_LEN, 0.1) + 10.0),
                ("tone",     _noise(SEG_LEN, 0.1) + _tone(SEG_LEN, 0.02, 5.0))):
            rows = iq_to_spectrogram(np.asarray(iq, np.complex64))[0].mean(axis=1)
            excess = float(rows[N_FFT // 2] - np.median(rows))
            assert abs(excess) < 0.35, f"{name}: {excess:+.2f} sigma"

    return c.report()


if __name__ == "__main__":
    run(main)
