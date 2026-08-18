"""Self-check for prepare_clip.py. Type `python tests/test_prepare_clip.py`.

The program is the only path between the RFUAV source and the transmitter, and a
mistake in it is silent: the replay works, the waterfall looks correct, and the
frequency scale or the level of the whole dataset is wrong.

The checks make a source at 100 Msps that holds tones at known frequencies, then they
put it through the real program and ask where the tones are.
"""

import os
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from _support import Checks, run
# transmitting/ has no __init__.py and it does not need one. _support puts the repo
# root on sys.path, thus Python finds it as a namespace package.
from transmitting.prepare_clip import (prepare, lowpass, shift_filter_decimate,
                                       level_scale, read_pack_xml, slice_signal_db,
                                       CLIPS_DIR, PEAK_TGT)

IN_RATE, OUT_RATE = 100e6, 10e6


def _tones(n, rate, freqs, amps=None):
    """A buffer of complex64 that holds one tone for each frequency."""
    t = np.arange(n, dtype=np.float64) / rate
    amps = amps or [1.0] * len(freqs)
    x = np.zeros(n, dtype=np.complex128)
    for f, a in zip(freqs, amps):
        x += a * np.exp(2j * np.pi * f * t)
    return x.astype(np.complex64)


def _write(dirpath, name, iq):
    p = Path(dirpath) / name
    iq.tofile(p)
    return str(p)


def _peak_hz(iq, rate):
    """The frequency of the strongest bin of a buffer."""
    n = 1 << 14
    m = min(n, len(iq))
    S = np.abs(np.fft.fftshift(np.fft.fft(iq[:m] * np.hanning(m), n)))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / rate))
    return float(f[int(np.argmax(S))])


def main():
    c = Checks("Clip preparation (prepare_clip.py)")
    tmp = tempfile.mkdtemp(prefix="rfscan_clip_")
    try:
        N = 4_000_000          # 40 ms at 100 Msps, thus several FFT blocks

        @c.check("the low-pass keeps its passband and rejects its stopband")
        def _():
            h = lowpass(301, 4.5e6 / IN_RATE)
            H = np.abs(np.fft.fft(h, 8192))
            f = np.fft.fftfreq(8192, 1.0 / IN_RATE)
            passb = H[np.abs(f) < 3.5e6].min()
            stopb = H[np.abs(f) > 6.5e6].max()
            c.note(f"passband {20*np.log10(passb):+.2f} dB, "
                   f"stopband {20*np.log10(stopb):.1f} dB")
            assert passb > 0.98, passb
            assert stopb < 1e-3, stopb

        @c.check("the sum of the taps is 1, thus the level does not change")
        def _():
            for n in (51, 301, 1001):
                assert abs(lowpass(n, 0.045).sum() - 1.0) < 1e-12

        @c.check("a tone at the middle of the slice comes out at 0 Hz")
        def _():
            src = _write(tmp, "a.iq", _tones(N, IN_RATE, [3e6]))
            iq, _n, _b = shift_filter_decimate(src, IN_RATE, OUT_RATE, 3e6,
                                           lowpass(301, 4.5e6 / IN_RATE))
            got = _peak_hz(iq, OUT_RATE)
            c.note(f"3.000 MHz of the source reads {got/1e6:+.4f} MHz of the slice")
            assert abs(got) < 5e3, got

        @c.check("a tone away from the middle keeps its distance from it")
        def _():
            # The source holds a tone 1.2 MHz above the middle of the slice. The
            # slice moves to the baseband, thus the tone must stay 1.2 MHz above 0.
            src = _write(tmp, "b.iq", _tones(N, IN_RATE, [4.2e6]))
            iq, _n, _b = shift_filter_decimate(src, IN_RATE, OUT_RATE, 3e6,
                                           lowpass(301, 4.5e6 / IN_RATE))
            got = _peak_hz(iq, OUT_RATE)
            assert abs(got - 1.2e6) < 5e3, got

        @c.check("a tone outside the slice does not alias into it")
        def _():
            # 20 MHz above the slice. Without the filter it would fold to 0 Hz,
            # because 20 MHz is a whole multiple of the output rate.
            src = _write(tmp, "c.iq", _tones(N, IN_RATE, [3e6, 23e6], [0.01, 1.0]))
            taps = lowpass(301, 4.5e6 / IN_RATE)
            iq, _n, _b = shift_filter_decimate(src, IN_RATE, OUT_RATE, 3e6, taps)
            got = _peak_hz(iq, OUT_RATE)
            c.note(f"the wanted tone is 40 dB weaker than the one outside, "
                   f"and the peak still reads {got/1e6:+.4f} MHz")
            assert abs(got) < 5e3, got

        @c.check("the phase ramp does not restart at each block")
        def _():
            # A block boundary that resets the ramp gives a phase step. The step
            # spreads the tone, thus the peak falls and the neighbours rise.
            src = _write(tmp, "d.iq", _tones(N, IN_RATE, [4.2e6]))
            taps = lowpass(301, 4.5e6 / IN_RATE)
            big, _n1, _b1 = shift_filter_decimate(src, IN_RATE, OUT_RATE, 3e6, taps,
                                           chunk_pow=22)
            small, _n2, _b2 = shift_filter_decimate(src, IN_RATE, OUT_RATE, 3e6, taps,
                                             chunk_pow=16)
            n = min(len(big), len(small))
            err = np.abs(big[:n] - small[:n]).max() / np.abs(big[:n]).mean()
            c.note(f"the block size 2^22 and 2^16 differ by {err:.2e} of the mean")
            assert err < 1e-4, err

        @c.check("the output rate divides the input rate, or the program stops")
        def _():
            src = _write(tmp, "e.iq", _tones(1000, IN_RATE, [0.0]))
            try:
                shift_filter_decimate(src, IN_RATE, 3e6, 0.0, lowpass(51, 0.01))
            except SystemExit as e:
                assert "whole multiple" in str(e), str(e)
            else:
                raise AssertionError("it accepted a rate that does not divide")

        @c.check("the level scale puts the percentile where it was asked to")
        def _():
            r = np.random.RandomState(0)
            iq = (r.randn(100_000) + 1j * r.randn(100_000)).astype(np.complex64)
            s = level_scale(iq, 99.9, 0.6)
            assert abs(np.percentile(np.abs(iq * s), 99.9) - 0.6) < 1e-4

        @c.check("a silent slice is refused and does not give a division by zero")
        def _():
            try:
                level_scale(np.zeros(1000, np.complex64), 99.9, 0.6)
            except SystemExit as e:
                assert "no signal" in str(e), str(e)
            else:
                raise AssertionError("it scaled a slice that holds nothing")

        @c.check("prepare writes the .iq and the .json, and the level is the target")
        def _():
            src = _write(tmp, "f.iq", _tones(N, IN_RATE, [4.2e6], [0.02]))
            out = str(Path(tmp) / "out" / "f_slice.iq")
            meta = prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE,
                           offset_hz=3e6, quiet=True)
            iq = np.fromfile(out, dtype=np.complex64)
            assert iq.size == meta["n_samples"] == N // 10, (iq.size, N // 10)
            assert abs(np.percentile(np.abs(iq), 99.9) - PEAK_TGT) < 1e-3
            with open(Path(out).with_suffix(".json")) as f:
                j = json.load(f)
            assert j["decimation"] == 10 and j["out_rate"] == OUT_RATE
            assert j["slice_offset_hz"] == 3e6
            assert j["source_samples"] == N
            c.note(f"the source at 0.02 full scale needed a scale of "
                   f"{j['scale_applied']:.1f}")

        def _band(sig=None, width=0):
            """A source band of noise, with an optional signal `width` bins wide at 0."""
            r = np.random.RandomState(0)
            b = np.abs(r.randn(4096) + 1j * r.randn(4096)) ** 2
            if sig:
                half = max(1, width // 2)
                b[:half] *= sig            # the slice sits at 0 Hz, thus at both ends
                b[-half:] *= sig
            return b

        @c.check("an empty slice reads near the floor of the band")
        def _():
            db = slice_signal_db(_band(), IN_RATE, 4.5e6)
            c.note(f"noise across the whole band: {db:.1f} dB")
            assert db < 12.0, db

        @c.check("a narrow signal in the slice stands above the floor")
        def _():
            db = slice_signal_db(_band(sig=10_000, width=40), IN_RATE, 4.5e6)
            c.note(f"a narrow carrier in the slice: {db:.1f} dB")
            assert db > 30.0, db

        @c.check("a signal that fills the whole slice still stands above the floor")
        def _():
            # This is the fault that refused all nine DJI clips. The DJI video link is
            # 9 MHz and the slice is 9 MHz, thus the signal fills it and the spectrum
            # inside the slice is flat. A test that takes its reference from inside
            # the slice reads that as noise. The reference must come from the band
            # outside, which is mostly empty.
            n_bins = int(4096 * 9e6 / IN_RATE)      # the slice is the whole 9 MHz
            db = slice_signal_db(_band(sig=10_000, width=n_bins), IN_RATE, 4.5e6)
            c.note(f"a signal that fills the slice: {db:.1f} dB")
            assert db > 30.0, db

        @c.check("a peak that would be cut is refused, and this clip is not lowered")
        def _():
            # OFDM has a longer tail than a hopping link: the DJI peak sits 4.4 dB
            # above its 99.9 percentile and the AT9S peak 0.5 to 1.4 dB above. Thus
            # one percentile target gave the DJI clips a peak of 1.01 and cut them,
            # and it cut that class only. A lower level for that one clip makes it
            # quieter than the others, which is the cue the level step must remove.
            src = _write(tmp, "hi.iq", _tones(N // 4, IN_RATE, [1e6], [0.02]))
            out = str(Path(tmp) / "hi_slice.iq")
            try:
                prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE, target=0.99,
                        quiet=True)
            except SystemExit as e:
                assert "Lower --peak for every class" in str(e), str(e)
            else:
                raise AssertionError("it wrote a clip that the transmitter would cut")
            assert not os.path.exists(out), "it wrote the file before it refused"

        @c.check("a slice that holds no transmitter is refused, not scaled up")
        def _():
            # The fault this catches: a hopping link puts nothing in the slice for a
            # whole second. The percentile then sits in the noise, the level step
            # raises that noise to the target, and the clip carries the label of a
            # drone. Two of the eight real AT9S Pro clips did exactly this, at 4.9 and
            # 6.8 dB, and were scaled by 489 and 380 to a peak of 0.95.
            r = np.random.RandomState(1)
            n = ((r.randn(N // 4) + 1j * r.randn(N // 4)) * 0.001).astype(np.complex64)
            src = _write(tmp, "quiet.iq", n)
            out = str(Path(tmp) / "quiet_slice.iq")
            try:
                prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE, quiet=True)
            except SystemExit as e:
                assert "holds no transmitter" in str(e), str(e)
            else:
                raise AssertionError("it scaled a slice of pure noise")
            assert not os.path.exists(out), "it wrote the file before it refused"
            # The override must still work, and it must record why.
            meta = prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE,
                           min_signal_db=0.0, quiet=True)
            assert meta["signal_db"] < 10.0, meta["signal_db"]
            assert meta["min_signal_db"] == 0.0

        @c.check("a second run does not write over a clip in silence")
        def _():
            # Two packs of two drones hold files with the same name, thus the
            # default output name can collide. The loss would be silent, and the
            # dataset would hold one drone under the name of another.
            src = _write(tmp, "g.iq", _tones(N // 4, IN_RATE, [1e6], [0.02]))
            out = str(Path(tmp) / "twice.iq")
            prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE, quiet=True)
            try:
                prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE, quiet=True)
            except SystemExit as e:
                assert "there already" in str(e), str(e)
            else:
                raise AssertionError("it wrote over the clip and said nothing")
            prepare(src, out, in_rate=IN_RATE, out_rate=OUT_RATE, force=True,
                    quiet=True)

        @c.check("the .json is enough to repeat the extraction")
        def _():
            # Every value that the operation used must be in the file. Without one of
            # them the report can not say what was done.
            with open(Path(tmp) / "out" / "f_slice.json") as f:
                j = json.load(f)
            for k in ("source", "source_rate", "slice_offset_hz", "out_rate",
                      "decimation", "cutoff_hz", "n_taps", "window", "level_pct",
                      "level_target", "scale_applied", "level_before", "made_at"):
                assert k in j, f"the .json has no {k}"

        @c.check("the pack .xml gives the rate, and it has precedence over the flag")
        def _():
            d = Path(tmp) / "pack"
            d.mkdir()
            (d / "pack9.xml").write_text(
                '<?xml version="1.0"?><SignalHoundIQFile Version="1.0">'
                "<Drone>TestDrone</Drone><DeviceType>USRPX310</DeviceType>"
                "<SampleRate>100000000</SampleRate>"
                "<CenterFrequency>2440000000.000</CenterFrequency>"
                "<IFBandwidth>100000000</IFBandwidth></SignalHoundIQFile>")
            got = read_pack_xml(str(d / "pack9_1-2s.iq"))
            assert got["sample_rate"] == 100e6 and got["drone"] == "TestDrone", got
            assert got["center_freq"] == 2.44e9

            src = _write(d, "pack9_1-2s.iq", _tones(N // 4, IN_RATE, [1e6], [0.02]))
            out = str(d / "o.iq")
            # The flag says 50 Msps and it is wrong. The .xml must win, thus the
            # decimation must be 10 and not 5.
            meta = prepare(src, out, in_rate=50e6, out_rate=OUT_RATE,
                           offset_hz=1e6, quiet=True)
            assert meta["decimation"] == 10, meta["decimation"]
            assert meta["source_center"] == 2.44e9
            assert meta["slice_center"] == 2.441e9, meta["slice_center"]

        @c.check("a clip with no .xml still works and says so")
        def _():
            d = Path(tmp) / "bare"
            d.mkdir()
            src = _write(d, "x.iq", _tones(N // 4, IN_RATE, [0.0], [0.02]))
            meta = prepare(src, str(d / "o.iq"), in_rate=IN_RATE, out_rate=OUT_RATE,
                           quiet=True)
            assert meta["source_xml"] is None
            assert meta["slice_center"] is None
            assert meta["source_rate"] == IN_RATE

        @c.check("the default output goes to clips/, with the flow graph")
        def _():
            # The directory of the program and not the current directory. Thus the
            # command gives the same answer from any place, and the clips of one
            # campaign stay with the flow graph that replays them.
            assert CLIPS_DIR.name == "clips", CLIPS_DIR
            assert (CLIPS_DIR.parent / "gnuradio" / "iqRepeat.grc").exists(), CLIPS_DIR
        return c.report()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run(main)
