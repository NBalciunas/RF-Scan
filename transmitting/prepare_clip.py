"""Prepare one RFUAV clip for replay. Type `python prepare_clip.py --help`.

RFUAV holds raw IQ at 100 Msps across a 100 MHz band. A USRP B210 gives 61.44 Msps,
and this program receives 10 Msps. Thus a clip can not go to the transmitter as it is.

A file source in GNU Radio does not know the rate of its file. It sends the samples at
the rate that the sink takes. Thus a clip of 100 Msps that goes out at 10 Msps is 10
times too slow in time and 10 times too narrow in frequency, and it is not the drone
any more. Nothing in GNU Radio reports this.

This program takes one slice of the source band, moves that slice to the baseband, and
writes a file at the rate that the replay uses. It also puts the clip at a chosen
level, because a difference of level between two classes becomes a class cue, and it
makes the TX gain useless as a control of the signal-to-noise ratio.

The program writes a `.json` beside the output. The file holds every value that the
operation used, thus the extraction can be repeated and it can go in a report.

The output goes to `transmitting/clips/`. The flow graph that replays it is in
`transmitting/gnuradio/`. The directory is that of this program and not the current
directory. Thus the clips of a campaign stay in one place, whatever directory the
command runs from.

Example, for a 10 MHz slice 5 MHz above the middle of the source band:

    python transmitting/prepare_clip.py "pack1_1-2s.iq" --offset-hz 5e6

The name of the source gives the name of the output. Give a second path to choose it.
"""

import os
import sys
import json
import time
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

IN_RATE   = 100_000_000.0   # RFUAV. The .xml of the pack has precedence.
OUT_RATE  = 10_000_000.0    # the rate of terminal.py, thus the rate of the replay
N_TAPS    = 301
PEAK_PCT  = 99.9            # the percentile of |iq| that the level refers to
PEAK_TGT  = 0.5             # where that percentile goes, in the scale of UHD fc32
PEAK_MAX  = 0.95            # the true peak may not pass this. See the note in prepare.
MIN_SIGNAL_DB = 15.0        # the slice must hold a transmitter. Empty gives 4 to 6 dB.
CHUNK_POW = 21              # the FFT of the overlap-save filter is 2 to this power
BAND_BINS = 4096            # the bins of the source-band spectrum that gives the floor

# The directory of this program, and not the current directory. The clips of a
# campaign must stay together, and the command runs from anywhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
CLIPS_DIR   = _SCRIPT_DIR / "clips"


def read_pack_xml(iq_path):
    """Read the SignalHound sidecar of an RFUAV pack, or give an empty dict.

    One .xml describes the whole pack, thus `pack1_1-2s.iq` has `pack1.xml` and not
    `pack1_1-2s.xml`. The program tries the name before the first underscore, then any
    single .xml in the same directory."""
    d = Path(iq_path).parent
    stem = Path(iq_path).stem.split("_")[0]
    cand = [d / f"{stem}.xml"] + sorted(d.glob("*.xml"))
    for p in cand:
        if not p.exists():
            continue
        try:
            root = ET.parse(p).getroot()
        except Exception:
            continue
        got = {c.tag: c.text for c in root}
        return {
            "xml_file"     : p.name,
            "drone"        : got.get("Drone"),
            "device"       : got.get("DeviceType"),
            "sample_rate"  : float(got["SampleRate"]) if got.get("SampleRate") else None,
            "center_freq"  : (float(got["CenterFrequency"])
                              if got.get("CenterFrequency") else None),
            "if_bandwidth" : (float(got["IFBandwidth"])
                              if got.get("IFBandwidth") else None),
        }
    return {}


def lowpass(n_taps, cutoff_norm):
    """Give a low-pass FIR: a sinc with a Blackman window.

    `cutoff_norm` is the cut frequency divided by the sample rate. The sum of the taps
    is 1, thus the filter does not change the level of a signal in the passband."""
    m = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = 2.0 * cutoff_norm * np.sinc(2.0 * cutoff_norm * m) * np.blackman(n_taps)
    return h / h.sum()


def shift_filter_decimate(src, in_rate, out_rate, offset_hz, taps,
                          chunk_pow=CHUNK_POW):
    """Move `offset_hz` to 0 Hz, low-pass, and keep every Nth sample.

    The three operations are one pass over the file. The file is large, thus the read
    is in blocks, but the result must be the same as one operation on the whole file.
    Two things make that true:

      * The phase ramp uses the absolute index of each sample. Thus the ramp does not
        return to 0 at each block.
      * The filter uses overlap-save. Each block carries the last (len(taps) - 1)
        samples of the block before it, and the program keeps only the part of the
        result that has no edge effect.

    The decimation takes the samples whose absolute index divides by the factor. Thus
    a block boundary does not move the phase of the decimation.

    Gives (iq, n_in, band_spec). `band_spec` is the mean spectrum of the whole source
    band across the whole file, in BAND_BINS bins, with the slice at 0 Hz.
    `slice_signal_db` needs it, and it needs the whole file and not a sample of it: a
    control link hops, thus 80 ms of a second says nothing about the rest."""
    decim = int(round(in_rate / out_rate))
    if abs(decim * out_rate - in_rate) > 1.0:
        raise SystemExit(f"the rate {in_rate:.0f} is not a whole multiple of "
                         f"{out_rate:.0f}. Choose another output rate.")
    L    = len(taps)
    nfft = 1 << chunk_pow
    step = nfft - L + 1
    if step <= 0:
        raise SystemExit("the FFT is smaller than the filter. Raise chunk_pow.")
    H     = np.fft.fft(taps, nfft)
    carry = np.zeros(L - 1, dtype=np.complex64)
    ratio = offset_hz / in_rate
    n_in, out = 0, []
    acc, n_acc = np.zeros(BAND_BINS), 0
    with open(src, "rb") as f:
        while True:
            x = np.fromfile(f, dtype=np.complex64, count=step)
            if x.size == 0:
                break
            if offset_hz:
                k = np.arange(n_in, n_in + x.size, dtype=np.float64)
                # The modulo keeps the argument small. Thus the phase of a long file
                # stays correct.
                x = x * np.exp(-2j * np.pi * ((ratio * k) % 1.0)).astype(np.complex64)
            # The band spectrum comes from the new samples only, in whole windows of
            # BAND_BINS. It must not come from the FFT of the block: the last block is
            # padded with zeros, and a file shorter than one block gives no full block
            # at all, thus the reference would be empty and every slice would read
            # 0 dB and be refused.
            nw = x.size // BAND_BINS
            if nw:
                w = x[:nw * BAND_BINS].reshape(nw, BAND_BINS)
                acc += (np.abs(np.fft.fft(w, axis=1)) ** 2).sum(axis=0)
                n_acc += nw
            blk   = np.concatenate((carry, x))
            carry = blk[-(L - 1):].copy()
            if blk.size < nfft:
                blk = np.concatenate((blk, np.zeros(nfft - blk.size, np.complex64)))
            y = np.fft.ifft(np.fft.fft(blk) * H)[L - 1: L - 1 + x.size]
            first = (-n_in) % decim
            out.append(y[first::decim].astype(np.complex64))
            n_in += x.size
    if not out:
        raise SystemExit(f"{src} holds no complex64 sample.")
    return np.concatenate(out), n_in, acc / max(1, n_acc)


def slice_signal_db(band_spec, in_rate, cutoff_hz):
    """Say whether the slice holds a transmitter, in dB above the noise floor.

    `band_spec` is the mean spectrum of the **whole source band**, already moved so
    that the slice sits at 0 Hz. The floor is the median bin of that band, and the
    source is 100 MHz of which the greater part is empty, thus the median is the
    noise of the recording. The answer is the strongest bin of the slice above that
    floor.

    The reference must come from outside the slice. Two earlier versions failed
    because they took it from inside:

      1. The 99.9 percentile of |iq| above its median. That is burstiness, not
         presence. A hopping link gives 39 dB and **a carrier that never stops gives
         0 dB**, thus every DJI clip would have been refused.
      2. The strongest bin of the slice above the median bin **of the slice**. A
         signal that fills the slice is flat inside it, thus a 9 MHz video link in a
         9 MHz slice reads like noise. All nine DJI clips were refused by it.

    Measured with the floor outside: an empty slice 3.6 to 5.7 dB, the AT9S Pro where
    its hops are 29.7 to 33.5 dB, the DJI video link 42.9 to 43.7 dB.

    The whole file must feed this. A control link hops, thus one part of a second
    says nothing about the rest: `pack1_4-5s` reads 2.1 dB over its first 80 ms and
    it holds hops later."""
    n    = len(band_spec)
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / in_rate))
    p    = 10.0 * np.log10(np.fft.fftshift(band_spec) + 1e-30)
    m    = np.abs(freq) <= cutoff_hz
    if not m.any():
        return 0.0
    return float(p[m].max() - np.median(p))


def level_scale(iq, pct, target):
    """Give the factor that puts the `pct` percentile of |iq| at `target`.

    A percentile and not the mean power: a control link sends bursts and it is silent
    between them, thus the mean of a whole clip is the noise floor and not the signal.
    A high percentile of the magnitude sits inside a burst."""
    ref = float(np.percentile(np.abs(iq), pct))
    if ref <= 0.0:
        raise SystemExit("the slice holds no signal. Check --offset-hz.")
    return target / ref


def prepare(src, out, *, in_rate=IN_RATE, out_rate=OUT_RATE, offset_hz=0.0,
            n_taps=N_TAPS, cutoff_hz=None, pct=PEAK_PCT, target=PEAK_TGT,
            chunk_pow=CHUNK_POW, min_signal_db=MIN_SIGNAL_DB, force=False,
            quiet=False):
    """Do the whole operation and write the output and its .json. Give the meta."""
    # The name of the output comes from the name of the source, and two packs of two
    # drones use the same names. Thus a second run can replace the clip of another
    # class in silence. Give a different name, or --force.
    if os.path.exists(out) and not force:
        raise SystemExit(f"{out} is there already. Give another name for the output, "
                         f"or --force to write over it.")
    xml = read_pack_xml(src)
    if xml.get("sample_rate") and abs(xml["sample_rate"] - in_rate) > 1.0:
        print(f"[warn] {xml['xml_file']} says the source is "
              f"{xml['sample_rate']/1e6:.3f} Msps, and the argument says "
              f"{in_rate/1e6:.3f} Msps. The .xml has precedence.")
        in_rate = xml["sample_rate"]
    if cutoff_hz is None:
        # Below half the output rate. The part above it folds back on itself.
        cutoff_hz = 0.45 * out_rate
    if cutoff_hz >= 0.5 * out_rate:
        print(f"[warn] the cut frequency {cutoff_hz/1e6:.3f} MHz is not below half of "
              f"the output rate. The edge of the band folds back.")

    t0   = time.time()
    taps = lowpass(n_taps, cutoff_hz / in_rate)
    iq, n_in, band = shift_filter_decimate(src, in_rate, out_rate, offset_hz, taps,
                                           chunk_pow=chunk_pow)
    raw_pct  = float(np.percentile(np.abs(iq), pct))
    raw_peak = float(np.abs(iq).max())
    raw_rms  = float(np.sqrt((np.abs(iq) ** 2).mean()))
    signal_db = slice_signal_db(band, in_rate, cutoff_hz)
    m = np.abs(iq)
    med = float(np.median(m))
    duty_pct = 100.0 * float((m > 10.0 * med).mean()) if med > 0.0 else 0.0
    # A control link hops. Thus a slice of one second holds many hops, or none, and
    # the program can not see which. With none, the percentile sits in the noise and
    # the scale then raises that noise to the level of a drone. The clip looks
    # correct, it holds no drone, and it carries the label of one.
    if signal_db < min_signal_db:
        raise SystemExit(
            f"{src}\n"
            f"  The slice at {offset_hz/1e6:+.3f} MHz holds no transmitter: "
            f"{signal_db:.1f} dB above the floor of the band.\n"
            f"  An empty slice gives 4 to 6 dB and a drone gives 29 dB or more.\n"
            f"  The level step would raise this noise to the target and write a clip\n"
            f"  that holds no drone under the name of one.\n"
            f"  Choose another --offset-hz, or another second of the pack. Use\n"
            f"  --min-signal-db 0 to write it anyway.")
    scale    = level_scale(iq, pct, target)
    iq      *= np.complex64(scale)
    peak     = float(np.abs(iq).max())
    n_clip   = int((np.abs(iq) > 1.0).sum())
    # The peak, and not the percentile, decides whether the transmitter cuts the
    # signal. The two classes do not have the same distance between them: the AT9S
    # peak sits 0.5 to 1.4 dB above its 99.9 percentile and the DJI peak sits 4.4 dB
    # above, because OFDM has a longer tail. Thus one percentile target gives one
    # class a peak of 0.64 and the other a peak of 1.01.
    #
    # The program stops, and it does not lower this clip alone. A clip that is
    # quieter than the others is the class cue that the whole level step must remove.
    # Lower --peak for EVERY class of the campaign, and prepare them all again.
    if peak > PEAK_MAX:
        raise SystemExit(
            f"{out}\n"
            f"  The peak is {peak:.3f} and the limit is {PEAK_MAX}. "
            f"{n_clip} samples are above 1.0.\n"
            f"  The transmitter would cut them, and it would cut this class only.\n"
            f"  Lower --peak for every class of the campaign, not for this clip.")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    iq.tofile(out)
    meta = {
        "source"          : str(src),
        "source_xml"      : xml or None,
        "source_rate"     : in_rate,
        "source_center"   : xml.get("center_freq"),
        "source_samples"  : int(n_in),
        "slice_offset_hz" : float(offset_hz),
        "slice_center"    : (xml["center_freq"] + offset_hz
                             if xml.get("center_freq") else None),
        "out_rate"        : out_rate,
        "decimation"      : int(round(in_rate / out_rate)),
        "cutoff_hz"       : float(cutoff_hz),
        "n_taps"          : int(n_taps),
        "window"          : "blackman",
        "n_samples"       : int(iq.size),
        "seconds"         : iq.size / out_rate,
        "dtype"           : "complex64",
        "level_pct"       : float(pct),
        "level_target"    : float(target),
        "level_before"    : {"pct": raw_pct, "peak": raw_peak, "rms": raw_rms},
        "signal_db"       : signal_db,
        "duty_pct"        : duty_pct,
        "min_signal_db"   : float(min_signal_db),
        "scale_applied"   : float(scale),
        "level_after"     : {"peak": peak,
                             "rms": float(np.sqrt((np.abs(iq) ** 2).mean()))},
        "samples_clipped" : n_clip,
        "tool"            : "prepare_clip.py",
        "made_at"         : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(Path(out).with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not quiet:
        c = meta["slice_center"]
        print(f"  source        {Path(src).name}  {n_in:,} samples at "
              f"{in_rate/1e6:.3f} Msps")
        if xml:
            print(f"  pack .xml     {xml['xml_file']}  drone {xml.get('drone')}")
        print(f"  slice         {offset_hz/1e6:+.3f} MHz from the middle"
              + (f", thus {c/1e6:.3f} MHz" if c else ""))
        print(f"  filter        {n_taps} taps, cut at {cutoff_hz/1e6:.3f} MHz")
        print(f"  output        {iq.size:,} samples at {out_rate/1e6:.3f} Msps "
              f"({iq.size/out_rate:.3f} s)")
        print(f"  signal        {signal_db:.1f} dB above the floor of the band, duty "
              f"{duty_pct:.3f}%  (an empty slice gives 4 to 6 dB)")
        print(f"  level         p{pct} {raw_pct:.5f} -> {target}, "
              f"scale {scale:.2f}, peak now {peak:.3f}")
        print(f"  time          {time.time()-t0:.1f} s")
        print(f"[ok] {out}")
        print(f"     {Path(out).with_suffix('.json')}")
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Take one slice of an RFUAV clip and make it ready for replay.")
    p.add_argument("src", help="the .iq of the source, complex64")
    p.add_argument("out", nargs="?", default=None,
                   help="the .iq to write, complex64 at the output rate. The default "
                        f"is the name of the source in {CLIPS_DIR.name}/, beside the "
                        "flow graph.")
    p.add_argument("--offset-hz", type=float, default=0.0,
                   help="the middle of the slice, from the middle of the source band")
    p.add_argument("--in-rate",  type=float, default=IN_RATE,
                   help="the rate of the source. The pack .xml has precedence.")
    p.add_argument("--out-rate", type=float, default=OUT_RATE)
    p.add_argument("--cutoff-hz", type=float, default=None,
                   help="the cut frequency of the filter. The default is 0.45 of the "
                        "output rate.")
    p.add_argument("--taps", type=int, default=N_TAPS)
    p.add_argument("--pct",  type=float, default=PEAK_PCT,
                   help="the percentile of |iq| that gives the level")
    p.add_argument("--peak", type=float, default=PEAK_TGT,
                   help="where that percentile goes. Keep it below 1.0.")
    p.add_argument("--min-signal-db", type=float, default=MIN_SIGNAL_DB,
                   help="the strongest bin of the slice must stand this many dB above "
                        "the floor of the source band. An empty slice gives 4 to 6 dB. "
                        "Use 0 to write anything.")
    p.add_argument("--force", action="store_true",
                   help="write over an output that is there already")
    a = p.parse_args(argv)
    if not os.path.exists(a.src):
        raise SystemExit(f"no such file: {a.src}")
    if a.out is None:
        a.out = str(CLIPS_DIR / (Path(a.src).stem + ".iq"))
    prepare(a.src, a.out, in_rate=a.in_rate, out_rate=a.out_rate,
            offset_hz=a.offset_hz, n_taps=a.taps, cutoff_hz=a.cutoff_hz,
            pct=a.pct, target=a.peak, min_signal_db=a.min_signal_db, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
