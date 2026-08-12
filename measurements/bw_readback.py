"""Ask what the radio really does with rx_rf_bandwidth, and what the DC row does.

`RX_BW_HZ` moved from 4 to 8 MHz on 2026-08-12, because at 4 MHz in a 10 Msps FFT the
greater part of each spectrogram holds the skirt of the filter and no signal. Nothing
had asked the hardware whether it gives 8 MHz. The AD9361 builds its analog filter from
a table, thus a requested value is not always the value that arrives.

Two parts:

  1. The readback. For each requested bandwidth, the program writes it and reads it
     again. This is a property of the receiver alone, thus the termination does not
     change the answer.

  2. The DC row against the bandwidth. `test_spectrogram.py` holds an artifact that is
     calibrated to this radio: |mean|/rms -24.3 dB and the 0 Hz bin +8.6 dB, measured
     at 4 MHz. If 8 MHz moves those numbers, the calibration of the check is stale.
     This part **does** hold the air, thus state the termination.

The program receives. It does not transmit.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _support import stub_hardware                                    # noqa: E402
if "adi" in stub_hardware():
    sys.exit("adi is not installed.")

import adi                                                            # noqa: E402
from terminal import SAMPLE_RATE, RX_BW_HZ, GAIN                      # noqa: E402
from fp_spectrogram import remove_dc, iq_to_spectrogram, N_FFT        # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in ("antenna", "load"):
    sys.exit("The first argument must be 'antenna' or 'load'. This program can not\n"
             "see the termination, and a wrong one invalidates part 2.")
TERMINATION = sys.argv[1]
URI = sys.argv[2] if len(sys.argv) > 2 else "ip:192.168.2.1"
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else f"bw_readback_{TERMINATION}.json")

WANT = [1e6, 2e6, 4e6, 5e6, 8e6, 10e6, 15e6, 20e6]
FREQ = 2_440_000_000
BUF  = 262_144


def dc_figures(iq):
    """Give the constant part of the buffer and the height of the 0 Hz row.

    The two numbers that `test_spectrogram.py` calibrates its artifact against. The
    row height comes from the image before the 0 Hz row is replaced, thus it says
    what the front end has to remove and not what is left."""
    rms = float(np.sqrt((np.abs(iq) ** 2).mean()))
    const_db = 20.0 * np.log10(abs(complex(iq.mean())) / rms) if rms else float("nan")
    n = len(iq) // N_FFT
    w = iq[:n * N_FFT].reshape(n, N_FFT)
    p = 20.0 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(w, axis=1), axes=1)) + 1e-12)
    p = p.mean(axis=0)
    row_db = float(p[N_FFT // 2] - np.median(p))
    # The same buffer through the real front end. The row must be flat after it.
    spec = iq_to_spectrogram(remove_dc(iq[:4096]), N_FFT, 64)
    img  = spec[0]
    sigma = float((img[N_FFT // 2].mean() - img.mean()) / (img.std() or 1.0))
    return const_db, row_db, sigma


def clean_baseline(seed=0):
    """Give `row_sigma_after` for noise that holds no artifact at all.

    The number is not 0. `iq_to_spectrogram` puts the median row on the 0 Hz row, and
    a median sits below a mean for this distribution, thus the row keeps an offset
    from the mean of the image after the standardisation. A reading from the radio
    means nothing without this baseline beside it: the first run of this program read
    +0.14 sigma and it looked like an artifact, and clean noise gives +0.152."""
    r = np.random.RandomState(seed)
    iq = (r.randn(4096) + 1j * r.randn(4096)).astype(np.complex64)
    img = iq_to_spectrogram(remove_dc(iq), N_FFT, 64)[0]
    return float((img[N_FFT // 2].mean() - img.mean()) / (img.std() or 1.0))


def main():
    sdr = adi.Pluto(URI)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(FREQ)
    try:
        sdr.gain_control_mode_chan0 = "manual"
    except Exception:
        pass
    sdr.rx_hardwaregain_chan0 = int(GAIN)
    sdr.rx_buffer_size = BUF

    print(f"termination {TERMINATION}   uri {URI}   {SAMPLE_RATE/1e6:.3f} Msps   "
          f"gain {GAIN}   {FREQ/1e6:.3f} MHz\n")
    print("  wanted      readback     error    |mean|/rms   0 Hz row   row after")
    rows = []
    for want in WANT:
        sdr.rx_rf_bandwidth = int(want)
        time.sleep(0.15)
        got = int(sdr.rx_rf_bandwidth)
        sdr.rx()                       # one buffer to leave the change behind
        iq = np.asarray(sdr.rx(), dtype=np.complex64)
        const_db, row_db, sigma = dc_figures(iq)
        rows.append({"wanted_hz": float(want), "readback_hz": got,
                     "error_hz": got - float(want),
                     "const_db": const_db, "dc_row_db": row_db,
                     "row_sigma_after": sigma})
        print(f"  {want/1e6:6.3f} MHz  {got/1e6:7.3f} MHz  {(got-want)/1e3:+7.1f} kHz"
              f"  {const_db:9.1f} dB {row_db:9.2f} dB {sigma:+9.3f}")

    base = clean_baseline()
    print(f"\nClean noise with no artifact gives row_sigma_after {base:+.3f}. Read the "
          f"last\ncolumn against that number and not against zero.")

    app = [r for r in rows if r["wanted_hz"] == float(RX_BW_HZ)]
    print(f"\nThe default of terminal.py is {RX_BW_HZ/1e6:.3f} MHz.")
    if app:
        a = app[0]
        print(f"  The radio gives {a['readback_hz']/1e6:.3f} MHz, "
              f"{a['error_hz']/1e3:+.1f} kHz from the request.")
    else:
        print("  It is not in WANT. Add it.")

    OUT.write_text(json.dumps({
        "termination"  : TERMINATION,
        "uri"          : URI,
        "sample_rate"  : SAMPLE_RATE,
        "gain"         : GAIN,
        "center_freq"  : FREQ,
        "buffer"       : BUF,
        "app_rx_bw_hz" : RX_BW_HZ,
        "clean_baseline_row_sigma": base,
        "rows"         : rows,
        "measured_at"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"\n[ok] {OUT}")


if __name__ == "__main__":
    main()
