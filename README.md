# RF-Scan

PlutoSDR-based **RF fingerprinting**: monitor a band, lock onto signals, and
identify which device is transmitting from its RF fingerprint.

## Programs

### `terminal-ml-v2.py`
The live GUI (PlutoSDR wideband monitor + fingerprint detection). Three modes:

- **Locking** — auto loop: scan → lock the strongest new signal → classify → remember → Skip to the next.
- **Wideband** — continuous full-band scan only.
- **Focus** — lock one frequency and zoom into the held signal.

Wideband spectrum + waterfall on the left, the held-signal zoom + waterfall on the
right, controls on the far right.

**Recording** is a single panel with a **Device / Noise (band) / Noise (freq)** choice;
hitting Record routes to the right capture automatically:

- **Device** parks on *Focus freq* and saves the held signal to
  `fingerprint_data/<device>/session_N/`. The hardware fingerprint is largely
  *frequency-independent*, so recording at one spot generalises across the band.
- **Noise (band)** sweeps the whole band into `fingerprint_data/noise/`. Ambient is
  *frequency-dependent* (WiFi/BT/spurs sit at specific spots), so sweeping folds that
  whole variety into the "not-a-device" class.
- **Noise (freq)** parks on *Focus freq* (set it to a device's frequency, device OFF)
  and also saves to `fingerprint_data/noise/` — a frequency-matched negative: exactly
  what the detector sees at that device's spot when the device is silent.

All noise lands in one `noise/` class. The live display differs (narrowband peak-holds,
so it looks cleaner than the single-window wideband plot), but the saved `.iq` is the
full `rx()` buffer either way, so they're equivalent as training data.

### `train_fingerprint_spec.py`
Folder-driven trainer. Every top-level folder under `fingerprint_data/` is one class
(`noise`, `deviceA`, `deviceB`, …); `session_*` subfolders drive a **session-held-out**
split (train on one session, validate on another — the only honest number). Produces
`fingerprint_spec_model.pt` (+ `.meta.json`), which the GUI loads.

Memory/time is tunable — start with `--quick` (a fast "is this going the right way?"
run: few epochs + small caps). The caps `--max_files_per_class`, `--max_segs_per_file`,
`--max_segs_per_class` and `--store_dtype float16` (default) bound RAM; the spectrogram
cache size is printed at startup. `--max_segs_per_class` also keeps a big `noise` class
from dominating RAM and the loss. `--cpu` forces CPU.

### `fp_spectrogram.py`
Shared front-end imported by both of the above so preprocessing can't drift: 256-point
STFT spectrogram + the compact `SpecCNN` model + the `FingerprintModel` inference wrapper
(with the `unknown` confidence threshold). Approach follows RFUAV (arXiv:2503.09033),
right-sized for a small device count.

## Workflow

1. **Record** — in the GUI's Recording panel, choose **Device**, **Noise (band)**, or
   **Noise (freq)** and hit Record (see *Recording* above). Record noise with target
   devices off but the real environment on, so it captures the live WiFi/BT/ambient the
   detector will meet; **Noise (freq)** at each device's frequency gives a clean negative
   for the moments that device is silent. `noise` is a class like any other — record it
   across **≥2 separate sessions** too, same as each device (move the antenna between).
2. **Train** — `python train_fingerprint_spec.py` → `fingerprint_spec_model.pt`
   (try `--quick` first for a fast sanity check). Watch the cross-session validation
   accuracy, not the same-session number.
3. **Detect** — the GUI loads the model on launch and runs the Locking loop.

## Notes

- Data (`fingerprint_data/`) and trained models are git-ignored.
- Stock PlutoSDR caps ~20 MHz bandwidth and 3.8 GHz; DJI O3 (40 MHz) and 5.8 GHz need
  the AD9364 firmware hack. Fingerprint quality is SNR-dominated.
