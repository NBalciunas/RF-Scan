# RF-Scan

PlutoSDR-based **RF fingerprinting**: monitor a band, lock onto signals, and
identify which device is transmitting from its RF fingerprint.

## Programs

### `terminal-ml-v2.py`
The live GUI (PlutoSDR wideband monitor + fingerprint detection). Three modes:

- **Normal** — auto loop: scan → focus a new signal → classify → remember → move on.
- **Wideband** — continuous full-band scan only (optionally records the *noise* class).
- **Focus** — lock one frequency, zoom in, and record a device's fingerprints.

Wideband spectrum + waterfall on the left, the held-signal zoom + waterfall on the
right, controls on the far right.

### `train_fingerprint_spec.py`
Folder-driven trainer. Every top-level folder under `fingerprint_data/` is one class
(`noise`, `deviceA`, `deviceB`, …); `session_*` subfolders drive a **session-held-out**
split (train on one session, validate on another — the only honest number). Produces
`fingerprint_spec_model.pt` (+ `.meta.json`), which the GUI loads.

### `fp_spectrogram.py`
Shared front-end imported by both of the above so preprocessing can't drift: 256-point
STFT spectrogram + the compact `SpecCNN` model + the `FingerprintModel` inference wrapper
(with the `unknown` confidence threshold). Approach follows RFUAV (arXiv:2503.09033),
right-sized for a small device count.

## Workflow

1. **Record** — in the GUI, **Focus** mode + Record saves a device's fingerprints to
   `fingerprint_data/<device>/session_N/`; **Wideband** + Record saves the noise class.
   Record each device across **≥2 separate sessions** (move the antenna between them).
2. **Train** — `python train_fingerprint_spec.py` → `fingerprint_spec_model.pt`.
   Watch the cross-session validation accuracy, not the same-session number.
3. **Detect** — the GUI loads the model on launch and runs the Normal loop.

## Notes

- Data (`fingerprint_data/`) and trained models are git-ignored.
- Stock PlutoSDR caps ~20 MHz bandwidth and 3.8 GHz; DJI O3 (40 MHz) and 5.8 GHz need
  the AD9364 firmware hack. Fingerprint quality is SNR-dominated.
