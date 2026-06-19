# RF-Scan

PlutoSDR RF fingerprinting: watch a band, lock onto a signal, and identify which
device is transmitting from its RF fingerprint.

## Programs

- **`terminal_v2.py`** — the live GUI. Scans the band, locks onto signals, classifies
  them, and records training data. Recording has three kinds:
  - **Device** — park on a frequency and save that device's fingerprint.
  - **Noise (band)** — sweep the whole band (devices off).
  - **Noise (freq)** — park on a device's frequency with the device off (a clean
    negative for that spot).

  Everything is saved under `fingerprint_data/<class>/session_N/`.
- **`train_model.py`** — trains the classifier from `fingerprint_data/`. Each folder is
  one class; outputs `trained_model.pt` (+ `.meta.json`), which the GUI loads.
- **`fp_spectrogram.py`** — shared spectrogram + model code, imported by both so
  preprocessing can't drift.

## Workflow

1. **Record** (GUI): for each device pick **Device** and record at its frequency;
   record **Noise** with the devices off. Do each across **≥2 sessions** (move the
   antenna between them) so the held-out-session accuracy is honest.
2. **Train**: `python train_model.py` — add `--quick` for a fast sanity check.
3. **Detect**: launch the GUI; it loads the model and runs the Locking loop.

## Notes

- `fingerprint_data/` and trained models are git-ignored.
- Quality is SNR-dominated. The trainer drops device segments sitting at the noise floor
  (silent gaps), so record devices while they're actually transmitting.
- Stock PlutoSDR caps ~20 MHz bandwidth and 3.8 GHz; 5.8 GHz / 40 MHz need the AD9364
  firmware hack.
