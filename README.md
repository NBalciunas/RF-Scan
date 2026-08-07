# RF Scan

RF Scan is a monitor program for the PlutoSDR. It finds the devices that transmit in
a radio band.

The program sweeps a band. When a signal goes above the noise floor, the program
tunes to that frequency and holds it. A classifier reads the raw IQ data of the
signal and gives the signal a name.

The same program records the training data. Thus, the record function and the detect
function use the same data preparation.

To install the software, type this command:

```bash
pip install -r requirements.txt
```

## Programs

**terminal_v2.py** is the graphical interface (GUI). It sweeps, holds, classifies and
records. It writes the raw IQ data to `fingerprint_data/<class>/session_N/`. There
are three record modes:

- **Device** - the radio stays on one frequency. The device transmits.
- **Noise (band)** - the radio sweeps the full band. All the devices are off.
- **Noise (freq)** - the radio stays on the frequency of the device. The device is off.

**train_model.py** trains the classifier with the data in `fingerprint_data/`. Each
top-level folder is one class. The program keeps one full session out of the training
data. Then it measures the accuracy with that session. The program writes a `.pt`
file and a `.meta.json` file. The GUI reads these two files.

**The helper programs** contain the shared code and the self-checks:

- **`fp_spectrogram.py`** - the STFT front end and the SpecCNN model. The GUI and the
  trainer import this code. Thus, the two programs always prepare the data in the
  same way.
- **`tests/test_geometry.py`** - a self-check for the sweep geometry. To do the check,
  type `python tests/test_geometry.py`.
- **`tests/test_snr_aug.py`** - a self-check for the training data augmentation. To do
  the check, type `python tests/test_snr_aug.py`.

## How to train

1. Set all the devices to off. Record in the Noise (band) mode. This data becomes the
   background class.

2. Tune the radio to the frequency of the drone. Keep the drone off. Record in the
   Noise (freq) mode. This data shows the classifier the frequency without the drone.
   The classifier must not learn that all the energy at this frequency is a drone.

3. Set the drone to on. Make sure that the drone transmits. Record in the Device mode.

   Note: Do the steps 1 to 3 in two or more sessions. Move the antenna between the
   sessions. The accuracy data is only correct if the held-out session is new.

4. Train the classifier. Type this command:

   ```bash
   python train_model.py --out trained_model.pt
   ```

   To change the quality, add `--preset fast`, `--preset balanced` or `--preset best`.
   A high quality takes more time. The GUI reads the file `trained_model.pt`
   automatically.

5. Start the GUI again. Click the **⟳ Load / Reload Model** button.

6. Read the result on the badge. `clear` means that there is no device. A name and a
   percentage mean that there is a device. The GUI shows a name when the confidence
   is 60% or more. In the Auto mode, the program releases the hold when the noise
   value is 75% or more.

## Limits

The PlutoSDR has a maximum bandwidth of 20 MHz and a maximum frequency of 3.8 GHz.
For 5.8 GHz, you must install the AD9364 firmware modification.

Git ignores the folder `fingerprint_data/` and the model files.
