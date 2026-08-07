# RF Scan

RF Scan is a signal monitor for the PlutoSDR. It finds the devices that transmit in a
radio band, and it gives each device a name.

The program sweeps a band. When a signal goes above the noise floor, the program tunes
to that frequency and holds it. A small neural network then reads the raw IQ data of
the signal and identifies the transmitter. The same program also records the training
data. Thus the record function and the detect function always prepare the data in the
same way.

The first application is the 2.4 GHz band. The program separates a drone control link
or a drone video link from WiFi, from Bluetooth and from the background noise.

## Features

### The signal chain

The program has one signal chain. Each step gives its result to the next step.

| Step | Operation |
|---|---|
| 1. Sweep | The program divides the span into hops. It tunes to each hop, it waits for the settle time, and it reads one buffer. |
| 2. Spectrum | For each hop the program calculates 1024-bin FFTs across the full buffer. It keeps the maximum value of each bin. Thus a short burst stays visible. |
| 3. Composite | The program keeps the central bins of each hop and joins the parts end to end. One linear map gives the frequency of each bin. |
| 4. Peak | The program finds the strongest peak that is not in the memory of the caught signals. A peak above the threshold causes a lock. |
| 5. Lock | The radio holds that one frequency. The frequency does not move during the hold. |
| 6. Segments | The program cuts the held IQ data into segments of 4096 samples, with a step of 2048 samples. |
| 7. Spectrogram | Each segment becomes a 256-point STFT image of the log magnitude. The program normalizes each image with its own mean and standard deviation. Thus the result does not change with the gain. |
| 8. Classifier | A CNN gives a probability for each class and for each segment. The program calculates the mean, and it also counts the votes of the segments. |
| 9. Result | The badge shows the name of the device, or `clear`, or two names. |

The program releases the lock in three conditions: the user clicks **Skip lock**, the
signal stops for 2.5 s, or the Auto mode finds that the signal is noise. Then the
frequency goes into the memory of the caught signals for 30 s. Thus the scanner moves
through all the signals in the band, and it does not hold the strongest signal only.
An entry in the memory expires. Thus the program can find that signal again.

### The monitor

- Four modes: Auto, Locking, Wideband and Narrowband.
- Four live plots: a wideband spectrum with a waterfall, and a narrowband spectrum
  with a waterfall.
- A sweep with a configurable center, span, overlap, dwell time and settle time. The
  plot shows a line at each hop boundary.
- A peak-hold spectrum. Thus a short burst is visible at its true amplitude.
- A lock state machine with a caught list, a time-out, a Skip button and a Jump-to
  button.
- Markers for the middle and the two edges of the narrowband signal. The program
  measures the markers from the spectrum only.
- Manual gain. The program sets the automatic gain control to off, because a constant
  level is necessary for the fingerprints.

### The classifier

- A CNN that identifies the transmitter from the raw IQ data.
- A probability bar for each class.
- A limit on the rate of the classifier. Thus the plots stay fast.
- A switch that sets the classifier to off.
- A report of two or more transmitters in one capture, through the votes of the
  segments.
- The Auto mode releases a lock that the classifier calls noise.
- The program loads a new model without a restart.

### The recorder

- Three kinds of record: a device, the noise of the full band, and the noise at the
  frequency of the device.
- A JSON metadata file next to each capture.
- A limit on the number of files.
- The Record button starts the mode that gives the correct data.

### The trainer

- Three presets: fast, balanced and best.
- A split that holds one full session out of the training data.
- An energy gate that removes the quiet segments from the device classes.
- Three augmentations: a signal-to-noise mix, a frequency shift and a spectrogram
  mask.
- A confusion matrix, and the precision, the recall and the F1 value of each class.
- An accuracy value for a weak signal.

## Installation

The program needs Python 3.10 or a later version.

```bash
pip install -r requirements.txt
```

This command installs numpy, torch, pyqtgraph, PyQt5 and pyadi-iio. The package
pyadi-iio needs the libiio library of Analog Devices. Install the PlutoSDR drivers of
your operating system first.

The program connects to `ip:192.168.2.1`. This address is the factory address of the
PlutoSDR. If your radio has a different address, set the environment variable
`PLUTO_URI`.

```bash
set PLUTO_URI=ip:192.168.3.1
```

A USB address also operates. The command `iio_info -s` gives the address of each radio
that is connected.

```bash
set PLUTO_URI=usb:1.5.5
```

## Usage

Start the graphical interface from the directory of the project:

```bash
python terminal_v2.py
```

The program writes the recordings to `./fingerprint_data/`. This path is relative to
the current directory. Thus you must start the program from the directory of the
project.

The panel on the right side has four mode buttons:

| Mode | Function |
|---|---|
| Auto | The program sweeps, it locks each new signal, and it releases the lock automatically if the classifier calls the signal noise. This is the default mode. |
| Locking | The same operation, but the lock continues until you click **Skip lock**. |
| Wideband | The program sweeps the band only. There is no lock. |
| Narrowband | The radio holds one frequency only. Use this mode for a device record. |

The button **⟳ Apply Settings** sends the values of the SDR panel to the radio and
starts the sweep again. A change of the mode or of the settings stops a record.

## Viewing the program

The window shows four plots.

- **Wideband Spectrum** - the composite of the full sweep. The dashed grey lines show
  the hop boundaries. During a lock, the program puts the live narrowband spectrum
  into the composite. Thus the full band stays visible.
- **Wideband Waterfall** - the last 200 sweeps. The time axis goes up.
- **Narrowband Spectrum** - the spectrum of the held frequency. The two check boxes in
  the **Narrowband Markers** section put a red line at the middle of the signal, and
  two dashed red lines at the edges. The program calculates the edges at the point
  where the signal goes 3 dB above the noise floor. The middle is the center of the
  power between the two edges, and not the highest bin. Thus the marker is stable.
- **Narrowband Waterfall** - the last 200 held captures.

The section **Waterfall Scale (dBFS)** sets the minimum and the maximum of the color
scale of both waterfalls.

The **Status** section shows the model, the last result of the classifier, the mode,
the caught frequencies, the number of hops and the time of one sweep.

## Recording

The classifier learns from your recordings only. The section **Recording** has three
kinds of record. Each kind writes to a different class folder.

| Kind | Mode | Folder | Condition |
|---|---|---|---|
| Device | Narrowband | `<device label>/` | The device transmits. |
| Noise (band) | Wideband | `noise/` | All the devices are off. |
| Noise (freq) | Narrowband | `noise/` | The radio is on the frequency of the device. The device is off. |

The kind Noise (freq) is the most important kind. Without this data the classifier
learns that all the energy at that frequency is a drone. It does not learn the
fingerprint of the drone.

Set the **Device label**, the **Session** number and the **Focus freq**. Then click
**● Record**. The program starts the correct mode automatically. The button **Lock
freq** copies the frequency of the last lock into the Focus freq field.

The files go to this structure:

```
fingerprint_data/
  deviceA/
    session_1/
      deviceA_s1_1712345678901.iq      raw complex64, little-endian
      deviceA_s1_1712345678901.json    the metadata
  noise/
    session_1/
```

The name of the top-level folder is the name of the class. The trainer reads the name
of the folder only. It does not read the JSON file.

### The full procedure for the data collection

1. Set all the devices to off. Record in the Noise (band) mode.
2. Tune to the frequency of the drone. Keep the drone off. Record in the Noise (freq)
   mode.
3. Set the drone to on. Make sure that the drone transmits. Record in the Device mode.
4. Do the steps 1 to 3 again with a new session number. Move the antenna between the
   sessions. Two sessions are the minimum. Three sessions are better.

The step 4 is necessary. The trainer holds one full session out of the training data.
With one session only, the accuracy value has no meaning.

## Training ML

```bash
python train_model.py --preset balanced --out trained_model.pt
```

Each top-level folder in `fingerprint_data/` is one class. The trainer converts the
captures into spectrograms, it trains the network, and it writes two files: the
weights in `trained_model.pt` and the geometry and the class names in
`trained_model.meta.json`.

### The model

The network is `SpecCNN`, a small 2-D CNN. It has four blocks. Each block has a 3x3
convolution, a batch normalization, a GELU activation and a max pool. The number of
channels is 16, 32, 64, 64 for the presets fast and balanced. Then an adaptive average
pool, a dropout and one linear layer give the classes. The model has approximately
60 000 parameters, or 135 000 parameters with the preset best. The program trains the
model from the start. There is no pretrained model. The input is one channel: the log
magnitude of a 256-point STFT. The program does not keep the phase.

### The presets

| Preset | Epochs | Files for each class | Segments for each file | Channels |
|---|---|---|---|---|
| `fast` | 5 | 15 | 40 | 16 |
| `balanced` | 8 | 5000 | 150 | 16 |
| `best` | 30 | all | all | 24 |

The preset `fast` is a check of the procedure only. Its accuracy value is not the final
value. A run without the flag `--preset` uses `fast`, because the constant `PRESET` at
the top of the file gives that value.

### What the trainer does with your data

- The energy gate. A parked capture of a device is mostly silence between the bursts.
  These quiet segments have the label of the device, but they are noise. The gate
  removes each non-noise segment that is not 3 dB above the noise floor. The program
  calculates the floor from the train sessions of the noise class.
- The signal-to-noise augmentation. Your recordings are strong, because the antenna is
  near. A live signal is weak. The trainer puts one half of the train device segments
  into real recorded noise, at 0 to 20 dB.
- The frequency shift. Your recordings always put the device at the same spectrogram
  rows. Thus the network can learn the position and not the fingerprint. A random shift
  of ±10% of the sample rate prevents this.
- The spectrogram mask. The trainer hides a frequency band and a time stripe of each
  batch. Thus the network learns from a part of the evidence. A WiFi burst above your
  drone does not stop the identification.

### The other flags

Use `--epochs`, `--base_ch`, `--max_segs_per_class`, `--gate_margin_db`, `--snr_aug_p`
and `--freq_shift_frac` to change one value of the preset. A flag has precedence over
the preset. Use `--cpu` for the CPU, and `--store_dtype float32` for more precision in
the cache. The command `python train_model.py --help` gives the full list.

Give the flag `--out trained_model.pt` for each run. The default value of `--out` is a
different name, and the graphical interface does not find that file automatically.

## Loading ML

The graphical interface loads `trained_model.pt` from the directory of the script at
the start. The file `trained_model.meta.json` must be in the same directory, with the
same name. The model reads the geometry from that file. Thus an old model continues to
operate after a change of the constants.

To load a different model, click **Browse…**, select the `.pt` file, and click
**⟳ Load / Reload Model**. The program loads the model without a restart of the sweep.
The button **ML Inference: ON** sets the classifier to off. Then the program only shows
the spectrum, and it operates much faster.

The badge above the buttons shows the result:

| Badge | Meaning |
|---|---|
| `clear (82% noise)` | There is no device. |
| `deviceA (93%)` | The named device transmits. The limit is 60%. |
| `unknown device (71%)` | A device transmits, but no single class has a sufficient probability. |
| `deviceA 50% + droneB 40%` | The votes of the segments found two transmitters in one capture. |

The bar of each class shows the mean probability. The Auto mode releases a lock when
the probability of the class `noise` is 75% or more, after a dwell time of 5000 ms.
Both values are adjustable in the **Mode** section. The Auto mode needs a class with
the name `noise`. Without that class the program does not release a lock automatically,
and the panel gives a warning at the load of the model.

## Example

The images are placeholders. Replace them with screen captures of your system.

### The main window during a sweep

![The main window during a sweep](docs/img/main-window.png)

### A lock on a drone control link, with the markers of the signal

![A lock with the markers](docs/img/lock-markers.png)

### The recording panel during a device record

![The recording panel](docs/img/recording.png)

### The output of the trainer with the confusion matrix

![The output of the trainer](docs/img/training-output.png)

## Validation

### The self-checks

The project has three self-checks. They do not need an SDR, but numpy and torch must be
installed. Each check gives an exception if it fails.

```bash
python fp_spectrogram.py
```

This check gives the function `segment_vote` a buffer with two transmitters. The
function must find both names, and it must give the correct part for each name. A
buffer with a low probability must give no name.

```bash
python tests/test_geometry.py
```

This check calculates the FFT bin of a tone at 97 frequencies, in each hop. Then it
joins the hops as the sweep does. The linear map of the composite must give the
original frequency again. Thus the check finds a difference between the worker and the
graphical interface. It uses four configurations: the values of the program, a
bandwidth that is equal to the sample rate, a large overlap, and one hop only.

```bash
python tests/test_snr_aug.py
```

This check verifies the mathematics of the augmentation. The function `_mix_noise` must
put the tone at the exact signal-to-noise ratio, with an error of less than 0.1 dB. A
silent segment must not change. The function `_freq_shift` must move the FFT peak by the
correct quantity, and it must not change the total power.

### The validation of the model

The trainer measures the accuracy with a session that it did not train on. It gives
three results:

- ValAcc - the accuracy on the held-out session.
- The confusion matrix, with the precision, the recall and the F1 value of each class.
- Weak-signal val - the same held-out segments, but at 0 to 10 dB. This is the most
  important value. ValAcc covers the strong recordings only. Thus it does not show a
  decrease of the performance for a distant device.

If a class has one session only, the trainer uses a random split and gives a `[warn]`
message. That accuracy value is not correct, because the adjacent segments of one
recording are almost identical.

## Limitations

### The PlutoSDR

- The maximum bandwidth is 20 MHz. The program cannot see a wider signal in one hop.
  A sweep covers a wider span, but the hops are not simultaneous. Thus the program can
  miss a burst in a different hop.
- The maximum frequency is 3.8 GHz. For the 5.8 GHz band you must install the AD9364
  firmware modification.
- The radio has one receiver. The program cannot listen to two frequencies at the same
  time.
- The program does not verify the values that you type. The radio does not report an
  error for a span that it cannot receive.

### The detection

- The classifier knows your classes only. An unknown transmitter becomes `unknown
  device`, or the class that is the most similar.
- The program does not keep the phase. Thus it cannot separate two identical units of
  the same model.
- The votes of the segments separate two transmitters in time only. Two signals in one
  segment stay together.
- A weak signal below the peak threshold does not cause a lock. The threshold is 10 dB
  above the noise floor.
- The quality of the model depends on your recordings. Without a Noise (freq) record
  the model learns the frequency and not the fingerprint.

### The program

- There is no command-line interface for the monitor. It is a graphical program only.
- The memory of the caught frequencies does not continue after a change of the mode or
  of the settings.
- The counter of the files knows the files of the current sweep only. The old files on
  the disk stay. Thus a long session can use much space.
- Git ignores the folder `fingerprint_data/` and the model files. They are not in the
  repository.

## License

This project is licensed under the MIT License.
Copyright © 2026 Nojus Balčiūnas
