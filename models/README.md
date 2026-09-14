# The trained models

These are the models of my own runs, and they come with the project so that you can read the classifier without a PlutoSDR, a drone and two days of recording.
Each `.pt` has a `.meta.json` beside it with the geometry, the classes and the arguments of the run.
The two files travel together, because the program reads the geometry from the `.meta.json` and it cannot load the weights alone.

| Model                        | Classes                               | Preset     | ValAcc | Weak val |
|------------------------------|---------------------------------------|------------|--------|----------|
| `trained_model.pt`           | DJI-MINI-3, Radiolink-AT9S-Pro, noise | `balanced` | 98.65% | 97.34%   |
| `trained_model_wifi.pt`      | the same three, and wifi              | `balanced` | 98.68% | 97.41%   |
| `trained_model_best.pt`      | DJI-MINI-3, Radiolink-AT9S-Pro, noise | `best`     | 99.19% | 97.66%   |
| `trained_model_noiseval4.pt` | DJI-MINI-3, Radiolink-AT9S-Pro, noise | `balanced` | 99.24% | 96.67%   |

### Which one to use

`trained_model.pt` is the model of the project.
The program loads it at the start, it owns the held-out result below, and every number of the paper is its number.

`trained_model_wifi.pt` adds a `wifi` class, because a class that the model never met becomes the class that it most resembles, and a room with WiFi and no drone gave a 24% false alarm rate on the three-class model.
Use this one in a room that has WiFi. It has no held-out number, thus its figures are validation figures only.

The other two are one comparison of the presets and not models for use.
They are the `balanced` run and the `best` run of 2026-08-13, before the retrain that made the frozen model.
`best` is 30 epochs, the whole dataset and 135 819 parameters against approximately 60 000, it costs 17.4 minutes against 12, and it does not win: the energy gate and not the cap holds the AT9S Pro at 13 978 training segments under both presets, thus `best` doubled the two classes that needed nothing and gave the class that needs help nothing.

Their ValAcc above the frozen model is not an improvement either.
The same experiment ran twice with the same code, data, seed and arguments and gave 98.60% and 99.24%, thus no difference below approximately 0.6 points means anything on this setup.

### How to load one

The interface loads `trained_model.pt` at the start, from this folder or from the directory of the project.
For another model, put its path in the **Model** field with **Browse…**, then click **Load / Reload Model**.
The sweep continues and there is no restart.

```bash
python tools/evaluate.py models/trained_model.pt
python tools/eval_clip.py models/trained_model_wifi.pt <clip.iq> --expect DJI-MINI-3
```

### What the numbers mean

ValAcc and the weak-signal value are per-segment, over the segments that pass the energy gate, and a validation session chose the model.
Read the weak-signal value, because ValAcc covers the strong recordings only.
The weak-signal value is simulated, because every recording of the dataset holds the same transmit gain.

The held-out result of `trained_model.pt` is the one number that no model choice touched.
Session 4 of each class, 500 captures of each, read one time on 2026-08-14 and never again, and measured at the level of the badge that the user sees:

| True class         | Named right | Clear | Named wrong |
|--------------------|-------------|-------|-------------|
| DJI-MINI-3         | 490         | 10    | 0           |
| Radiolink-AT9S-Pro | 169         | 331   | 0           |
| noise              | -           | 498   | 2           |

The AT9S Pro link is bursty, thus the program calls most of its captures clear, and it gives no capture the name of the other drone.
The 2 of 500 in the last row are a false alarm rate of 0.4%, and that is the rate of that quiet room and not a property of the model.
`results/` holds the full metrics of each run, and `results/heldout.metrics.json` holds this table.

### What these models expect

Every model here was trained at 10 Msps and 8 MHz of bandwidth, on 2.4 GHz, at RX gain 10, with one PlutoSDR and one antenna in my rooms.
The program warns you when your sample rate is not the rate of the model, because one spectrogram row is `sample_rate / n_fft` Hz and the model then reads the wrong frequency for each row.

The signals are the [RFUAV](https://github.com/kitoweeknd/RFUAV) clips of two drones, which a USRP B210 replayed over the air.
Thus, the model names the type of the drone and not one unit, and it knows my radio and my rooms.
On your radio, in your room, record your own captures and train your own model, and use these as the check of the procedure.
