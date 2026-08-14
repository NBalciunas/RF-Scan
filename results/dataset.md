# The dataset, as `tools/dataset_info.py` reports it

Phase 6 item 4. Generated on 2026-08-14 from the tree of that day. `NOTES.md` is
gitignored, thus the table that the report needs lives here instead.

Reproduce with:

```
python tools/dataset_info.py
python tools/dataset_info.py --data_dir ./heldout_data
```

Every capture: 10 Msps, 8 MHz receiver bandwidth, RX gain 10, complex64, 50 ms.
Segments are counted at `seg_len` 4096 and `seg_hop` 2048.

### Train and validate: `fingerprint_data/`

| class | session | files | segments | seconds | size | gain |
| --- | --- | --- | --- | --- | --- | --- |
| DJI-MINI-3 | session_1 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| DJI-MINI-3 | session_2 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| DJI-MINI-3 | session_3 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| DJI-MINI-3 | total | 1,500 | 364,500 | 75.0 | 6.00 GB | |
| noise | session_1 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| noise | session_2 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| noise | session_3 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| noise | total | 1,500 | 364,500 | 75.0 | 6.00 GB | |
| Radiolink-AT9S-Pro | session_1 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| Radiolink-AT9S-Pro | session_2 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| Radiolink-AT9S-Pro | session_3 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| Radiolink-AT9S-Pro | total | 1,500 | 364,500 | 75.0 | 6.00 GB | |
| all | | 4,500 | 1,093,500 | 225.0 | 18.00 GB | |

Sessions 1 and 2 of each class train, session 3 validates. The trainer holds out the
last session by natural sort and does it by itself.

`dataset_info.py` gives no warning on this tree.

### Held out, read once: `heldout_data/`

| class | session | files | segments | seconds | size | gain |
| --- | --- | --- | --- | --- | --- | --- |
| DJI-MINI-3 | session_4 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| noise | session_4 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| Radiolink-AT9S-Pro | session_4 | 500 | 121,500 | 25.0 | 2.00 GB | 10 |
| all | | 1,500 | 364,500 | 75.0 | 6.00 GB | |

The tool prints three warnings on this tree, one for each class, that the class has
one session and the trainer would fall back to a random split. That is correct and it
is not a fault: this tree is never trained on. It is read by `tools/evaluate.py`
alone, and it was read once, on 2026-08-14. The result is `results/heldout.metrics.json`.

### The whole campaign

6,000 captures, 24.00 GB, 300.0 s of air, three classes, four sessions of each, one
TX gain. Recorded over the air, B210 at TX gain 70 dB and 2440 MHz to a Pluto at RX
gain 10, basic 2.4 GHz antennas 10 to 15 cm apart. The level that arrives was never
measured, thus no capture carries an SNR.

### The centre frequencies, and the one exception

| class | centre frequencies of the captures (MHz) |
| --- | --- |
| DJI-MINI-3 | 2440.000 |
| Radiolink-AT9S-Pro | 2440.000 |
| noise, `heldout_data/` | 2440.000 |
| noise, `fingerprint_data/` | 2432.800, 2438.400, 2440.000, 2444.000, 2449.600 |

`noise` session 1 is a band record and not a device record: 125 captures at each of
2432.8, 2438.4, 2444.0 and 2449.6 MHz, and none at 2440. It is a different record kind
from the other 4,500 files. Nothing in a 10 MHz image states its absolute frequency,
thus the variety helps the `noise` class rather than teaching it a place in the band.
The risk is in the split and not in the data: hold out session 1 and the `noise`
validation set becomes the band record alone, at a geometry that no drone capture
uses. The trainer holds out the last session, thus this never happens. See §9 Phase 3
item 10 of `NOTES.md`.

### Which source clip fed which session

Measured by envelope cross correlation on 2026-08-13, not from memory. Eight sessions
and eight different clips, thus the session split is honest and the held-out set is
independent of everything the model trains on.

| class | session 1 | session 2 | session 3 | session 4, held out |
| --- | --- | --- | --- | --- |
| DJI-MINI-3 | `dji_bw10_0-1s` | `dji_bw10_2-3s` | `dji_bw10_6-7s` | `dji_bw10_3-4s` |
| Radiolink-AT9S-Pro | `at9s_0-1s` | `at9s_1-2s` | `at9s_7-8s` | `at9s_6-7s` |
