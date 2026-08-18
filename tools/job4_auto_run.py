"""Drive Auto against the real radio with no window, and log what it decided.

Receive only. It never records and it never transmits. It exists because the live
path can only be judged from a radio, and a person at a screen can not write down
500 events. See NOTES.md section 9.2 job 4.

  python tools/job4_auto_run.py <model.pt> <seconds> <out.json>

Every sweep writes the strongest bin of the composite and the level in the band the
transmitter occupies. Thus a run where Auto never locks the drone still says whether
the drone was on the air, which is the question a run of 2026-08-18 could not answer.
"""
import os, sys, time, json
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import terminal as T
from PyQt5 import QtCore
from fp_spectrogram import FingerprintModel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "trained_model_wifi.pt"
SECS  = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
OUT   = sys.argv[3] if len(sys.argv) > 3 else "job4_events.json"
TX_LO, TX_HI = 2_435e6, 2_445e6          # the B210 replays a 10 MHz slice at 2440

cfg = {
    "sample_rate": T.SAMPLE_RATE, "rx_bw": T.RX_BW_HZ, "gain": T.GAIN,
    "center_freq": 2_440_000_000, "total_span": T.TOTAL_SPAN_HZ,
    "dwell_ms": T.HOP_DWELL_MS, "settle_ms": T.HOP_SETTLE_MS,
    "overlap_pct": T.HOP_OVERLAP_PCT, "hop_freqs": [],
    "op_mode": "auto", "ml_enabled": True, "ml_interval_s": T.ML_INTERVAL_S,
    "record": False, "record_kind": "device", "skip_lock": False, "jump_to": None,
    "auto_dwell_ms": T.FP_AUTO_DWELL_MS, "auto_noise_pct": T.FP_AUTO_NOISE_PCT,
    "record_device": "deviceA", "record_session": "1", "focus_freq": T.CENTER_FREQ,
    "record_max_files": 1000, "record_every_n": T.RECORD_EVERY_N,
    "fp_peak_thresh_db": T.FP_PEAK_THRESH_DB, "fp_hold_settle_ms": T.FP_HOLD_SETTLE_MS,
    "fp_gone_s": T.FP_GONE_S, "fp_memory_ttl_s": T.FP_MEMORY_TTL_S,
    "fp_memory_guard_hz": T.FP_MEMORY_GUARD_HZ,
}
eff_bw = min(cfg["sample_rate"], cfg["rx_bw"])
cfg["hop_freqs"] = T.compute_hop_freqs(cfg["center_freq"], cfg["total_span"],
                                       eff_bw, cfg["overlap_pct"])
_nk, F0, F1 = T.composite_geometry(cfg)
print("span   : %.1f to %.1f MHz over %d hops" % (F0/1e6, F1/1e6, len(cfg["hop_freqs"])))

import adi
sdr = adi.Pluto(T.SDR_URI)
T.configure_sdr(sdr, cfg)
engine = FingerprintModel(MODEL)
print("model  : %s  %s" % (os.path.basename(MODEL), engine.classes))
print("lock needs %.0f dB over the floor; Auto releases at %d%% background"
      % (T.FP_PEAK_THRESH_DB, T.FP_AUTO_NOISE_PCT))

app = QtCore.QCoreApplication(sys.argv[:1])
w   = T.SweepWorker(sdr, cfg, engine=engine)
t0, ev = time.time(), []
st = {"mode": None, "freq": None, "since": t0, "caught": []}

def log(kind, **kw):
    e = dict(t=round(time.time() - t0, 2), kind=kind, **kw)
    ev.append(e)
    line = "[%6.2f] %-9s %s" % (e["t"], kind,
                                " ".join("%s=%s" % (k, v) for k, v in kw.items()))
    try:
        print(line, flush=True)
    except UnicodeEncodeError:               # the console here is cp1252
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)

def on_status(msg):
    log("status", msg=msg)

def on_caught(freqs):
    st["caught"] = [round(f/1e6, 3) for f in freqs]
    log("caught", freqs_mhz=st["caught"])

def on_fp(res):
    text, style = T.badge_for(res)
    # The share of every device vote, and not the badge alone. A run with no second
    # name on the badge has two possible causes: no second vote happened at all, or
    # FP_SECOND_NAME_SHARE suppressed one. Only the second says the limit of #41 was
    # exercised, and the badge string can not separate the two causes.
    dv = sorted((round(d["share"], 3), d["label"]) for d in T.device_votes(res))
    log("classify", badge=text, style=style, plan=res.get("band_plan") or "-",
        ambient=round(T.SweepWorker._ambient_prob(res), 3),
        dev_share=round(T.device_share(res), 3),
        votes=[[lb, sh] for sh, lb in reversed(dv)],
        second=dv[-2][0] if len(dv) >= 2 else None)

def on_sweep(comp, _bufs):
    """The evidence that a run with no lock still needs: was the drone on the air?"""
    if comp is None:
        return
    c = np.asarray(comp, dtype=float)
    live = c > T.EMPTY_SLOT_DB + 1.0
    if not live.any():
        return
    floor = float(np.median(c[live])) - w._psd_bias_db
    n = len(c)
    b0 = max(0, T.hz_to_bin(TX_LO, F0, F1, n))
    b1 = min(n, T.hz_to_bin(TX_HI, F0, F1, n) + 1)
    tx = c[b0:b1]; tx = tx[tx > T.EMPTY_SLOT_DB + 1.0]
    i  = int(np.argmax(np.where(live, c, -1e9)))
    log("sweep",
        floor_db=round(floor, 1),
        peak_db=round(float(c[i]) - floor, 1),
        peak_mhz=round((F0 + (i + 0.5) * (F1 - F0) / n) / 1e6, 3),
        tx_band_db=round(float(tx.max()) - floor, 1) if tx.size else None,
        caught=len(st["caught"]))

w.status_msg.connect(on_status)
w.fingerprint_ready.connect(on_fp)
w.sweep_ready.connect(on_sweep)
w.caught_changed.connect(on_caught)

QtCore.QTimer.singleShot(int(SECS * 1000), app.quit)
w.start()
print("--- running %.0f s ---" % SECS, flush=True)
app.exec_()
w.stop(); w.wait(5000)
json.dump(ev, open(OUT, "w"), indent=1)
sw = [e for e in ev if e["kind"] == "sweep" and e["tx_band_db"] is not None]
if sw:
    v = np.array([e["tx_band_db"] for e in sw])
    print("transmitter band %.0f to %.0f MHz over %d sweeps: min %.1f median %.1f max %.1f dB"
          % (TX_LO/1e6, TX_HI/1e6, v.size, v.min(), np.median(v), v.max()))
    print("sweeps where it cleared the lock threshold: %d of %d"
          % (int((v >= T.FP_PEAK_THRESH_DB).sum()), v.size))
print("--- stopped, %d events -> %s ---" % (len(ev), OUT))
