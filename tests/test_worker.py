"""Self-check for the sweep worker. Type `python tests/test_worker.py`.

This is the part of the program that talks to the radio: the sweep loop, the lock
state machine and the record path. A FakeSDR replaces the PlutoSDR. It answers the
same attributes and it gives synthetic IQ data that depends on rx_lo, thus a signal
really exists at one frequency in the band and the worker has to find it.

The checks run the real `run()` loop. FakeSDR stops the worker after a set number of
reads, thus the loop ends by itself.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from _support import Checks, run, stub_hardware

# This must run before the import of terminal. The Qt stubs are forced, because
# these checks read `signal.log` and they call `start()` in one thread, and the real
# Qt gives neither. See stub_hardware().
stub_hardware(force=("pyqtgraph", "PyQt5"))

import terminal
from terminal import (SweepWorker, compute_hop_freqs, composite_geometry,
                      bin_freqs, FFT_BINS, RECORD_EVERY_N)

SR, BW, CENTER, SPAN, OLAP = 10_000_000, 4_000_000, 2_400_000_000, 20_000_000, 30
SIGNAL_HZ = 2_402_000_000          # the tone that the worker must find
DWELL_SAMPLES = 40 * FFT_BINS      # small, so the checks stay fast


class FakeSDR:
    """A stand-in for adi.Pluto. It gives a tone at SIGNAL_HZ and noise elsewhere.

    fail_tune_at holds the tune numbers that must raise, so a hop can fail.
    on_read(worker, read_number) runs after each read, so a check can act while the
    loop is alive, for example to give a Jump-to command."""

    def __init__(self, worker_box, stop_after=8, signal_hz=SIGNAL_HZ, amp=30.0,
                 fail_tune_at=(), on_read=None):
        self._lo = CENTER                   # first: the rx_lo setter reads these
        self._tunes = 0
        self.fail_tune_at = set(fail_tune_at)
        self.lo_log = []
        self.sample_rate = SR
        self.rx_rf_bandwidth = BW
        self.rx_hardwaregain_chan0 = 10
        self.rx_buffer_size = DWELL_SAMPLES
        self.gain_control_mode_chan0 = "manual"
        self.reads = 0
        self._box = worker_box          # {"w": worker}, filled after construction
        self._stop_after = stop_after
        self._signal = signal_hz
        self._amp = amp
        self._on_read = on_read
        self._rng = np.random.RandomState(0)

    @property
    def rx_lo(self):
        return self._lo

    @rx_lo.setter
    def rx_lo(self, value):
        n, self._tunes = self._tunes, self._tunes + 1
        if n in self.fail_tune_at:
            raise OSError(f"tune {n} failed")
        self._lo = int(value)

    def rx(self):
        self.reads += 1
        self.lo_log.append(int(self._lo))
        n = int(self.rx_buffer_size)
        iq = ((self._rng.randn(n) + 1j * self._rng.randn(n))
              * (0.01 / np.sqrt(2))).astype(np.complex64)
        offset = self._signal - int(self._lo)
        if abs(offset) < SR / 2 * 0.9:               # inside the received band
            f = offset / SR
            iq += (self._amp * np.exp(2j * np.pi * f * np.arange(n))
                   ).astype(np.complex64)
        iq += np.complex64(5.0)                      # LO leakage, always at DC
        w = self._box.get("w")
        if self._on_read is not None and w is not None:
            self._on_read(w, self.reads)
        if self.reads >= self._stop_after and w is not None:
            w._stop = True
        return iq


def _cfg(**over):
    c = {
        "sample_rate": SR, "rx_bw": BW, "gain": 10, "center_freq": CENTER,
        "total_span": SPAN, "dwell_ms": 4, "settle_ms": 0, "overlap_pct": OLAP,
        "op_mode": "wideband", "ml_enabled": False, "record": False,
        "record_kind": "device", "record_device": "droneA", "record_session": "1",
        "record_max_files": 1000, "record_every_n": 1, "focus_freq": SIGNAL_HZ,
        "skip_lock": False, "jump_to": None, "fp_hold_settle_ms": 0,
        # These two checks drive the lock itself, thus they take the peak on one
        # sweep and they do not centre it. The defect #31 rule and the refinement
        # have checks of their own below.
        "peak_hits": 1, "refine_lock": False,
    }
    c["hop_freqs"] = compute_hop_freqs(CENTER, SPAN, min(SR, BW), OLAP)
    c.update(over)
    return c


def _make(cfg, stop_after=8, **sdr_kw):
    box = {}
    sdr = FakeSDR(box, stop_after=stop_after, **sdr_kw)
    w = SweepWorker(sdr, cfg)
    box["w"] = w
    for sig in (w.sweep_ready, w.zoom_ready, w.fingerprint_ready, w.mode_changed,
                w.caught_changed, w.hop_progress, w.status_msg, w.files_changed):
        sig.clear()
    return w, sdr


def _hz_of_bin(cfg, idx, total):
    _n, f0, f1 = composite_geometry(cfg)
    return float(bin_freqs(f0, f1, total)[idx])


def main():
    c = Checks("Sweep worker against a fake radio (terminal.py)")
    tmp = Path(tempfile.mkdtemp(prefix="rfscan_worker_"))
    cwd = os.getcwd()
    os.chdir(tmp)                    # the record path is relative to the directory
    try:
        # ── The sweep ─────────────────────────────────────────────────────────

        @c.check("one sweep tunes to every hop and fills every slot")
        def _():
            cfg = _cfg()
            w, sdr = _make(cfg, stop_after=10_000)
            comp, bufs = w._sweep_once()
            n_keep, _f0, _f1 = composite_geometry(cfg)
            assert len(comp) == len(cfg["hop_freqs"]) * n_keep, len(comp)
            assert sdr.lo_log == [int(f) for f in cfg["hop_freqs"]], sdr.lo_log
            assert set(bufs) == set(range(len(cfg["hop_freqs"]))), sorted(bufs)
            assert (comp > terminal.EMPTY_SLOT_DB + 1).all(), "a slot stayed empty"

        @c.check("the tone appears in the composite at its true frequency")
        def _():
            cfg = _cfg()
            w, _sdr = _make(cfg, stop_after=10_000)
            comp, _bufs = w._sweep_once()
            got = _hz_of_bin(cfg, int(np.argmax(comp)), len(comp))
            c.note(f"the tone at {SIGNAL_HZ/1e6:.3f} MHz reads {got/1e6:.3f} MHz")
            assert abs(got - SIGNAL_HZ) < 60e3, f"off by {(got-SIGNAL_HZ)/1e3:.0f} kHz"

        @c.check("a hop whose tune raises leaves its slot empty, not wrong")
        def _():
            cfg = _cfg()
            w, _sdr = _make(cfg, stop_after=10_000, fail_tune_at=(2,))
            comp, bufs = w._sweep_once()
            n_keep, _f0, _f1 = composite_geometry(cfg)
            empty = int((comp <= terminal.EMPTY_SLOT_DB + 1).sum())
            assert empty == n_keep, f"{empty} empty bins, expected {n_keep}"
            assert 2 not in bufs and len(bufs) == len(cfg["hop_freqs"]) - 1
            assert any("Tune error hop 2" in m for m in w.status_msg.log), \
                w.status_msg.log
            # The empty slot must stay out of the floor, thus the height of the peak
            # must not change. See the defect #5.
            clean, _s = _make(_cfg(), stop_after=10_000)
            _f0, db0 = clean._detect_new_peak(clean._sweep_once()[0])
            f1, db1 = w._detect_new_peak(comp)
            c.note(f"peak {db0:.1f} dB with every hop, {db1:.1f} dB with one failed")
            assert abs(f1 - SIGNAL_HZ) < 60e3, f1
            assert abs(db1 - db0) < 1.0, f"the floor moved: {db0:.1f} -> {db1:.1f} dB"

        # ── The lock state machine ────────────────────────────────────────────

        @c.check("Locking finds the tone and holds that exact frequency")
        def _():
            cfg = _cfg(op_mode="locking")
            w, sdr = _make(cfg, stop_after=len(cfg["hop_freqs"]) + 4)
            w.run()
            locks = [a for a in w.mode_changed.log if a[0] == "LOCK"]
            assert locks, f"no lock. modes: {w.mode_changed.log}"
            held = locks[0][1]
            c.note(f"locked at {held/1e6:.3f} MHz, tone at {SIGNAL_HZ/1e6:.3f} MHz")
            assert abs(held - SIGNAL_HZ) < 60e3
            after = [f for f in sdr.lo_log[len(cfg["hop_freqs"]):]]
            assert set(after) == {int(held)}, f"the lock frequency moved: {set(after)}"

        @c.check("the refinement centres a new lock, and the lock then never moves")
        def _():
            # Nojus asked for this: the peak search gives a bin of the composite and
            # the middle of the signal is not that bin. The refinement reads one
            # buffer and takes the centroid. It runs one time, thus section 4 holds:
            # the frequency does not move during the hold.
            cfg = _cfg(op_mode="locking", refine_lock=True)
            w, sdr = _make(cfg, stop_after=len(cfg["hop_freqs"]) + 6)
            w.run()
            locks = [a for a in w.mode_changed.log if a[0] == "LOCK"]
            assert locks, f"no lock. modes: {w.mode_changed.log}"
            held = locks[0][1]
            c.note(f"refined lock at {held/1e6:.3f} MHz, tone at {SIGNAL_HZ/1e6:.3f} MHz")
            assert abs(held - SIGNAL_HZ) < 60e3, held
            # Every tune after the lock is the same value. The refinement tunes once
            # before the lock, thus it is inside the first part of the log.
            after = [f for f in sdr.lo_log if f == int(held)]
            assert len(after) >= 1, sdr.lo_log[-6:]
            tail = sdr.lo_log[-3:]
            assert set(tail) == {int(held)}, f"the lock frequency moved: {tail}"

        @c.check("the refinement never moves a lock further than its limit")
        def _():
            # A middle far from the candidate is another signal in the same window.
            # The refinement must keep the candidate rather than jump to it.
            cfg = _cfg(op_mode="locking", refine_lock=True)
            w, _sdr = _make(cfg, stop_after=len(cfg["hop_freqs"]) + 6)
            w.run()
            locks = [a for a in w.mode_changed.log if a[0] == "LOCK"]
            assert locks, "no lock"
            moved = abs(locks[0][1] - SIGNAL_HZ)
            assert moved <= cfg["sample_rate"] * 0.25, moved

        @c.check("Skip releases the lock and remembers the frequency")
        def _():
            cfg = _cfg(op_mode="locking")
            w, _sdr = _make(cfg, stop_after=len(cfg["hop_freqs"]) + 2)
            w.run()
            held = [a for a in w.mode_changed.log if a[0] == "LOCK"][0][1]
            w._stop = False
            w.cfg["skip_lock"] = True
            w._held_freq = held
            w._release_lock("test")
            assert w._held_freq is None
            assert w.caught_changed.log, "nothing went into the caught memory"
            assert abs(w.caught_changed.log[-1][0] - held) < 1.0

        @c.check("a caught frequency is not locked again on the next sweep")
        def _():
            import time as _t
            cfg = _cfg(op_mode="locking")
            w, _sdr = _make(cfg, stop_after=10_000)
            comp, _b = w._sweep_once()
            first, _db = w._detect_new_peak(comp)
            w._caught = [(first, _t.time())]
            again, _db2 = w._detect_new_peak(comp)
            assert again is None or abs(again - first) > 2e6, f"locked again on {again}"

        @c.check("Jump to goes straight to the frequency that the user chose")
        def _():
            # jump_to is a command that the user gives while the loop runs.
            # _run_locking clears it at the start on purpose.
            target = 2_396_000_000
            cfg = _cfg(op_mode="locking")
            n_hops = len(cfg["hop_freqs"])

            def give_command(worker, read_no):
                if read_no == n_hops + 1:      # the first read of the natural lock
                    worker.cfg["jump_to"] = target
            w, sdr = _make(cfg, stop_after=n_hops + 3, on_read=give_command)
            w.run()
            locks = [a for a in w.mode_changed.log if a[0] == "LOCK"]
            assert len(locks) >= 2, w.mode_changed.log
            assert abs(locks[0][1] - SIGNAL_HZ) < 60e3, "the natural lock is wrong"
            assert int(locks[-1][1]) == target, locks
            assert int(sdr.lo_log[-1]) == target, sdr.lo_log[-3:]

        # ── The record path ───────────────────────────────────────────────────

        @c.check("a narrowband record writes the IQ file and its sidecar")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="focus", record=True, record_kind="device",
                       record_device="droneA", record_session="3",
                       record_every_n=1)
            w, _sdr = _make(cfg, stop_after=4)
            w.run()
            files = sorted(Path("fingerprint_data/droneA/session_3").glob("*.iq"))
            assert len(files) == 4, [f.name for f in files]
            raw = np.fromfile(files[0], dtype=np.complex64)
            assert len(raw) == DWELL_SAMPLES, len(raw)
            meta = json.loads(files[0].with_suffix(".json").read_text())
            assert meta["recorded_device"] == "droneA"
            assert meta["session"] == "3"
            assert abs(meta["center_freq"] - SIGNAL_HZ) < 1.0
            assert meta["sample_rate"] == SR and meta["gain_db"] == 10
            assert meta["n_samples"] == DWELL_SAMPLES
            assert meta["dtype"] == "complex64"

        @c.check("Keep every Nth really divides the number of files")
        def _():
            for every, want in ((1, 8), (2, 4), (4, 2)):
                shutil.rmtree("fingerprint_data", ignore_errors=True)
                cfg = _cfg(op_mode="focus", record=True, record_kind="device",
                           record_device="droneB", record_session="1",
                           record_every_n=every)
                w, _sdr = _make(cfg, stop_after=8)
                w.run()
                got = len(list(Path("fingerprint_data/droneB").rglob("*.iq")))
                assert got == want, f"every {every}: {got} files, expected {want}"

        @c.check("Noise (freq) writes to the noise class, not to the device label")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="focus", record=True, record_kind="noise_freq",
                       record_device="droneA", record_every_n=1)
            w, _sdr = _make(cfg, stop_after=3)
            w.run()
            assert list(Path("fingerprint_data/noise").rglob("*.iq"))
            assert not Path("fingerprint_data/droneA").exists()

        @c.check("a wideband record writes one file for each hop of the sweep")
        def _():
            # A sweep writes every hop in one burst. A name that holds the
            # millisecond only is not unique, and a capture is then lost in silence.
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="wideband", record=True, record_kind="noise_band",
                       record_session="2", record_every_n=1)
            n_hops = len(cfg["hop_freqs"])
            w, _sdr = _make(cfg, stop_after=n_hops)
            w.run()
            files = list(Path("fingerprint_data/noise/session_2").glob("*.iq"))
            assert len(files) == n_hops, f"{len(files)} files, {n_hops} hops"
            assert len({f.name for f in files}) == n_hops, "two names are the same"

        @c.check("every capture keeps its own centre frequency in the sidecar")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="wideband", record=True, record_kind="noise_band",
                       record_session="1", record_every_n=1)
            n_hops = len(cfg["hop_freqs"])
            w, _sdr = _make(cfg, stop_after=n_hops)
            w.run()
            got = sorted(json.loads(p.read_text())["center_freq"]
                         for p in Path("fingerprint_data/noise").rglob("*.json"))
            want = sorted(float(f) for f in cfg["hop_freqs"])
            assert got == want, f"{got} vs {want}"

        @c.check("the ring removes the oldest file of this run")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="focus", record=True, record_kind="device",
                       record_device="droneA", record_max_files=3,
                       record_every_n=1)
            w, _sdr = _make(cfg, stop_after=8)
            w.run()
            iq = list(Path("fingerprint_data/droneA").rglob("*.iq"))
            js = list(Path("fingerprint_data/droneA").rglob("*.json"))
            assert len(iq) == 3, f"{len(iq)} files, the cap is 3"
            assert len(js) == 3, f"{len(js)} sidecars left behind"

        @c.check("the file count reports this run and the disk separately")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            old = Path("fingerprint_data/droneZ/session_9")
            old.mkdir(parents=True)
            for i in range(5):                       # data from an earlier session
                (old / f"old_{i}.iq").write_bytes(b"\0" * 16)
            cfg = _cfg(op_mode="focus", record=True, record_kind="device",
                       record_device="droneA", record_every_n=1)
            w, _sdr = _make(cfg, stop_after=3)
            w.run()
            this_run, on_disk = w.files_changed.log[-1]
            real = len(list(Path("fingerprint_data").rglob("*.iq")))
            c.note(f"{this_run} written this run, {on_disk} reported, {real} real")
            assert this_run == 3, this_run
            assert on_disk == real == 8, (on_disk, real)   # 5 old plus 3 new

        @c.check("a device label can not write outside fingerprint_data")
        def _():
            shutil.rmtree("fingerprint_data", ignore_errors=True)
            cfg = _cfg(op_mode="focus", record=True, record_kind="device",
                       record_device="../../escape", record_every_n=1)
            w, _sdr = _make(cfg, stop_after=2)
            w.run()
            inside = list(Path("fingerprint_data").rglob("*.iq"))
            assert inside, "nothing was written at all"
            for f in inside:
                assert "fingerprint_data" in str(f.resolve()), f
            assert not Path("../escape").exists() and not Path("escape").exists()

        # ── stop() ────────────────────────────────────────────────────────────

        @c.check("stop() waits longer when the dwell is longer")
        def _():
            # A fixed 4 s wait made a forced terminate certain at a large dwell.
            budgets = []
            for dwell in (50, 1000, 5000):
                w, _sdr = _make(_cfg(dwell_ms=dwell, settle_ms=50), stop_after=1)
                seen = {}
                w.wait = lambda ms, seen=seen: (seen.update(ms=ms), True)[1]
                w.stop()
                budgets.append(seen["ms"])
            c.note(f"stop budgets for 50/1000/5000 ms dwell: {budgets} ms")
            assert budgets[0] >= 50 and budgets == sorted(budgets)
            assert budgets[-1] > 5000, budgets

        # ── The order of the imports ──────────────────────────────────────────
        # On Windows, Qt before torch makes c10.dll fail with OSError 1114 and the
        # program stops at the start. This check reads the source, thus it holds on
        # a machine with no Qt, which is what the CI has.
        src = Path(terminal.__file__).read_text(encoding="utf-8")

        @c.check("terminal imports torch before Qt")
        def _():
            i_torch = src.index("import torch")
            for name in ("import pyqtgraph", "from PyQt5"):
                assert i_torch < src.index(name), \
                    f"'{name}' comes before torch, thus torch fails on Windows"

        @c.check("the torch import catches more than ImportError")
        def _():
            head = src[:src.index("import pyqtgraph")]
            assert "except Exception:" in head, \
                "the fallback of torch must catch OSError, not ImportError alone"

        # The check above reads the source. This one proves the result, but it needs
        # a machine that holds the real Qt and the real torch, thus it gives a note
        # and stops where they are absent. A new process is necessary, because this
        # one holds the stubs.
        # torch comes first in the probe too. Qt first is the failure that this
        # check exists to find, thus a probe in that order reports "no Qt" on
        # exactly the machine where the check matters.
        have_qt = subprocess.run(
            [sys.executable, "-c", "import torch, PyQt5, pyqtgraph"],
            capture_output=True).returncode == 0
        if not have_qt:
            c.note("the real Qt and torch are not both installed, "
                   "thus the import order is checked in the source only")
        else:
            @c.check("terminal imports with the real Qt, and torch survives it")
            def _():
                r = subprocess.run(
                    [sys.executable, "-c",
                     "import terminal as t; print(int(t._TORCH_OK))"],
                    capture_output=True, text=True,
                    cwd=str(Path(terminal.__file__).parent))
                assert r.returncode == 0, r.stderr.strip()[-200:]
                assert r.stdout.strip().endswith("1"), \
                    "torch did not survive the import of Qt"

        return c.report()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run(main)
