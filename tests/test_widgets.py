"""The Qt widgets: the badge, the markers and the waterfall, with no radio.

Every other script of this suite checks a plain function. This one builds the real
`PlutoApp` and drives its handlers with arrays, thus it checks the part that the user
reads: the text of the badge, the title of the narrowband plot, and the lines that the
program draws on it.

Two things make it possible and both are recent.

  * `PlutoApp(connect=False)` builds the window and does not open the radio. Before
    that split the constructor connected and started a worker, thus the class could
    not be built at a desk at all.
  * `QT_QPA_PLATFORM=offscreen` gives Qt a platform that needs no screen.

The script needs a real PyQt5 and a real pyqtgraph. The CI installs neither, thus it
reports that it is not applicable and gives 0 there. It does not need a radio, it does
not need libiio, and it must never need them.
"""

import os

# Both must come before any import of Qt: the platform is read when the plugin loads,
# and the scale policy of terminal.py runs in its __main__ block only.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util
import numpy as np

import _support
from _support import Checks, run


def _qt_is_real():
    """True when a real PyQt5 and a real pyqtgraph are on this machine."""
    try:
        return all(importlib.util.find_spec(n) is not None
                   for n in ("PyQt5", "pyqtgraph"))
    except Exception:
        return False


def _zoom_arrays(sample_rate=10e6, n=1024, center=2.44e9):
    """Give (freqs, psd) of one narrowband window with a flat noise floor."""
    freqs = np.linspace(center - sample_rate / 2, center + sample_rate / 2, n)
    return freqs, np.full(n, -80.0)


def main():
    c = Checks("The Qt widgets, offscreen")
    if not _qt_is_real():
        c.note("PyQt5 or pyqtgraph is absent. The widgets need the real Qt, thus "
               "this script does nothing here.")
        return c.report()

    # The stubs must not win over the real Qt in this script, and adi must not be
    # imported: the machine may hold libiio and the check must not touch a radio.
    _support.stub_hardware()
    from PyQt5 import QtWidgets
    import terminal

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = terminal.PlutoApp(connect=False)
    # A line of the plot is a QGraphicsItem, thus it is visible only when every
    # parent above it is. The offscreen platform draws to memory and needs no screen.
    win.show()

    @c.check("the window builds with no radio, and it opens none")
    def _():
        assert win.sdr is None
        assert not hasattr(win, "worker"), "connect=False must start no worker"
        assert win.total_bins > 0 and win.wf_data.shape[0] == win.total_bins

    @c.check("a sweep reaches the curve and the waterfall")
    def _():
        comp = np.linspace(-90.0, -10.0, win.total_bins)
        win._on_sweep_ready(comp, {})
        assert np.allclose(win.wf_data[:, -1], comp)
        assert len(win.curve.getData()[1]) == win.total_bins

    @c.check("a composite of the wrong length is dropped and draws nothing")
    def _():
        before = win.wf_data[:, -1].copy()
        win._on_sweep_ready(np.zeros(win.total_bins + 7), {})
        assert np.allclose(win.wf_data[:, -1], before)

    # ── The badge ─────────────────────────────────────────────────────────────
    dev = {"label": "DJI-MINI-3", "confidence": 0.93,
           "probs": {"DJI-MINI-3": 0.93, "Radiolink-AT9S-Pro": 0.02, "noise": 0.05},
           "detections": [{"label": "DJI-MINI-3", "share": 0.37,
                           "confidence": 0.9}]}
    quiet = {"label": "noise", "confidence": 0.97,
             "probs": {"DJI-MINI-3": 0.02, "Radiolink-AT9S-Pro": 0.01, "noise": 0.97},
             "detections": []}

    @c.check("the badge carries the name of the device")
    def _():
        win._on_fingerprint_ready(dev)
        assert win.det_badge.text().startswith("DJI-MINI-3"), win.det_badge.text()

    @c.check("the badge reads clear on a quiet capture")
    def _():
        win._on_fingerprint_ready(quiet)
        assert win.det_badge.text().startswith("clear"), win.det_badge.text()

    @c.check("the confidence bars follow the probabilities")
    def _():
        win._rebuild_conf_bars(list(dev["probs"]))
        win._on_fingerprint_ready(dev)
        assert win._conf_bars["DJI-MINI-3"].value() == 93
        assert win._conf_labels["noise"].text() == "5%"

    @c.check("the band plan is a second line and never the name")
    def _():
        win._on_fingerprint_ready(dict(dev, band_plan="WiFi ch 6"))
        text = win.det_badge.text()
        assert text.startswith("DJI-MINI-3") and "band plan: WiFi ch 6" in text

    # The warning of §5: the model and the radio disagree about the sample rate. The
    # arithmetic was checked and the panel was not, because PlutoApp was never built.
    class _FakeEngine:
        classes = ["DJI-MINI-3", "noise"]
        sample_rate = 20e6

    @c.check("the badge warns when the model and the radio disagree on the rate")
    def _():
        win._engine = _FakeEngine()
        assert win._sr_mismatch() is True
        win._on_fingerprint_ready(dev)
        assert "sample rate" in win.det_badge.text()
        win._engine = None
        win._on_fingerprint_ready(dev)
        assert "sample rate" not in win.det_badge.text()

    # ── The narrowband plot and its markers ───────────────────────────────────
    @c.check("the markers stay off while the user has not asked for them")
    def _():
        assert not win.w_show_mid.isChecked() and not win.w_show_borders.isChecked()
        freqs, psd = _zoom_arrays()
        psd[500:540] = -40.0
        win._on_zoom_ready(freqs, psd, 2.44e9)
        assert not win.mid_line.isVisible()

    win.w_show_mid.setChecked(True)
    win.w_show_borders.setChecked(True)

    @c.check("a signal in the window puts the middle line on it")
    def _():
        freqs, psd = _zoom_arrays()
        mid = len(psd) // 2
        psd[mid - 40: mid + 40] = -40.0
        win._on_zoom_ready(freqs, psd, 2.44e9)
        assert win.mid_line.isVisible()
        assert abs(win.mid_line.value() - freqs[mid]) < 0.2e6
        assert all(ln.isVisible() for ln in win.edge_lines)
        assert "full of signal" not in win.p_zoom.titleLabel.text

    @c.check("a full window says so in the title and draws no line")
    def _():
        freqs, psd = _zoom_arrays()
        psd[:] = -40.0                      # the window is full from side to side
        win._on_zoom_ready(freqs, psd, 2.44e9, band_floor=-80.0)
        assert "full of signal" in win.p_zoom.titleLabel.text
        assert not win.mid_line.isVisible()
        assert not any(ln.isVisible() for ln in win.edge_lines)

    @c.check("the band floor reaches the panel, and says when there is none")
    def _():
        freqs, psd = _zoom_arrays()
        win._on_zoom_ready(freqs, psd, 2.44e9, band_floor=-77.5)
        assert "-77.5 dB" in win.floor_lbl.text(), win.floor_lbl.text()
        # Narrowband never sweeps, thus it has no band floor and must not show one.
        win._on_zoom_ready(freqs, psd, 2.44e9, band_floor=None)
        assert "no sweep" in win.floor_lbl.text(), win.floor_lbl.text()

    @c.check("the caught list reaches its label and the Jump to list")
    def _():
        win._on_caught_changed([2.412e9, 2.44e9])
        text = win.caught_lbl.text()
        assert "2412.00" in text and "2440.00" in text, text
        assert win.w_jump_combo.count() == 2
        win._on_caught_changed([])
        assert win.caught_lbl.text() == "Caught: —"
        assert win.w_jump_combo.count() == 0

    win.close()
    return c.report()


if __name__ == "__main__":
    run(main)
