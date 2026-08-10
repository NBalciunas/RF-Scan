"""Shared support for the self-checks.

Each test script imports this module first. The module does three things:

  1. It puts the repo root on sys.path. Thus a script runs from any directory.
  2. It replaces the modules that need hardware drivers. Thus the pure mathematics
     runs on a machine with no PlutoSDR and no Qt.
  3. It gives the Checks harness, which prints one line for each check and gives the
     exit code.

The harness has two kinds of check.

  check()  - the behaviour is correct now. A failure is a defect.
  defect() - the behaviour is a known defect. See the section 8 of NOTES.md. A
             failure is the expected result. If the check passes, the defect is
             corrected, and the script says so.

Thus the suite stays green while the defects are open, and it tells you which defect
you corrected.
"""

import sys
import types
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── The stubs for the drivers ─────────────────────────────────────────────────

def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def stub_hardware(force=()):
    """Replace adi, pyqtgraph and PyQt5 if they are absent.

    terminal_v2 imports the three at the top of the file, but it uses them in the
    methods only. The stubs give the names that the class definitions need. Thus a
    test of the mathematics does not need libiio, and it does not need Qt.

    A real module has precedence, unless its name is in `force`. Force the stub when
    a check needs the behaviour of the stub itself: `_Signal` records what a worker
    emitted and `_QThread.start()` calls `run()` in the same thread, and the real Qt
    gives neither. Without `force` the result of a check would depend on the packages
    that the machine happens to hold, which is not a property a suite may have.

    Force `pyqtgraph` together with `PyQt5`. pyqtgraph imports Qt itself, thus a real
    pyqtgraph beside a stubbed PyQt5 gives two different Qt in one process."""
    stubbed = []

    def _absent(name):
        if name in force:
            sys.modules.pop(name, None)
            for k in [k for k in sys.modules if k.startswith(name + ".")]:
                del sys.modules[k]
            return True
        try:
            __import__(name)
            return False
        except Exception:
            return True

    if _absent("adi"):
        _stub_module("adi", Pluto=object)
        stubbed.append("adi")

    if _absent("pyqtgraph"):
        _stub_module("pyqtgraph", mkPen=lambda *a, **k: None,
                     InfiniteLine=object, ImageItem=object,
                     GraphicsLayoutWidget=object, colormap=None)
        stubbed.append("pyqtgraph")

    if _absent("PyQt5"):
        class _Signal:
            """A signal that records what was emitted, so a check can read it.

            It is a class attribute, exactly as pyqtSignal is, so every worker of
            one class shares it. A check uses one worker at a time."""

            def __init__(self, *a, **k):
                self._subs = []
                self.log = []

            def connect(self, fn):
                self._subs.append(fn)

            def disconnect(self, fn=None):
                self._subs = [] if fn is None else [s for s in self._subs if s is not fn]

            def emit(self, *args):
                self.log.append(args if len(args) != 1 else args[0])
                for fn in list(self._subs):
                    fn(*args)

            def clear(self):
                self.log.clear()

        class _Qt:
            DashLine = 2
            AlignCenter = 0x84

        class _QThread:
            def __init__(self, *a, **k):
                pass

            def wait(self, _ms=0):
                return True

            def terminate(self):
                pass

            def start(self):
                self.run()

        qtcore = _stub_module("PyQt5.QtCore", QThread=_QThread,
                              pyqtSignal=_Signal, QObject=object,
                              QRectF=object, Qt=_Qt)
        qtwidgets = _stub_module("PyQt5.QtWidgets", QMainWindow=object,
                                 QWidget=object)
        qtgui = _stub_module("PyQt5.QtGui", QFont=object, QColor=object)
        _stub_module("PyQt5", QtCore=qtcore, QtWidgets=qtwidgets, QtGui=qtgui)
        stubbed.append("PyQt5")

    return stubbed


# ── The harness ───────────────────────────────────────────────────────────────

class Checks:
    """Collect the result of each check and print a summary."""

    def __init__(self, title):
        self.title = title
        self.passed, self.failed, self.known, self.fixed = [], [], [], []
        print(f"\n{title}")
        print("=" * len(title))

    @staticmethod
    def _run(fn):
        try:
            fn()
            return None
        except Exception:
            lines = traceback.format_exc().strip().splitlines()
            return lines[-1][:150]

    def check(self, desc):
        """The behaviour must be correct. A failure fails the script."""
        def deco(fn):
            err = self._run(fn)
            if err is None:
                self.passed.append(desc)
                print(f"  ok      {desc}")
            else:
                self.failed.append((desc, err))
                print(f"  FAIL    {desc}\n            {err}")
            return fn
        return deco

    def defect(self, ref, desc):
        """The behaviour is the known defect `ref` of NOTES.md section 8."""
        def deco(fn):
            err = self._run(fn)
            if err is None:
                self.fixed.append((ref, desc))
                print(f"  FIXED   {ref} {desc}")
                print(f"            it passes now. Move it to check() and close {ref}.")
            else:
                self.known.append((ref, desc, err))
                print(f"  defect  {ref} {desc}")
            return fn
        return deco

    def note(self, text):
        """Print a measurement. It is not a check."""
        print(f"  ..      {text}")

    def report(self):
        """Print the summary and give the exit code. 0 = no unexpected failure."""
        n_ok, n_bad = len(self.passed), len(self.failed)
        parts = [f"{n_ok} ok"]
        if n_bad:
            parts.append(f"{n_bad} FAILED")
        if self.known:
            parts.append(f"{len(self.known)} known defects "
                         f"({', '.join(sorted({r for r, _d, _e in self.known}))})")
        if self.fixed:
            parts.append(f"{len(self.fixed)} FIXED "
                         f"({', '.join(r for r, _d in self.fixed)})")
        # run_all.py splits this line on '::', thus a title may hold a colon.
        print(f"  -> {self.title} :: {'  |  '.join(parts)}")
        return 1 if n_bad else 0


def run(main_fn):
    """Run a test script body and exit with its code."""
    sys.exit(main_fn())
