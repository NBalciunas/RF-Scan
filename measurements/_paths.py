"""Put the repository and its tests on sys.path, from wherever this folder sits.

Each program in measurements/ imports this first. Thus the programs do not hold an
absolute path to one machine, and they run from any directory."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
