"""Shared plotting utilities for the pre-registered report figures.

Color palette and the guarded matplotlib import live here so the plotting
script stays thin and so importing this module (e.g. in tests or analysis
notebooks) never requires a display or a plotting backend installed.
"""

from __future__ import annotations

try:  # matplotlib is a plotting extra, not a core dependency of the package.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Consistent verdict colors across every report figure.
PASS_COLOR = "#1a7f37"
FAIL_COLOR = "#b42318"

__all__ = [
    "MATPLOTLIB_AVAILABLE",
    "PASS_COLOR",
    "FAIL_COLOR",
    "matplotlib",
    "plt",
]
