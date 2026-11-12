"""
Verifier's drawing of the traced lines of the six sheets with the arc length written along them, and the verifier's own
turning measure (how much the line's direction changes within 3 cm before and after each place).

Read-only; reads ../inputs/audit_line_profiles.npz (the analyst's copy of the audit's traced lines) and writes one PNG.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy
from matplotlib import pyplot

LINE_PARAMETERS = {
    "profiles_path": Path(__file__).resolve().parent.parent / "inputs" / "audit_line_profiles.npz",
    "half_window_cm": 3.0,
    "label_every_cm": 5.0
}


def own_turning(points: numpy.ndarray, half_window_cm: float) -> tuple:
    """
    Return the line's absolute and net turning within a window round every sample (the verifier's own code).

    The heading is taken from chords of 1 cm (not from single 1-mm steps), which removes the tracing noise without a
    separate smoothing step.

    Parameters
    ----------
    points : numpy.ndarray
        (samples, 2) in cm, one sample every 0.1 cm.
    half_window_cm : float
        Half the window along the line.

    Returns
    -------
    tuple of numpy.ndarray
        Absolute turning and net turning in degrees, per sample.
    """

    count = len(points)
    chord = 5
    ahead = numpy.clip(numpy.arange(count) + chord, 0, count - 1)
    behind = numpy.clip(numpy.arange(count) - chord, 0, count - 1)
    vector = points[ahead] - points[behind]
    heading = numpy.unwrap(numpy.arctan2(vector[:, 1], vector[:, 0]))
    half = int(round(half_window_cm / 0.1))
    absolute = numpy.zeros(count)
    net = numpy.zeros(count)
    change = numpy.abs(numpy.diff(heading, prepend = heading[0]))
    cumulative = numpy.cumsum(change)
    for index in range(count):
        low, high = max(0, index - half), min(count - 1, index + half)
        absolute[index] = numpy.degrees(cumulative[high] - cumulative[low])
        net[index] = numpy.degrees(heading[high] - heading[low])
    return absolute, net


if __name__ == "__main__":
    settings = dict(LINE_PARAMETERS)
    profiles = numpy.load(settings["profiles_path"])
    figure, axes = pyplot.subplots(2, 3, figsize = (18, 12))
    for axis, number in zip(axes.ravel(), ["01", "02", "03", "04", "05", "06"]):
        points, arc = profiles[f"points_{number}"], profiles[f"arc_{number}"]
        absolute, net = own_turning(points, settings["half_window_cm"])
        kind_colour = numpy.where(numpy.abs(net) >= 150, "red", numpy.where(absolute >= 60, "orange", numpy.where(absolute >= 15, "gold", "grey")))
        axis.scatter(points[:, 0], points[:, 1], c = kind_colour, s = 8)
        for label_cm in numpy.arange(0.0, arc[-1], settings["label_every_cm"]):
            index = int(numpy.argmin(numpy.abs(arc - label_cm)))
            axis.annotate(f"{label_cm:.0f}", points[index], fontsize = 8, color = "blue")
        axis.set_aspect("equal")
        axis.set_title(f"sheet {number}: {arc[-1]:.1f} cm; hairpin samples {int((numpy.abs(net) >= 150).sum())}, "
                       f"sharp {int(((absolute >= 60) & (numpy.abs(net) < 150)).sum())}, largest net turning {numpy.abs(net).max():.0f} deg")
        print(f"sheet {number}: length {arc[-1]:.1f} cm; share of line straight {float((absolute < 15).mean()):.3f}, gentle "
              f"{float(((absolute >= 15) & (absolute < 60)).mean()):.3f}, sharp {float(((absolute >= 60) & (numpy.abs(net) < 150)).mean()):.3f}, "
              f"hairpin {float((numpy.abs(net) >= 150).mean()):.3f}; hairpin arcs: "
              f"{[round(float(value), 1) for value in arc[numpy.abs(net) >= 150][[0, -1]]] if (numpy.abs(net) >= 150).any() else []}")
        if number == "04":
            hairpin = numpy.abs(net) >= 150
            edges = numpy.flatnonzero(numpy.diff(hairpin.astype(int)))
            print("   sheet 04 hairpin stretches (arc from A, cm):", [round(float(arc[edge]), 1) for edge in edges])
    figure.tight_layout()
    figure.savefig(Path(__file__).resolve().parent / "v02_sheet_lines.png", dpi = 70)
