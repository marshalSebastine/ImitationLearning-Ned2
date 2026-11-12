"""
Verifier 20: would the proposed stand-still guard (last 8 decisions: tip moved under 1 cm net AND forearm turned under 5
degrees net) have ended a run that worked? Calculated, first order: the logged answers of each run carried out with the
asked length (x 1.0 and x 1.3, kept within 0.2 to 1.5 cm) along the logged headings. Also the guard already in the code
(30 pictures: all tips within a box of 3 cm diagonal AND joint 4 within 15 degrees), and where in the run each would fire.
"""
import numpy
from vcommon import run_log, RUNS

print("run  outcome-of-11.11  decisions | proposed guard (8 / 1 cm / 5 deg): first decision where it would fire, x1.0 and x1.3 | guard in the code (30 / 3 cm / 15 deg), x1.0 | asked: share under 0.3 cm and 1.5 deg")
outcome = {1: "end reached", 2: "end reached", 3: "failed", 4: "end reached", 5: "failed", 6: "failed", 7: "end reached", 8: "failed", 9: "failed", 10: "failed"}
for number in RUNS:
    s = run_log(number)
    s = s[s["action"] == "model_step"].reset_index(drop = True)
    heading = numpy.radians(s["picture_heading_deg"].to_numpy())
    angle = numpy.radians(s["model_angle_deg"].to_numpy())
    twist = s["model_twist_deg"].to_numpy()
    fired = {}
    for factor in (1.0, 1.3):
        length = numpy.clip(factor * s["asked_cm"].to_numpy(), 0.2, 1.5)
        x = numpy.r_[0.0, numpy.cumsum(length * numpy.cos(heading + angle))]
        y = numpy.r_[0.0, numpy.cumsum(length * numpy.sin(heading + angle))]
        turned = numpy.r_[0.0, numpy.cumsum(twist)]
        first = None
        for k in range(8, len(s) + 1):
            if numpy.hypot(x[k] - x[k - 8], y[k] - y[k - 8]) < 1.0 and abs(turned[k] - turned[k - 8]) < 5.0:
                first = k
                break
        fired[factor] = first
        if factor == 1.0:
            coded = None
            for k in range(30, len(s) + 1):
                box = numpy.hypot(x[k - 30:k + 1].max() - x[k - 30:k + 1].min(), y[k - 30:k + 1].max() - y[k - 30:k + 1].min())
                if box < 3.0 and turned[k - 30:k + 1].max() - turned[k - 30:k + 1].min() < 15.0:
                    coded = k
                    break
    still = ((s["asked_cm"] < 0.3) & (s["model_twist_deg"].abs() < 1.5)).mean()
    print(f"{number:3d}  {outcome[number]:12s} {len(s):9d} | x1.0: {fired[1.0]}   x1.3: {fired[1.3]} | {coded} | {still:.2f}")
print("None = would not fire. First order only: after the first different step the robot would have seen other pictures.")
