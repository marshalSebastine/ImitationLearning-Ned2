"""
Lower the rod onto the taught start pose one centimetre at a time, waiting for a key before each centimetre.

For checking a new camera mount at the pushing pose nearest the robot, where the design review of 26.10.26 calculated
that a camera body may touch the forearm: the robot goes 4 cm above the start pose and then descends in 1 cm stages;
after each stage the live picture with its sharpness is shown and the robot waits. Press n in the window when the gap
between the camera body and the forearm is still clear, q to lift and go home at once. Stop button in reach. Run from
02_code, nothing else connected to the robot:
    ..\\.venv\\Scripts\\python.exe clearance_descent.py
"""

from focus_sweep import focus_sweep

CLEARANCE_PARAMETERS = {
    "approach_height_m": 0.04,
    "tip_heights_above_paper_m": [0.04, 0.03, 0.02, 0.01, 0.0],
    # Bracket F: the lens sits about 12 cm above the paper at pushing height (calc from the bracket's design, 2.11.26).
    "lens_above_tip_m": 0.12
}


if __name__ == "__main__":
    focus_sweep(CLEARANCE_PARAMETERS)
