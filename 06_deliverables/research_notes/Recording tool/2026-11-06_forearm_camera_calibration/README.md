# Forearm camera calibration and puck tracking, 6.11.26

Why: Emranul asked on 5.11.26 for the red puck to be tracked "properly, like in real life" in the recording tool's
orientation window. That needs a detector that finds the puck in the picture and a camera model that turns its pixel into
a place on the table. Both were built overnight by helper agents of the Recording and data session, without the robot,
from the recorded episodes 037 to 045 (sheet 04 on the old desk, the bare camera taped on the forearm cover). The code now
lives in `02_code` (`puck_detector.py`, `forearm_camera_model.py`, `puck_tracking.py`, tests in `test_puck_tracking.py`)
and the numbers in `02_code/config.py` (`FOREARM_CAMERA_PARAMETERS`, `PUCK_DETECTOR_PARAMETERS`,
`PUCK_TRACKING_PARAMETERS`). Display only: nothing the robot does depends on it.

## The camera (fitted, `calibrated_camera_parameters.json`)

- Fixed in the arm model's `joint_4` frame (it turns with joint 4; joints 5 and 6 do not move it): 58 mm from the forearm's
  axis, 18 mm along it, looking 6.2 degrees off the axis. Focal length 502 px (Niryo's file for this camera: 508), field
  of view 65 x 51 degrees, almost no lens distortion (k1 -0.02; Niryo's own distortion does not fit these pictures).
- Evidence: the white sheet's straight edges in 1603 pictures (each episode's sheet a rectangle of unknown place, all of
  one unknown size) and the puck in 85 runs of pictures in which the robot cannot have moved it, each picture with its
  logged joints; robust least squares.
- Checks the fit never used (measured through the model): the sheet's edges in held-out pictures land within 2.0 mm
  (median) and 5.1 mm (90th percentile) on the table; a puck that does not move wanders 1.9 mm (median) on the map; the
  end of the line in 039's first picture maps 1.1 mm from the start taught on 31.10; the fitted sheet is 294 x 207 mm (A4
  is 297 x 210); the black line drawn from every second picture of a day forms one sharp 1 cm band
  (`figures/line_consistency_2026-11-05.png`). Leaving one episode out at a time moves the mapped puck by under 1 mm.
- Not removable by any calibration: each picture's view scatters by about 6 px (about 2 mm on the table), shared by
  neighbouring pictures; the cause (joint play or the taped camera rocking) is not known.
- The values before the fit (Niryo's lens, the camera's place guessed from photos) were 1 to 2 cm off on the table.

Valid while: the camera stays as taped on 4.11 evening (the tool's camera check under about 4 px), the focus ring is
untouched, the robot's automatic calibration holds, and table heights come from a table plane taught on the table in use.

## The detector

- Finds the puck in 1390 of the 1391 pictures of 037 to 045 that show more than a sliver of it; no false detection in a
  checked sample of 175 pictures, nor in episode 039 (run without the puck, with a red mouse and red cables in view). 1.6
  ms per picture on the laptop (`detector_benchmark.json`).
- Misses: slivers of the puck beside the block (32 pictures), pucks held in a hand, a puck in deep shadow under a blue
  cast. It needs some white paper in view. Tuned in one room under one set of lamps: check it on the first pictures of a
  new table or room.

## Things the night's work found that matter elsewhere

- After a push, the puck sits 31 mm (median) ahead of the magnet tip, not the 18.5 mm in config's
  `puck_centre_ahead_of_tip_m`; touching would be 22 to 25 mm (Niryo's magnet cover is 20.5 to 26 mm across). Not changed
  in config; worth a measurement with a ruler.
- The sheet moved about 1 cm between 4.11 and 5.11 while the robot did not: the taught start was right on 4.11 only.
- The camera crept 4.9 px from 037 to 045 (about 2 px between neighbouring episodes).

## If the camera moves

Re-tape it to its reference picture (`06_deliverables/research_notes/Camera mount options/forearm_mount/`). If that is not
possible, record under a new mount name (`--camera-mount ..._v2`): the tool then shows the puck only in the picture, not on
the map, until the camera is fitted again. A refit uses the first episodes recorded with the new mount (the sheet's edges
must be in view) and the scripts in `scripts/fit_from_recordings/` (about 13 minutes on the laptop, no robot). They are
kept as they ran on 6.11: their paths point to the night's working folder and must be adapted (`calibration_data.py`), and
the detections are rebuilt with `scripts/detector_evaluation/build_detections_json.py`. Since the fit moved into
`02_code`, the name `FOREARM_CAMERA_PARAMETERS` that these scripts import as their "initial guess" holds the fitted camera
of 6.11 (a good start for a camera that has only moved a little); the original guess from Niryo's lens and photos is kept
as `initial_guess_for_comparison` in `calibrated_camera_parameters.json`, so start the "initial guess" comparisons from
that. `02_code/test_forearm_camera_model.py` checks the fit itself (it recovers a camera moved by a few degrees,
millimetres and pixels from synthetic pictures, with the numpy solver every laptop has and with scipy's where installed).

`scripts/capture_on_robot_not_run/` holds a calibration routine that moves the robot over pucks at known places. It ran
only in the simulator (ten fault tests pass) and is NOT needed for the map; never run it on the robot without
Emranul's go and its `operator_checklist.md`.
