# Checklist: camera taped to the upper arm, a view test (4.11.26)

Why: Emranul reported on 3.11 that other teams tape the camera to the upper-arm cover (between shoulder and elbow)
and see the whole sheet, and that the professor allows the camera anywhere on the robot. Our wrist camera (bracket F3)
cannot see the line behind the rod when backing. This test takes the same pictures from both places so the team can
decide on pictures, not on guesses. About 20 minutes. Nothing is cut, glued or screwed into the robot: tape only.

What the test can tell: whether the forearm or wrist hides the puck or the line from the upper arm, how big the sheet,
the line and the puck are in the picture, and how the view moves with the arm. What it cannot tell: sharpness at the
new distance, because the lens is focused at about 12 cm (refocused 2.11.26); at 35 to 45 cm the picture will be
soft. Refocusing is a separate, one-way decision (see step 8).

## Before

- [ ] The recording tool is closed (Esc in its window; the arm parks and "Disconnected" appears in the terminal).
- [ ] Robot calibrated, light steady, stop button in reach, nobody within the arm's reach while it moves.
- [ ] Sheet 04 lies as on 2.11; the puck sits on the start point (17.1 cm out, 3.7 cm left), so it appears in the pictures.

## Pictures with the wrist camera (bracket F3), the "before"

- [ ] From `02_code`: `..\.venv\Scripts\python.exe view_survey.py wrist_F3`
      The arm visits three points (17, 26 and 35 cm out) at 1 cm above pushing height, saves a picture at each, parks.
- [ ] Three files `2026-11-04_HHMM_wrist_F3_<point>.jpg` exist in `03_data/raw/camera_check/`.

## Move the camera (motors on, arm parked, nobody pulls on the arm)

- [ ] Undo the two camera screws on the F3 plate. Hold the camera by its housing only. Do not touch the focus ring.
- [ ] Leave bracket F3 on the wrist (it is not in the way) or take it off; note which.
- [ ] Tape the camera to the SIDE of the upper-arm cover, half-way between shoulder and elbow, lens pointing forward
      and about 45 degrees down, on the side where the cable runs most freely. Route the cable so no joint can pinch it.
- [ ] Check by eye with the arm parked: the lens sees past the forearm towards the sheet.

## Pictures with the taped camera, the "after"

- [ ] From `02_code`: `..\.venv\Scripts\python.exe view_survey.py upper_arm_tape`
- [ ] Same three points, three files `..._upper_arm_tape_<point>.jpg`.

## Look at the pictures together

- [ ] In the "after" pictures: is the puck visible at all three points? Is the line behind the puck (towards the base)
      visible? Does the forearm or the wrist cover anything? How wide is the line in pixels (zoom in)?
- [ ] Compare with the "before" pictures from the same points.

## Only if the team wants to go on with the upper arm (step 8)

- [ ] Refocus for the new distance with `park_for_refocus.py 0.26 0.0` (live sharpness number; turn the ring in small
      steps until the number peaks on a text page). This is the one planned refocus for the new position; going back
      to the wrist later means refocusing again.
- [ ] Every episode recorded so far (sheets 01 and 04, 2.11 and 3.11) belongs to the wrist camera and cannot be mixed
      with new data: the recording tool gets a new `camera_mount` and `mapping_version`.
