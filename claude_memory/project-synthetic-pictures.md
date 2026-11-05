---
name: project-synthetic-pictures
description: "Webots twin for synthetic pictures, cancelled by Emranul on 5.11.26; code only on branch emranul/synthetic-pictures (not merged into main); still useful: URDF arm model, Niryo's lens file, how the forearm camera was fitted and why the other fits failed, the chessboard proposal"
metadata:
  type: project
---

**Cancelled by Emranul on 5.11.26** (passed on by the orchestrator session): no Webots twin and no simulated teacher.
The graded AI-project model learns from human demonstrations (keyboard teleop or hand-guided replay), as Emranul had
decided earlier that day. The code (Webots scenes, the forearm-camera twin, `02_code/synthetic_pictures.py`,
`config.SYNTHETIC_PARAMETERS`) stays on branch `emranul/synthetic-pictures` on GitHub and is not merged into main. The
meeting-2 figure and the transfer test of the session's brief were not made.

What it was: on 5.11.26 the orchestrator started a "Synthetic data" session to test whether synthetic camera pictures
can replace part of the human demonstrations. Emranul chose the Webots simulator (R2025a, already installed on Emranul's
laptop) and wanted the twin plus a simulated teacher as a comparison arm for the planned paper (contribution A of the
paper assessment of 4.11.26, branch `emranul/paper-publication`). When it was cancelled, the twin replayed episode 042
beside the real pictures (match checked by eye).

**Worth keeping without Webots:**
- Arm model: the URDF chain in `06_deliverables/research_notes/Camera mount options/scripts/ned2_fk.py` (on main)
  reproduces the robot's logged tool position from the logged joints to 0.01 mm (median over all 3. to 5.11 episodes;
  95th percentile 1.9 mm, probably the timing between the joint and pose reads).
- Lens: Niryo's generic calibration `niryo_files/ned_ros_vision_cam_intrinsics.yaml` (fx 508.1, fy 506.4, cx 332.8,
  cy 243.0, k1 -0.415, k2 0.241, k3 -0.104), not measured on our camera. Its distortion polynomial folds back beyond a
  normalised radius of about 1.05, so the picture's corners lie outside its valid range.
- Forearm-cover camera: it turns with joint 4. Fits against the sheet failed: its paper edges (bluish pictures make the
  wood look like paper; the sheet is never fully in view), its printed corners (found in no picture) and the traced line
  (traced from a phone photo, about 5 percent off; free placements found wrong minima: line shrunk, camera off the
  forearm). What worked: a fit from 552 pairs of pictures of episodes 037 to 043, because the joints give how the camera
  moved and the table must look the same in both pictures. All six starting guesses agreed, warped pictures line up
  within a few pixels and Niryo's forearm mesh falls on the blue cheeks seen in every picture; not checked against a
  measurement. The mount is in `config.SYNTHETIC_PARAMETERS["forearm_camera_mounts"]` on the branch; the fitting script
  itself was never committed.
- Chessboard proposal (not done): an exactly known reference for a precise calibration, about 15 minutes at the robot:
  a printed chessboard on the table, pictures logged by the recording tool at varied arm poses (hand-eye calibration),
  plus teaching the sheet corners with the rod. Needs a checklist and Emranul's go-ahead.
- The tape moved once, between tests 4 and 5 on 4.11.26 (the recording session's camera-shift check): episodes 033 to
  036 and 037 to 043 have different camera positions. The puck at the start of 037 and 042 sits about 1 to 1.5 cm
  further out than the tool's start of 17.07 cm assumes: worth re-teaching the start with `teach_sheet_start.py`.
- If a simulated teacher comes back for the paper: it must decide only from what the camera can see (EgoPush: teachers
  above 98 percent, camera-only students 0, 55 and 87 percent), and simulated demonstrations stay out of the graded
  model unless Prof. Nierhoff agrees.

See [[project-paper-publication]] and [[project-recording-tool]].

**Why:** so no session restarts the Webots work or waits for the branch to be merged, and the geometry findings are not
lost with it.

**How to apply:** do not merge or build on `emranul/synthetic-pictures` unless Emranul revives it; for any picture
geometry start from `ned2_fk.py` and Niryo's lens file; the camera is held by double-sided tape, so recalibrate after any
bump or re-taping.
