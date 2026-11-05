---
name: project-synthetic-pictures
description: "Synthetic pictures with a Webots twin (5.11.26): Emranul decided the graded model learns from human demonstrations and the Webots twin plus a simulated teacher are an extra comparison arm for the paper; state and next step"
metadata:
  type: project
---

On 5.11.26 the orchestrator started a "Synthetic data" session (branch `emranul/synthetic-pictures`) to test cheaply
whether synthetic camera pictures can replace part of the human demonstrations. Emranul chose the Webots simulator
(R2025a, already installed on his laptop since 9.7.26) over pasting cut-outs of real pictures, and proposed a robot in
Webots trained by deep reinforcement learning as the demonstrator.

**Decided by Emranul (5.11.26):** the graded AI-project model is trained on human demonstrations (keyboard teleop or
hand-guided replay). The Webots twin and a simulated teacher are built in addition, as a comparison arm for the planned
paper (data-collection paper, contribution A of the paper assessment of 4.11.26 on branch
`claude/paper-publication-81bd4c`).

**Claude's cautions (not decided):** a teacher must decide only from what the camera can see (EgoPush: teachers above
98 percent, camera-only students 0, 55 and 87 percent); a written rule is a perfect teacher in the simulator already,
so deep reinforcement learning is worth trying only where the rule fails; simulated pictures need a precise twin and
fine-tuning on real demonstrations; simulated demonstrations in the graded model need Prof. Nierhoff's approval.

**State on 5.11.26:** Webots scene for bracket F3 with page textures of the six traced sheets (commits c5d1f6d,
cb8f8ee; `python 02_code/synthetic_pictures.py open` shows it). For the current forearm-cover camera: the Ned2 forward
kinematics from Niryo's URDF reproduce the robot's logged tool position to 0.01 mm (median); Niryo's generic lens
calibration (fx 508, k1 -0.41) is in `niryo_files/ned_ros_vision_cam_intrinsics.yaml`; the camera turns with joint 4;
its mount on the forearm was then fitted WITHOUT the robot (Emranul asked to use the existing data, 5.11.26): from 552
pairs of pictures of episodes 037 to 043, because the joints give how the camera moved and the table must look the
same in both pictures. All six starting guesses gave the same mount (in `config.SYNTHETIC_PARAMETERS
["forearm_camera_mounts"]`, commit a8113a2); warped pictures line up within a few pixels and the forearm fork from
Niryo's mesh falls on the blue cheeks seen in every picture. Fitting the sheet's line or outline directly had failed
(sheet only traced from a phone photo, never fully in view, bluish pictures make wood look like paper).

**Found on the way:** the recording session's camera-shift check (5.11.26) shows the tape moved once, between tests 4
and 5 on 4.11.26; episodes 033 to 036 need their own mount. The puck at the start of 037 and 042 sits about 1 to 1.5 cm
further out than the tool's start of 17.07 cm assumes (line start about 17.8 cm on 4.11, 18.9 cm on 5.11, from the
line fit): worth re-teaching the start with `teach_sheet_start.py` for the forearm camera.

**Next:** bring the fitting script into the repository, fit the earlier tape position, map sheet 04 from the calibrated
pictures (first map works; line detection in bluish pictures needs tuning), then replay a recorded episode in Webots
beside the real pictures. A chessboard session stays an optional precision check (the lens is Niryo's generic file).
See [[project-paper-publication]] and [[project-recording-tool]].

**Why:** a twin is only worth comparing against real demonstrations if its pictures match the real camera.

**How to apply:** keep simulated demonstrations out of the graded training set unless Prof. Nierhoff agrees; the camera
is held by double-sided tape, so recalibrate after any bump or re-taping.
