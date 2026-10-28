---
name: project-recording-tool
description: "Recording tool decisions of 28.10.26: actions in the picture's frame, keyboard plus hand-guided (guide then replay) modes, one shared step loop; what is untested"
metadata:
  type: project
  modified: 2026-10-28T18:00:00.000Z
---

On 28.10.26 Emranul and Marshal decided (Marshal reported it): step directions are named in the **picture's frame** (forward = top of the picture, left = picture's left), not the robot base's; and the recording tool must support **both keyboard and hand-guided** demonstrations. Marshal builds the tool (`02_code/record_episodes.py`, branch `marshal/recording-tool`).

Design (Marshal approved "guide, then replay" for hand-guiding): every mode runs one loop (move finished → first fresh picture → decider → log → 1 cm tip move). Hand-guided = motors off, a person pushes the puck to the goal while the tip's path is logged; then the puck goes back and the robot replays the path in 1 cm steps, snapped onto it at every step, with per-step key override and a discard key. Only the replay is training data, because hand-guided pictures are blurred, show the hand and are taken mid-motion (SOTA report, Li, Cui and Sadigh 2025). Marshal raised that a smooth hand push and a 1 cm replay may move the puck differently: the override, discard and per-episode count of overridden steps handle it, and the phase-2 trial should measure how often it happens. Later the trained model becomes the decider in the same loop, and model plus key override is the correction-round mode.

**Why:** pictures in training must look like pictures at test time; the model's input is the picture only, so actions are defined where the model can see them.

**How to apply:** UNTESTED on the robot: `picture_left_sign` (whether picture-left is counter-clockwise; the first robot test checks one "left" step moves the scene right in the picture) and the camera-heading model `config.camera_heading_rad` (radial from the base; must be re-checked for the new mount). The simulator (`--simulated`, writes to `03_data/raw_simulated`) has rough, unmeasured push physics; never draw conclusions about the puck from it. On sheet 01 the line runs about 10 degrees off the picture's forward at the start (line 2 cm right of the base axis), so "forward" alone drifts; the operator corrects with forward-left/right. See [[ai-project-workspace-setup]].
