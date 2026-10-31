---
name: project-recording-tool
description: "Recording tool: decisions of 28.10.26 (picture-frame actions, keyboard plus guide-then-replay), the 30.10.26 data-collection analysis and its recommendations, what is untested"
metadata:
  type: project
  modified: 2026-10-31T12:00:00.000Z
---

On 28.10.26 Emranul and Marshal decided (Marshal reported it): step directions are named in the **picture's frame** (forward = top of the picture, left = picture's left), not the robot base's; and the recording tool must support **both keyboard and hand-guided** demonstrations. Marshal builds the tool (`02_code/record_episodes.py`, branch `marshal/recording-tool`).

Design (Marshal approved "guide, then replay" for hand-guiding): every mode runs one loop (move finished → first fresh picture → decider → log → 1 cm tip move). Hand-guided = motors off, a person pushes the puck to the goal while the tip's path is logged; then the puck goes back and the robot replays the path in 1 cm steps, snapped onto it at every step, with per-step key override and a discard key. Only the replay is training data, because hand-guided pictures are blurred, show the hand and are taken mid-motion (SOTA report, Li, Cui and Sadigh 2025). Marshal raised that a smooth hand push and a 1 cm replay may move the puck differently: the override, discard and per-episode count of overridden steps handle it, and the phase-2 trial should measure how often it happens. Later the trained model becomes the decider in the same loop, and model plus key override is the correction-round mode.

Action list (Marshal, 29.10.26): keep the seven actions (forward, forward-left, forward-right, left, right, back, stop) for the first robot test. Marshal proposed back-left, back-right and a "hop over the puck to the other side"; after discussing the costs they were deferred: fewer examples per action (rare actions learnt badly), inconsistent labels between near-equal actions and two operators, large hard-to-undo moves compound errors, a one-picture model without memory can oscillate on back moves, and a hop to the puck's far side leaves the puck out of the picture (calculated from the measured view, which starts at about the puck's centre and points away from the base). Rule agreed: while recording, note each time an operator wishes for one of these actions and in what situation; add an action only if the same situation recurs and its effect is visible to the camera. A sideways hop (to the puck's left or right) is the strongest candidate if pucks often drift off the line; a far-side hop is ruled out with this camera.

**Why:** pictures in training must look like pictures at test time; the model's input is the picture only, so actions are defined where the model can see them.

**How to apply:** UNTESTED on the robot: `picture_left_sign` (whether picture-left is counter-clockwise; the first robot test checks one "left" step moves the scene right in the picture) and the camera-heading model `config.camera_heading_rad` (radial from the base; must be re-checked for the new mount). The simulator (`--simulated`, writes to `03_data/raw_simulated`) has rough, unmeasured push physics; never draw conclusions about the puck from it. On sheet 01 the line runs about 10 degrees off the picture's forward at the start (line 2 cm right of the base axis), so "forward" alone drifts; the operator corrects with forward-left/right. See [[ai-project-workspace-setup]].

**Analysis of 30.10.26 (Emranul's session, with Emranul and Marshal present; branch `emranul/weekend-recording-plan`):**
`06_deliverables/research_notes/Recording tool/2026-10-30_data_collection_analysis.md` criticises 13 collection approaches
against 100 checked sources. Its recommendations, not yet decided by the students: keyboard stepping stays primary (training
pictures equal test pictures); guide-then-replay only as a measured pilot with "Enter accepts, a direction key overrides"
(the 400 ms timeout is shorter than a seven-way choice), kept only if overrides stay under 15 percent of steps, discards
under 20 percent of episodes and time under 1.5 times keyboard; raw kinesthetic frames logged for free but never trained on
without a closed-loop test; trace-then-replay, a scripted line teacher and an action-choosing colour detector dropped as
label sources (outside "we push the puck ourselves"); human-gated correction rounds from the first model on, trained after
30 to 40 episodes; 30 percent of episodes start with a deliberate offset and 25 percent near the line end; keep the seven
classes and add tool keys (`u` marks the previous step a mistake, `l` marks lost) instead of classes. Budget: about 410
episodes and 30 of 36 robot hours until 20 Dec with the traced sheet lengths (24, 27, 55, 77, 57, 57 cm; the meander needs a
step budget of 116, the tool's is 80). Corrections to the SOTA report are listed at the end of its section 7 (for example
the Li, Cui and Sadigh mixture gain is about 5 percent, not 20; HG-DAgger's behaviour-cloning counts are 1 and 6, not 11 and
66). Emranul chose on 30.10 to cap the source checks at 100 to save time; 190 sources are cited only as unverified.

**Decided 31.10.26 by Emranul and Marshal:** both recommendations adopted. Keyboard stepping is the main method; guide-then-replay stays only as a measured pilot with Enter-to-confirm (kept if overrides under 15 percent of steps, discards under 20 percent, time under 1.5 x keyboard; dropped if the Ctrl-C safety test fails). The seven classes stay; tool keys `u` (previous step was a mistake) and `l` (line lost) are added instead of classes. Record in `06_deliverables/research_notes/Recording tool/2026-10-30_data_collection_decisions.md`.

**Tool state 31.10.26 (Emranul's session, branch `emranul/recording-tool-fixes`):** the fixes from the decisions are in `record_episodes.py` and `config.py` and pass the simulator tests: key drain after each picture, `u` and `l` flags, Enter-to-confirm replay, motors stay off until the person confirms release, puck-centre workspace guard, per-sheet step budget (`step_budget_for`), three timestamps per step, episode metadata (`camera_mount`, `mapping_version`, orientation, start index, offset, lighting, laptop), operator in the session folder name. Still untested on the robot. When bracket F is fitted, change `RECORDING_PARAMETERS["camera_mount"]` and `mapping_version` and re-check `camera_heading_rad`.
