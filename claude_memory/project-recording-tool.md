---
name: project-recording-tool
description: "Recording tool: decisions of 28.10.26 (picture-frame actions), the 30.10.26 analysis, its state on 5.11.26 (four modes, all episodes so far tool tests), and 6.11.26 (puck tracking, episode checker, start-gap finding, the recording mode chosen)"
metadata:
  type: project
  modified: 2026-11-06T09:30:00.000Z
---

On 28.10.26 Emranul and Marshal decided (Marshal reported it): step directions are named in the **picture's frame** (forward = top of the picture, left = picture's left), not the robot base's; and the recording tool must support **both keyboard and hand-guided** demonstrations. Marshal builds the tool (`02_code/record_episodes.py`, branch `marshal/recording-tool`).

Design (Marshal approved "guide, then replay" for hand-guiding): every mode runs one loop (move finished → first fresh picture → decider → log → 1 cm tip move). Hand-guided = motors off, a person pushes the puck to the goal while the tip's path is logged; then the puck goes back and the robot replays the path in 1 cm steps, snapped onto it at every step, with per-step key override and a discard key. Only the replay is training data, because hand-guided pictures are blurred, show the hand and are taken mid-motion (SOTA report, Li, Cui and Sadigh 2025). Marshal raised that a smooth hand push and a 1 cm replay may move the puck differently: the override, discard and per-episode count of overridden steps handle it, and the phase-2 trial should measure how often it happens. Later the trained model becomes the decider in the same loop, and model plus key override is the correction-round mode.

Action list (Marshal, 29.10.26): keep the seven actions (forward, forward-left, forward-right, left, right, back, stop) for the first robot test. Marshal proposed back-left, back-right and a "hop over the puck to the other side"; after discussing the costs they were deferred: fewer examples per action (rare actions learnt badly), inconsistent labels between near-equal actions and two operators, large hard-to-undo moves compound errors, a one-picture model without memory can oscillate on back moves, and a hop to the puck's far side leaves the puck out of the picture (calculated from the measured view, which starts at about the puck's centre and points away from the base). Rule agreed: while recording, note each time an operator wishes for one of these actions and in what situation; add an action only if the same situation recurs and its effect is visible to the camera. A sideways hop (to the puck's left or right) is the strongest candidate if pucks often drift off the line; a far-side hop is ruled out with this camera.

Update 4.11.26 (Marshal): back_left and back_right added (nine classes, keys z and c), because pulling the puck towards the base cannot be avoided by reorienting the sheet, and with only `back` a pull cannot be steered. Marshal shared a bracket F picture of a pull: the block hides the puck except for a red sliver, and the path ahead is out of view. The earlier "far-side hop is ruled out" no longer holds outright, since pull sections leave the puck hidden anyway. Whether pulls are learnable is untested; the planned first test is straight pulls on sheet 01. Also added: step_over_forward and step_over_back (keys r and f, eleven classes), one decision that lifts, crosses 4.2 cm and lowers, chosen over a group of directional steps because a memoryless model cannot tell how far round it has got. Marshal reports pulling by teleoperation works; learnability is untested, and the 4.2 cm is calculated, not tried on the robot. Emranul has yet to confirm these changes. Every episode.json now also stores stepping_rule_version, outcome and discard_reason (asked in the window after each episode), and camera_mount defaults to bracket_F, so a rehearsal on the old holder must pass --camera-mount stock_wrist_holder.

Side view (Marshal's idea, 4.11.26, untested): turning joint 4 gave Marshal a side view of the pull path by eye. No wrist rotation can fully correct the rod's lean, because joint 5 tilts only in the plane set by the turn and joint 6 spins the rod about its own axis; lean reaches zero only with no turn or a vertical forearm. Calculated least lean at a 90 degree turn with the tip down: 1.5, 9.5, 14, 20 and 33 degrees at 20, 25, 28, 32 and 40 cm; it is larger with the tip lifted. When pulling, the rod leans over the puck, so tipping is the first risk. Emranul runs `02_code/side_view_test.py` with `06_deliverables/research_notes/Recording tool/side_view_test_checklist.md`. If it works, the tool's direction mapping must account for the turn.

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

**Tool state 5.11.26 (Emranul's session, branch `emranul/recording-tool-fixes`; Emranul tested at the robot 31.10 to 5.11 on sheet 04):** the camera is now on the forearm cover ([[project-forearm-camera-mount]]), so the picture turns with joint 4 and its heading comes from the arm model. Modes: `keyboard` (q w e a d s y c steps; `y`/`c` replace `z` for Emranul's German keyboard; `v`/`b` turn the forearm 30 degrees with a 2.5 cm lift and a 1.5 cm landing retreat; `n`/`m` lean; flick and step-over keys unbound at Emranul's request), `hand_guided` (1 cm replay that copies forearm turns), `hand_guided_exact` (replays every joint, waypoints every 1 cm or 5 degrees, `g` runs on, `h` corrects by hand; Emranul's wish: "give us all the freedom ... in motor off version and replicate that same movement"), and `gamepad` (USB pad read through XInput; full freedom on Emranul's request: only joint limits and the table-edge box refuse a step; L2/R2 height). Also: camera shift check at every episode start, steps written at every decision (a crash lost episode 041's labels), an orientation window with a key list per mode, `02_code/HOW_TO_RUN.md` so Emranul can start every tool without asking. Robot-tested: keyboard with turns, both hand-guided modes, `h`, gamepad (episode 045 reached the goal). Not robot-tested: gamepad height and wrist bend in use, the redrawn orientation window, step-over. ALL episodes up to 045 are tool tests, not training data (Emranul, 5.11.26). Plan (Emranul, 5.11.26): real data collection starts 6.11 with `hand_guided_exact`, on a big table on the first floor, one trial episode per sheet first, surroundings masked later. Open (Emranul's question, 5.11.26): should the exam start be a fixed start pose or should the robot find the start itself? Emranul expects 4 to 5 sheets at the exam; the professor allowed us to choose start and orientation (26.10.26).

**6.11.26 (night, Emranul's Recording and data session, branch `emranul/recording-tool-fixes`):** on Emranul's request
the orientation window tracks the red puck: `puck_detector.py`, `forearm_camera_model.py` (camera fitted without the robot
to episodes 037 to 045, map error about 2 mm median) and `puck_tracking.py`; display only, nothing the robot does depends
on it. The fitted camera confirms `picture_left_sign` 1 and the forearm heading model within 1 degree (no longer
untested). `check_episodes.py` checks recorded episodes (read-only). Written for Emranul: the exam-start opinion (one taped
start mark, taught start as fallback, six questions for meeting 2), the first-data-day checklist and a 200-episode plan
per sheet, in `06_deliverables/research_notes/Recording tool/2026-11-06_*`. Found: (1) the start pose puts the block
exactly touching a puck that stands on the taught point (`puck_centre_ahead_of_tip_m` 1.85 cm, block 13 mm deep); it
worked on 4 to 5.11 only because the start was taught where the ink begins, 7 mm short of the round end's centre, so keep
teaching there and centre the puck on the round end. (2) The exact replay's data do not match the planned 1 cm class
model: the rod leans 16 to 22 degrees (median), the forearm twists a little at almost every step without a turn label,
and 13 to 41 percent of the rows move under 3 mm. OPEN, for Emranul and Marshal before mass collection: keep the 1 cm
classes and record with `hand_guided` (1 cm replay), keyboard or gamepad (Claude's lean, and the 31.10 decision), or keep
`hand_guided_exact` and change the model to predict the motion itself. An independent review of that night's code (6.11
morning) found only low-severity issues, all confirmed ones fixed: the map now shows the tool's believed puck position
(hollow "puck?", which keyboard v/b turns land from) beside the seen puck when they differ by over 1.5 cm, a fault in
drawing the map never stops an episode, and the tests report skipped picture checks instead of passing them.
DECIDED 6.11.26 by Emranul (at the first-floor table, to be confirmed with Marshal): record with `hand_guided` (1 cm
replay) and train first on the move names (11 classes); every row also keeps the six joints, the pose and the exact
step, so a model on the exact numbers can be tried later from the same data if the class model fails.
`hand_guided_exact` is kept as an idea for the future (free leans and twists, e.g. if the tight loops of sheets 5 and
6 need them). New table taught 6.11 (config commit 036eee8): start x 0.1593, y -0.0152.
CHANGED later on 6.11.26 by Emranul, after Marshal's tests on sheet 04 (episodes 046 gamepad, 047 hand_guided,
048 hand_guided_exact): the main data is recorded with `hand_guided_exact` (every joint replayed, waypoints every 1 cm
or 5 deg, free leans and twists). Every row still keeps the move name (action, 11 classes), the exact step
(tip_forward_m, tip_left_m, forearm_change_deg), the pose and the six joints, so the plan stays: train on the move
names first, the exact numbers if that fails. Marshal is building a form to fill in each episode's notes first.
