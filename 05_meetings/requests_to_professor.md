# Requests to Prof. Nierhoff

This is the running list of what we need from the professor. Before each meeting, copy that meeting's open items into its note. Afterwards, record the answer here, set the status to `answered`, and cite the meeting note it came from.

## Meeting 1 (26.10.26): needed before we record data or plan the write-up

Order for the call: 4, 9, 1, 6, then 8 if time allows. Question 7 is moved to meeting 2; question 2 is merged into 9.

| # | Need | Why | Status |
| --- | --- | --- | --- |
| 4 | For the demonstrations the robot learns from: may we drive the arm from the keyboard in small steps, or should we guide it by hand in learning mode? (The pusher is the magnet block on the rod.) | Both are imitation learning but give different data: keyboard driving gives the same pictures the robot sees later; hand-guiding is quicker but shakier | answered 26.10.26: either is fine; in his own thesis hand-guiding was faster; we may do both, compare them and combine them. The magnet block on the rod stays as the pusher (the puck must not be drilled or fitted with a magnet) |
| 9 | Camera and sheet (merges question 2). Our camera sits on the wrist and always looks straight ahead, away from the robot: about 9 cm across and 7 cm ahead of the puck (measured 25.10.26 from the push pictures). It can follow a line that runs away from the robot and bends less than about 45 degrees (sheets 01, most of 02) but not hairpins or loops (sheets 03 to 06). (a) On the exam day, may we choose how the sheet lies and where the start is, and tape it down? (b) May we re-mount the camera on the hand (Maker Space holder) so it turns with the pusher, or add a second camera? (c) Will the test path be like 01 and 02, or loop back like 04 to 06? (d) Last semester's groups managed this task: which camera set-up and which kinds of paths did they use, did their test path loop back, and may we see one of their reports or set-ups? | Decides the camera set-up and the whole recording plan; the course text says "webcam mounted onto the end effector", which a hand mount fits | answered 26.10.26 for (a): we may decide start and orientation, taping is allowed; (b): a modified camera mount may be printed by the Maker Space (STEP/STL/3MF plus group number), but the end effector must not be modified for easier pushing (FAQ); (c): unseen tracks have the same overall dimensions and line thickness as the provided ones (FAQ) |
| 1 | May we have the original path-sheet files (SVG or PDF)? And may we make our own lines (for example generated SVG paths), print them and train on them for more variety? | Exact reprints; more variety generalises better to the unseen path (this merges the old question 3) | answered 26.10.26 (team's note): more SVG track files will follow later for training; whether we may print our own is to be confirmed |
| 6 | Has SSH access to the GPU pool (DC 1.07) been activated for us? | Training needs a GPU; Moodle (12.10.26) said access follows in a few days | answered 26.10.26: we can get SSH access; ask for activation and test the login |
| 8 | How will the final run be graded: only whether the puck reaches the goal, or also how closely it follows the line and how long it takes? Must the whole line be followed, or does reaching the goal on its own count? | We train and test for what is graded | answered 26.10.26: first criterion is finishing the track (far more important than speed); then the finishing time counts and is ranked against the other groups; evaluation on the provided tracks and new unseen ones of the same size and line thickness (FAQ) |
| 2 | (merged into 9 on 25.10.26) | | merged |
| 3 | (merged into 1) | | merged |
| 5 | Which software version is the robot on, and may we update it? | PyNiryo 1.2.x needs 5.5.0 or newer | answered 23.10.26 by checking: robot software 6.1.2-b374 (current), NiryoStudio 1.16.3; no update needed (`2026-10-26_meeting_1.md`) |

## Meeting 2 (09.11.26): needed before training

| # | Need | Why | Status |
| --- | --- | --- | --- |
| 8 | (moved to meeting 1, asked if time allows) | | see meeting 1 |
| 9 | (moved to meeting 1) | | see meeting 1 |
| 7 | The MAI course catalogue (Modulhandbuch Master MAI, 19.05.2026, entry "AI project (AIPR)", pages 15 to 16) lists as the examination: "Presentation in the team (15 minutes), project documentation (7 pages)", with "a theoretical part (literature review) and practical part (software programming)". The Moodle post of 05.10.26 says only Project 3 needs documentation. Which applies to us, and what exactly do we hand in by 10.01.27 (code, model, data, report, video) and where? | Decides how much we write and lets us plan the deliverables early | open; moved to meeting 2 on the team's decision (25.10.26) |
| 10 | Must the final policy run offline or locally? May we use pretrained vision models (e.g. a ResNet backbone or LeRobot's ACT)? | Decides how much we can build on existing work, and the deployment set-up | partly answered 26.10.26: pretrained vision models may be used; offline/local requirement still open |
| 13 | May a non-printed, reversible clamp mount for the camera (hose clamp plus aluminium strip, no marks on the robot) be used for recording while the printed mount is pending, or even for the final run? | The one print must copy a geometry already photographed working; a clamp prototype gives those pictures and could start the recording earlier | open; added 26.10.26 after the design review |
| 14 | Only if a hand-mounted camera ever survives the prototype: may the mount clamp on the rod above the magnet block, turning with joint 6? | The FAQ allows a modified camera mount; a rod clamp touches the end effector's rod | open, low priority; added 26.10.26 |
| 15 | May we print our own training tracks (generated lines of the same size and line thickness)? Question 1 was answered only for the files that will follow | The plan needs 20 to 30 printed paths for variety | open; added 26.10.26 |
| 16 | Is a second camera allowed (fixed overhead or on the arm), or must the input be the end-effector camera only? | Decides whether an overhead view may help on loops | open; added 26.10.26 (asked in 9b, not answered) |

## Meeting 3 (23.11.26): needed for the submission

| # | Need | Why | Status |
| --- | --- | --- | --- |
| 11 | Presentation on 11.01.27: 15-minute team presentation with a live robot demo; we assume yes, please confirm | Prepare the slides and the demo | assumed yes (team), to confirm |

## Meeting 4 (07.12.26): final logistics

| # | Need | Why | Status |
| --- | --- | --- | --- |
| 12 | When and where is the unseen-path test run? We assume the test path is handed to us on the exam day; please confirm | Book the robot and plan the final runs | assumed (team), to confirm |
