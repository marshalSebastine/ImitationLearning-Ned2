# Data-collection decisions of 30 and 31 October 2026

Decided by Emranul Hoque and Marshal Sebastine together on 31.10.26, after reading the summary of
`2026-10-30_data_collection_analysis.md` (the long note has the evidence; `2026-10-30_data_collection_summary.md` the short version).
The analysis was prepared by Emranul's Claude session; the decisions are the students'.

## Decision 1: how hand-guided demonstrations are collected

- Keyboard stepping is the main collection method: one key per 1 cm step, the picture taken after the move settles, recorded through
  the same loop the trained model will later run.
- Marshal's hand-guided mode ("guide with motors off, then the robot replays the tip path in 1 cm steps") stays in the tool only as a
  measured pilot, with one change: the 400 ms override window is replaced by "Enter executes the proposed step, a direction key
  overrides, no timeout".
- The pilot: 10 keyboard and 10 hand-guided episodes per operator on sheet 01 with the stock camera (rehearsal), repeated on sheet 02
  with bracket F. Hand-guiding stays if overrides are under 15 percent of steps, discards under 20 percent of episodes and the time per
  finished episode under 1.5 times keyboard. Above 30 percent overrides it survives only as an on-screen suggestion while a person
  presses every key. It is dropped if the safety test fails (Ctrl-C while someone holds the torque-free arm must not move the robot).
- Only replay pictures, never pictures taken while a hand guides the arm, may enter a training set, and only after a closed-loop test.

## Decision 2: the set of output classes

- The seven classes stay: forward, forward_left, forward_right, left, right, back, stop, named in the picture's frame.
- Two tool keys are added instead of classes: `u` marks the previous step as a mistake (training drops it and the step that undid it;
  the episode survives), `l` marks "line lost" (the frame gets no label).
- A class is added only if the recording notes show the same situation recurring and the camera can see the action's effect.
- Comes with it: one written stepping rule both operators follow, drawn as a reference band on the screen; a 60-picture agreement test
  (about 80 percent agreement before real recording); quotas of about 30 percent deliberately offset starts and about 25 percent
  near-end starts per sheet.

## What this changes in the code (`02_code/`, before the first robot test)

- `record_episodes.py`: drain the key buffer after each picture is shown; keys `u` and `l`; Enter-to-confirm replay; exception-safe
  guiding (nothing moves on Ctrl-C while the arm is held, torque state shown); laptop name in the session folder; camera mount, mapping
  version, sheet orientation, start index, deliberate offset and lighting note in `episode.json`; workspace guard on the predicted puck
  centre; three timestamps per step.
- `config.py`: `RECORDING_PARAMETERS["step_budget"]` per sheet (1.5 x line length in centimetres; the meander needs 116), the new keys,
  `replay_key_wait_ms` replaced by the confirm rule.

## Open for the professor (meeting 2, 9 November)

Whether a confirmed replay and human-gated correction rounds count as "pushing the puck ourselves"; printed training sheets; a visible
line end on the evaluation sheets; how "finishing the track" is judged. Full list of 14 questions in the analysis, section 7.
