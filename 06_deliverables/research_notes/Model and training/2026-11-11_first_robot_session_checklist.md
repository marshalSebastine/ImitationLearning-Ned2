# First robot session with the model mode: checklist

Written 11.11.26 by the Model and training session for Emranul (and Marshal). It follows report section 7.4 (phases 0a,
0b and 1a) and the project's robot rules. Nothing on this list has been tried on the robot yet. Tick each box; stop and
ask when anything looks different from what is written.

## Before the robot is switched on (laptop only)

- [ ] Only one program will talk to the robot: close the recording page, viewers and any other script.
- [ ] In a terminal: `cd D:\AI_Project\.claude\worktrees\training-analysis\02_code`
- [ ] The offline checks pass: `..\..\..\..\.venv\Scripts\python.exe test_model_mode.py` ends with "all tests passed"
      (about 1 minute; no robot needed).
- [ ] The model files are there: `..\04_runs\gpu_ki014\2026-11-10_resnet18_model_1_all_sheets\checkpoints\all_sheets.pt`
      and the fold models in `..\04_runs\gpu_ki014\2026-11-09_resnet18_final_candidate\checkpoints\without_path_01.pt`
      to `without_path_06.pt`.
- [ ] Decide who runs the laptop and who watches the arm and the puck (and keeps the stop button in reach).

## Robot set-up (the usual rules)

- [ ] Robot on its table, away from the table edge; the sheet 5 cm in front of the base, line pointing away (as in the
      recordings).
- [ ] Robot switched on; only the automatic calibration after a restart (`calibrate_robot.py` if asked), never manual.
- [ ] The camera and its focus ring are not touched. If the camera's tape looks moved, stop and follow the camera-mount
      checklist.
- [ ] Ceiling light on, as in most recordings. Note if daylight is strong.
- [ ] Laptop connected by the Ethernet cable (the address in `config.py`).

## How one run works (read once before the first run)

1. Start the program (commands below). It asks you to put the puck at the line's start, press Enter.
2. **The motors go OFF**: hold the arm. Guide the magnet tip just behind the puck, the camera looking along the line.
3. **Let go of the arm**, then press Enter. The motors come back on. The program checks the placed point (height,
   reach, workspace, forearm twist); if it refuses, it says why and nothing moves.
4. The arm lifts 3 cm and comes back down with the wrist and hand at their held angles.
5. A picture window opens. **Click on it**, then press **Enter** there to let the model push (Esc there = do not start).
6. The model pushes, 1 cm per picture. The window shows its direction (green arrow) and its stop chance.
   **Esc ends the run** (in the window or the terminal); the arm then lifts 5 cm. The stop button is for anything
   unsafe.
7. The run ends by itself when the stop chance is 0.5 or more on two pictures in a row, after 250 steps, or at the first
   step a guard refuses. Every picture and the model's answers are saved in `D:\AI_Project\03_data\model_runs\`.

## Phase 0a: movement check, no puck (about 15 minutes)

Goal: the arm moves the right way; nothing is learned about the model.

- [ ] Sheet 01, **no puck**. Command:
      `..\..\..\..\.venv\Scripts\python.exe model_mode.py robot --label check_no_puck`
- [ ] Place the tip at the line's start as if the puck were there. After Enter in the window, watch the first 5 steps:
      each moves about 1 cm, roughly along the green arrow's direction on the table, never down into the sheet.
- [ ] Press Esc after about 10 steps: the run ends at once and the arm lifts.
- [ ] Repeat once and let it run: note how it ends (stopped / refused / step budget).
- [ ] If any move looks wrong (the wrong way, a jump, the rod scraping the paper): stop, write down what you saw and the
      run folder's name, and do not go on to phase 0b.

## Phase 0b: model 1 on known sheets (about 30 minutes)

Model 1 learned these sheets, so this shows whether the whole chain works, not the exam.

- [ ] Sheet 01 (straight line), puck on, 3 runs: `... model_mode.py robot --label model1_sheet01`
- [ ] Sheet 02 (arc), 3 runs: `--label model1_sheet02`

## Phase 1a: like the exam, unseen sheets (about 1 hour)

On each sheet, the fold model that was trained **without** that sheet (it has never seen it). For sheet 03:

`..\..\..\..\.venv\Scripts\python.exe model_mode.py robot --label fold03_sheet03 --run-name resnet18_final_candidate --checkpoint ..\04_runs\gpu_ki014\2026-11-09_resnet18_final_candidate\checkpoints\without_path_03.pt --metrics ..\04_runs\gpu_ki014\2026-11-09_resnet18_final_candidate\metrics.json`

- [ ] Sheet 03 with `without_path_03.pt`, 3 runs
- [ ] Sheet 04 with `without_path_04.pt`, 3 runs
- [ ] Sheet 05 with `without_path_05.pt`, 3 runs
- [ ] Sheet 06 with `without_path_06.pt`, 3 runs
- [ ] Only if time is left: model 2 or 3 from the night queue, if the morning comparison says they are worth it.

## The models ready for the robot (11.11.26 morning)

All trained on all six sheets on ki-014 and checked in the simulator. Offline scores are on the six held-out sheets
(the six-sheet test of the same recipe): path within 45 degrees, and right stops of 209 under the fixed rule. Write
`G` for `..\04_runs\gpu_ki014` in the commands below.

| Letter | Model | Offline path within 45 | Right stops | Command after `..\..\..\..\.venv\Scripts\python.exe model_mode.py robot --label <name>` |
|---|---|---|---|---|
| A | Model 1, one network (the default) | 0.787 | 92 | nothing more |
| B | Team of the 5 model-1 seeds, answers averaged | 0.807 | 95 | `--checkpoint G\2026-11-10_resnet18_model_1_all_sheets\checkpoints\all_sheets.pt G\2026-11-10_resnet18_model_1_all_sheets_seed_2\checkpoints\all_sheets.pt G\2026-11-10_resnet18_model_1_all_sheets_seed_3\checkpoints\all_sheets.pt G\2026-11-10_resnet18_model_1_all_sheets_seed_4\checkpoints\all_sheets.pt G\2026-11-10_resnet18_model_1_all_sheets_seed_5\checkpoints\all_sheets.pt` |
| C | Model 2, the chunk recipe (also learns the next 5 steps) | 0.817 and 0.802 (two seeds) | 87 | `--run-name resnet18_model_2_chunk_all_sheets --checkpoint G\2026-11-10_resnet18_model_2_chunk_all_sheets\checkpoints\all_sheets.pt --metrics G\2026-11-10_resnet18_model_2_chunk_all_sheets\metrics.json` |
| D | Model 3, 16 rounds | 0.777 | 103 (fewest early stops, 14) | `--run-name resnet18_model_3_16_rounds_all_sheets --checkpoint G\2026-11-10_resnet18_model_3_16_rounds_all_sheets\checkpoints\all_sheets.pt --metrics G\2026-11-10_resnet18_model_3_16_rounds_all_sheets\metrics.json` |
| E | Model 4, chunk with the stop weighted 25 (round 1) | 0.808 and 0.800 | 103 and 105 (never stopped 71 and 51) | `--run-name resnet18_model_4_chunk_stop_25_all_sheets --checkpoint G\2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets\checkpoints\all_sheets.pt --metrics G\2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets\metrics.json` |
| F | Model 5, chunk with ResNet-34 (round 1) | 0.813 and 0.807 | 95 and 101 | `--run-name resnet18_model_5_chunk_resnet34_all_sheets --checkpoint G\2026-11-10_resnet18_model_5_chunk_resnet34_all_sheets\checkpoints\all_sheets.pt --metrics G\2026-11-10_resnet18_model_5_chunk_resnet34_all_sheets\metrics.json` |

Not for the robot: model 6 (ResNet-34 and the stop weighted 25 together) failed the fixed rule (path 0.801 and 0.795,
right stops 92 and 109, too-early stops rose to 55 and 47).

Suggested order: A first (the reference), then E, F and B on the same sheets and starts (E is the favourite: the best
stopping without losing steering), C if time allows, D only if stopping is the main failure. Run them in a mixed order, not all A runs first, so a change of light or tape over the session does not favour
one model.

**One setting to try on the robot, not offline:** `--twist-gain 1.5` makes the forearm turn 1.5 times what the model
says. Offline it makes single steps slightly worse (the best fit was 0.9), so try it only if the robot clearly turns too
late in the curves and loops.

## Score every run (write it down right after the run)

| Run label | Sheet | How it ended | Puck reached the line's end? | Puck stayed on the line? | Where it went wrong | Steps |
|---|---|---|---|---|---|---|
| | | | | | | |

"Where it went wrong", one of: left the line, stopped too early, did not stop at the end, stalled or went back and
forth, refused step, other (describe). A run spoiled by something that is not the model's doing (lost connection, a
person in view, a set-up mistake) is marked "void" and repeated.

## Known limits of this first version (from the code review of 11.11.26)

- The first refused step ends the run (no automatic fallback yet); a refusal near the end of a line is still a useful
  result.
- The workspace guard is the area all demonstrations covered, not the sheet in use; it does not stop the puck from
  being pushed off a small sheet. Watch the puck near the paper edge.
- Esc is read once per step, after the model's decision: a run ends at most one step after Esc.

## After the session

- [ ] Motors off before anyone touches the arm; park the arm.
- [ ] Copy the score table into the session notes (`05_meetings` or `STATUS.md`) and tell Claude the run folders' names,
      so the failures can be studied and corrections planned.
