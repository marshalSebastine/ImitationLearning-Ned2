# The recording program and the new tracks (for the recording of 14.11.26)

Written 12.11.26. What was changed so that the recording page offers the 16 new training tracks (T01 to T16, each
printed solid and dashed) exactly as it offers the six course sheets, what a student does differently, and what the
training code will have to learn later. Nothing here was tried at the robot: every check ran without it.

## What was changed

- `02_code/config.py`
  - `NEW_TRACK_RECORDING_PARAMETERS` (new, after `step_budget_for`): the 16 tracks (name, label, line length, standard
    starts per print, which print has a near-end start), the mark of the lay-out rule, the plan notes. The function
    `add_new_track_sheets` adds both prints of every track to the page's sheets, set-up notes and episode plan, and their
    line lengths to the step budgets. `new_track_sheet_id` names a print.
  - `SHAPE_OF_SHEET` (new): sheet name -> the shape it carries. Both prints of a track share one shape ("T03"); each
    course sheet is its own ("01" to "06").
  - `RECORDING_PAGE_PARAMETERS["first_round"]`: now the pilot (plans 1 and 2 of T01 to T04 solid). It named the first 12
    episodes of sheets 01, 02 and 04, all of which were done (210 of 210 course-sheet episodes done on 12.11.26).
  - `RECORDING_PARAMETERS`: `near_puck_velocity_percent` (90) and `is_lift_after_guiding_by_joints` (True), see below.
- `02_code/setup_visuals.py`: the set-up pictures of the new tracks, drawn from `tracks.json` by its lay-out rule, and
  `table_marks.pdf`. `python setup_visuals.py --new-tracks` draws only these; the course sheets' pictures stay as they are.
- `02_code/recording_page.py`, `record_episodes.py`, `teach_positions.py`, `check_episodes.py`: the data folder (03_data)
  is the main checkout's when the program runs from a git worktree (`config.main_checkout_folder()`); from the main
  checkout nothing changes.
- `02_code/record_episodes.py`: the moves beside the puck that push nothing run at `near_puck_velocity_percent`, and
  after hand guiding the arm lifts with joints 2 and 3 at once (Emranul, 12.11.26).
- Tests: `test_recording_page.py`, `test_setup_visuals.py`.
- Pictures: `06_deliverables/research_notes/Recording tool/figures/setup_per_episode/plan_11-01.png` to `plan_46-01.png`
  (58), `table_marks.pdf`, and `index.html` (rewritten with the new sheets).

## Where the page is started from

From this working tree, which has the new sheets, pictures and settings:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/.claude/worktrees/training-analysis/02_code/recording_page.py

Recordings, trials, practice runs, the page's logs, the camera reference pictures, taught positions and the scan for
plans already done all go to the main checkout's `D:/AI_Project/03_data` (checked by printing the resolved paths; nothing
was written there). Code, set-up pictures and the code version (`code_commit` in episode.json) come from this working
tree, so commit the changes before Saturday. Never make a folder `03_data` inside the working tree: the program would
then take it for the data folder. The main checkout's own `02_code` cannot simply take these files: its config.py has no
`main_checkout_folder()` and still limits the forearm to 90 degrees (this tree: 117).

## The new sheets on the page

| print | sheet number | sheet name on the page | folder name begins |
|---|---|---|---|
| T01 solid ... T16 solid | 11 ... 26 (10 + track) | "T01 gentle S-curves, solid" ... | `path_11_T01_solid_episode_...` |
| T01 dashed ... T16 dashed | 31 ... 46 (30 + track) | "T01 gentle S-curves, dashed" ... | `path_31_T01_dashed_episode_...` |

Plan numbers are the sheet number and the episode: 13-01 is the first episode on the solid print of T03, 33-01 the
first on its dashed print. The set-up picture of 13-01 is `plan_13-01.png`.

## How a student picks a new track

Nothing to pick in most cases: after each scored episode the page chooses the next episode by itself. It keeps the
sheet on the table until all its episodes are done, so every print is taped down once. The order (58 episodes):

1. The pilot: T01 solid (11-01 Emranul, 11-02 Marshal), T02 solid (12-01 Emranul, 12-02 Marshal), T03 solid
   (13-01, 13-02), T04 solid (14-01, 14-02). The sheet stays taped while the two change places. The page does not
   switch the Operator box by itself: set it to the person the plan names ("Pilot: recorded by Emranul" or "by Marshal"
   under "Set up this episode").
2. T01 dashed (31-01), T02 dashed (32-01), T03 dashed (33-01), T04 dashed (34-01).
3. T05 solid, T05 dashed, T06 solid, T06 dashed, and so on to T16 dashed, each with all its episodes; on a print with a
   near-end start (T06, T08, T10, T12, T14 solid; T05, T07, T09, T11, T13 dashed) that start comes last.

To choose by hand: open the "Sheet" list and pick, for example, "T05 loop and corner, dashed"; the page then offers that
print's next episode not yet done.

## What is different from a course sheet

- The sheet is laid on the NEW-TRACK MARK (table_marks.pdf), not on the course sheets' start mark: the centre of end A's
  round cap on the mark, 19.5 cm in front of the robot's base axis and 1.5 cm to its right, the first straight piece
  along the direction line, the sheet not turned. Each set-up picture gives the paper corners' distances from the mark.
- The recorder itself does nothing different: the arm comes down at its usual start (magnet tip at 14.0, -1.5 cm),
  about 5.5 cm behind the cap's centre, as in the 209 course-sheet recordings, where the caps lay a median 19.2 cm out
  (measured in the check of 11.11.26). In the hand-guided modes the hand brings the block up to the puck, as before.
- Every new track starts from end A. A near-end start (NE) is taught once per print with "Teach start point", as on the
  course sheets; the puck goes 6 cm before the end cap.
- Only in the mode "Hand guided, replay in 1 cm steps" (not the main mode, which replays every joint): the recorder
  refuses a step that would take the puck outside the box of sheet 01 as taught on 6.11.26 plus 2 cm (x 13.0 to 46.5 cm,
  y -15.2 to +10.0 cm) or outside the workspace box (y up to +13.0 cm). T01 (62 mm of line), T04 (5 mm) and T12
  (310 mm) leave that box (calc). In "Hand guided, replay of every joint" there is no such check.

## Faster moves beside the puck (Emranul, 12.11.26)

"make sure to increase that thing from 20% to 90%, I will take care of the puck". Two settings in
`RECORDING_PARAMETERS`: `near_puck_velocity_percent` 90 (20 gives back the old behaviour) for the lift off the puck or the
line and the last move down at the start or after a correction; `is_lift_after_guiding_by_joints` True (False gives
back the old order), so that after hand guiding the arm lifts with joints 2 and 3 at once instead of first asking three
times for the straight lift. Every pushing move, the step-over and the hand flick keep their speed. Not tried at the
robot; the time saved is a guess until timed there. What to watch at the first real episode:

- The arm leaves the puck quickly after the stop and after guiding: hands away from the arm when the motors come on.
- The last move down at the start ends 5 mm above the paper, now at up to 90 %: the puck must not stand under the block.
- The taped camera can shift under quick starts and stops: read the camera check line ("... px from the reference") of
  the next episode's first step; above 4 px the recorder asks whether to discard.

## What the training code will have to learn (not changed)

- `training_dataset.py` line 259: `"sheet": path_id[:7]` makes "path_12" and "path_32" two different sheets, although
  they are the same shape. Use `config.SHAPE_OF_SHEET` for anything that groups by shape.
- `training_targets.leave_one_sheet_out_folds`: one fold per value of "sheet"; with the new prints this gives 38 folds,
  and the solid and dashed print of one track would sit on both sides of a fold. Fold by shape (and decide whether the
  new tracks are held out at all).
- `training_probes.py` (docstring and folds: "five folds spread over all six sheets"), `config.py`
  `TRAINING_COMPARISON_PARAMETERS["matched_folds"]` (`without_path_03`, `_05`, `_06`), `MODEL_MODE_PARAMETERS["replay_fold"]`
  (`without_path_01`), `train_policy.py` (`--folds without_path_04 ...`): fold names built from the six sheets.
- `training_dataset.py` lines 133 to 138 read the start kind per plan number from the episode plan: these work for the
  new plan numbers as they are.
- Dashed or solid: the sheet name ends in `_dashed` or `_solid`; course sheet 06 is the only dashed course sheet.
