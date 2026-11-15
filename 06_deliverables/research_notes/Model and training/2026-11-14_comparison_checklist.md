# Checklist: comparison at the robot, 14.11.26 (Emranul and Marshal)

Run list: `2026-11-14_comparison_runs.csv` (C01 to C12; Marshal's model is added as model M when its file and its
picture format are known). Every run with 0.65 cm steps, the same starts, lay-outs, faint pictures and placing aid as
on Thursday (`2026-11-12_test_visit_checklist.md` keeps the full procedure and the rules for scoring). Driver: the
`training-analysis` working copy (model_mode.py as frozen on 12.11, plus the previous picture for model 7 only).

## Bring

- Laptop and charger; the six grey course sheets (`course_sheets_original_from_moodle.pdf`, A4 at 100 %), at least
  sheets 01, 03 and 04 fresh; the puck; light tape; a ruler in mm; a phone for a photo with the ruler after each run.
- Marshal: his model file, the code that loads it, and how it reads a picture (size, colour order, normalisation, what
  it outputs).

## Safety, before the first run

1. Nothing else connected to the robot (close every viewer; one program at a time).
2. Stop button in reach; the robot away from the table edge; automatic calibration only.
3. Motors off (FreeMotion) before anyone touches the arm; never push on a held arm.
4. The camera and its focus ring are not touched.

## Order (each sheet taped once)

| Runs | Sheet | Models | Scored by |
|---|---|---|---|
| C01 to C06 | 03, first bend | E, N, E, N, E, N | within 12 mm up to 119 mm along |
| C07, C08 | 01, whole line | E, E | where the stop lands (17 / 12 / 7 mm) |
| C09 to C12 | 04, first sharp bend | E, N, E, N | within 12 mm up to 160 mm along |
| (M lines) | the same sheets | Marshal's model | the same |

N is model 7 (two pictures): it failed the offline rule on stops, so it runs only as an extra on the bend sheets.

## Each run

1. `python visit_runs.py "../06_deliverables/research_notes/Model and training/2026-11-14_comparison_runs.csv" C01`
   from `D:\AI_Project\.claude\worktrees\training-analysis\02_code`; read the line, Enter.
2. Lay the puck as the faint picture shows; place the tip with the placing line; Enter to go.
3. Esc ends a run (never Ctrl+C); one word back in the chat on how it went; photo with the ruler.
