# Checklist: a first look at model 10 on the robot (14.11.26, night; Emranul and Marshal)

Run list: `2026-11-14_night_look_runs.csv` (L01 to L06), every line checked with `--dry` at 23:40. Whole lines, 0.65 cm
steps, the driver as on 12.11 (`2026-11-12_test_visit_checklist.md` has the full procedure). A look, not a test: five
of the six runs are on tracks model 10 has trained on, so they say whether it drives and stops there at all, not how
it does on an unseen line (that needs a print of a development track D01 to D10).

## Before the first run

1. VPN off (it blocks the robot). The training on ki-014 goes on; only the download of results pauses until the
   fetch-only login afterwards.
2. Nothing else connected to the robot: the recording page is closed, no viewer open.
3. Stop button in reach; the robot away from the table edge; automatic calibration only.
4. Motors off (FreeMotion) before anyone touches the arm; never push on a held arm.
5. The camera and its focus ring are not touched.

## Runs

| Run | Sheet | Model | What to watch |
|---|---|---|---|
| L01 | T01 solid | X (model 10) | stays on the line; stops with the puck on the end |
| L02 | T01 solid | E (round 1 only) | the same, for comparison |
| L03 | T07 solid | X | the two corners; does it stop on the end in the middle of the paper? |
| L04 | T07 solid | E | the same, for comparison |
| L05 | T03 dashed | X | the dashed loop; the stop in the middle of the paper |
| L06 | sheet 04 | X | the first sharp bend, where the round 1 models failed |

## Each run

1. From `D:\AI_Project\.claude\worktrees\training-analysis\02_code`:
   `python visit_runs.py "../06_deliverables/research_notes/Model and training/2026-11-14_night_look_runs.csv" L01`;
   read the line, Enter.
2. Lay the sheet and the puck as the faint picture shows; place the tip with the placing line; Enter to go.
3. Esc ends a run (never Ctrl+C). If it drives past the end: Esc.
4. One line back in the chat: stayed on the line or where it left; stopped on the end, early, or not at all.
