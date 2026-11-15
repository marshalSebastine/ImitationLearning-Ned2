# Checklist: a first look at model 10 on the robot (14.11.26, night; Emranul and Marshal)

Run list: `2026-11-14_night_look_runs.csv` (S01 to S06, SE3, SE4, L01 to L05), every line checked with `--dry` at
23:40. Whole lines, 0.65 cm steps, the driver as on 12.11 (`2026-11-12_test_visit_checklist.md` has the full
procedure). A look, not a test: model 10 has trained on every sheet in the list, so the runs say whether it drives and
stops there, not how it does on an unseen line (that needs a print of a development track D01 to D10). The original
course sheets come first (Emranul, 14.11.26: "why don't we try on the og sheets?"), because earlier models were driven
on them on 11 and 12.11, so there is something to compare with.

## Before the first run

1. VPN off (it blocks the robot). The training on ki-014 goes on; only the download of results pauses until the
   fetch-only login afterwards.
2. Nothing else connected to the robot: the recording page is closed, no viewer open.
3. Stop button in reach; the robot away from the table edge; automatic calibration only.
4. Motors off (FreeMotion) before anyone touches the arm; never push on a held arm.
5. The camera and its focus ring are not touched.

## Runs (any order; each is one command)

| Run | Sheet | Model | What to watch | Earlier models on the robot |
|---|---|---|---|---|
| S01 | 01 straight line | X (model 10) | where the stop lands | 12.11: stopped 2.1 to 2.7 cm left |
| S02 | 02 arc | X | stays on the line, stops on the end | 11.11: reached the end |
| S03 | 03 two curves | X | the first bend | 11.11: lost the line there three times |
| S04 | 04 meander | X | the first sharp bend | failed on 11 and 12.11 |
| S05 | 05 loop | X | the crossing, the stop | 11.11: reached the end |
| S06 | 06 dashed loop | X | the dashes, the crossing | 11.11: lost the line |
| SE3, SE4 | 03, 04 | E (round 1 only) | the same bends tonight, for comparison | |
| L01, L02 | T01 solid | X, then E | a new track that ends near the paper's edge | |
| L03, L04 | T07 solid | X, then E | two corners; the stop in the MIDDLE of the paper | |
| L05 | T03 dashed | X | dashed loop; the stop in the middle of the paper | |

## Each run

1. From `D:\AI_Project\.claude\worktrees\training-analysis\02_code`:
   `python visit_runs.py "../06_deliverables/research_notes/Model and training/2026-11-14_night_look_runs.csv" S01`;
   read the line, Enter.
2. Lay the sheet and the puck as the faint picture shows; place the tip with the placing line; Enter to go.
3. Esc ends a run (never Ctrl+C). If it drives past the end: Esc.
4. One line back in the chat: stayed on the line or where it left; stopped on the end, early, or not at all.
