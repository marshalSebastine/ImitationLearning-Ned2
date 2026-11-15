---
name: project-test-visit-2026-11-12
description: 12.11.26 evening, Emranul at the robot - 36 model-mode runs (R00 to R31, X01 to X03, one loop demo); by his eye shorter steps (0.65 cm) kept sheet 03's first bend more reliably than 1 cm, the team of five was bad, sheet 04 failed again; desk scoring from the pictures still open
metadata:
  type: project
---

12.11.26, about 17:40 to 19:40, Emranul alone at the robot (a friend watching), guided run by run in the chat by the
Model and training session. Driver: the frozen working copy `.claude/worktrees/visit-frozen-2026-11-12` at commit
9725336 (model mode with the switches of 12.11, started through `visit_runs.py` with the run list
`06_deliverables/research_notes/Model and training/2026-11-12_test_visit_runs.csv`; checklist beside it). Logs:
`03_data/model_runs/2026-11-12_*`, 36 folders. Sheets used: fresh inkjet copies of sheets 01, 03 and 04, redrawn from
the tracing (the originals are torn, Emranul 12.11.26), see [[project-new-tracks-2026-11-12]].

What Emranul reported (his eye and his ruler; NOT yet scored from the pictures by the rules written in the checklist):

- Loop demo, model E on sheet 05, start as on 11.11: whole loop, 88 pictures, stopped by itself (as on 11.11).
- Sheet 01, model E, 1 cm steps. New start (tip at the recorded start with the placing line, sheet laid by the faint
  picture; R00, R02, R03, R04): stopped by itself 2 cm left of the line's end, four times, 28 to 29 pictures. Start as
  on 11.11 (tip just behind the puck; X01 to X03): 1 to 1.5 cm left, three times, 26 pictures. So the control did not
  "finish" by the 17 mm rule, and the start changes where the stop lands.
- R01 (own step length, no puck): small steps are smooth on the real arm; it ended "refused" after 20 pictures.
- Sheet 03, first bend, short runs. Model E in five pairs: 1 cm = barely touching, almost good, lost (cut across),
  "a bit faster" (line not reported), mostly on; 0.65 cm = on the line or close to it all five times. The model
  trained without sheet 03 in three pairs: 1 cm = lost after 4 to 5 steps, off early then barely touching, about as its
  partner; 0.65 cm = best run of the evening, slow but on the line, mostly on.
- Team of five, 1 cm (R21, R23, R25, R27): bad four times. Model E with its own step length (R22, R24, R26, R28,
  90 steps): once almost the whole track (ended 3 to 4 cm before the finish), twice ended early on the left, once
  mostly off the line.
- Sheet 04, model E, 1 cm (R29 to R31): lost after 4 to 5 steps all three times. In R29 the sheet lay turned about
  10 degrees against the recorded picture [seen by Claude in the first picture]; for R30 Emranul turned it onto the
  faint line; whether it matched is to be checked in the pictures.

How it went as a procedure: every start placed with the placing line was within 1 cm (0.1 to 0.93 cm). Emranul found
the faint-picture step hard to understand at first; he works best with three or four numbered lines per run and one
word back. The app's terminal panel was 56 columns wide at first and cut the placing line off. Claude may hold only six
terminal tabs of its own; Emranul closed old ones.

**Scored from the pictures on 13.11.26** (`06_deliverables/research_notes/Model and training/2026-11-13_test_visit_scores.md`,
evidence in `.../evidence/test_visit_2026-11-12/`; puck against the printed drawing laid on each picture's ink):
- Sheet 03, model E's five pairs: 1.0 cm kept the line in 1 (2 doubtful), 0.65 cm in 3 (2 doubtful). The rule needs
  3 more for 0.65 cm, so by the rule written beforehand 0.65 cm has NOT passed (it would only if all doubtful runs fell
  its way). Own step length kept 4 of 4, which by the rule lets it go on, but only because two 1.0 cm runs count as
  lost by 0.4 to 5 mm: not to be acted on without repeating.
- Model trained without sheet 03: 1.0 cm 0 of 3, 0.65 cm 3 of 3 (sure). Team of five: 0 of 4.
- Sheet 04: 0 of 3, lost at the first sharp right bend (72 to 63 mm along); R30 lay on its faint copy within 2 mm,
  so the failure is not the placement.
- Sheet 01: every stop 21 to 27 mm left of the end's centre, old and new start alike (Emranul's ruler said 1 to 1.5 cm
  for the old start); 0 of 6 finished. The drift starts about 130 mm along. Sheet 01 lay about 1.6 cm beside its faint
  copy from R00 on.
- Emranul's eye and the measure disagree on R06, R12 (worse than seen) and R20.

Still to do: check the doubtful runs against the phone photos, decide the step length (12 mm rule, the 119 mm and 160 mm
marks), decide the step length from the five pairs of model E, look into the stop landing left of the end on sheet 01,
remove the frozen working copy (10 GB of copied models) when the scoring is done.
