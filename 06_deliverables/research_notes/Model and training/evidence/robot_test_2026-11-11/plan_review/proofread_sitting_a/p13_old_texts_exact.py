"""
Proofreader sitting_a, check 13: every "old text" of my corrections must stand in version 2 exactly once, on one line.

Run:  D:/AI_Project/.venv/Scripts/python.exe -B p13_old_texts_exact.py > p13_old_texts_exact.out.txt
"""

from pathlib import Path

PLAN = Path("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/2026-11-11_plan_after_first_robot_test.md")
text = PLAN.read_text(encoding = "utf-8")
lines = text.splitlines()
OLD = {
    "C01": "**Sitting A: the small steps, the measure, the driver's step (about 2.5 hours; sheets 01, 03, 04 and the circle print)**",
    "C02": "The sheet lies on tape marks;",
    "C03": "and the puck ahead of the tip as in the recordings.",
    "C04": "tool makes it (lift, travel 5 cm above, straight down);",
    "C05": "a fixed list of answers (ten steps each of 1, 2 and 3 mm, five twists in place of 5 and of 10 degrees), first without the puck, then with the puck on sheet 01",
    "C06": "fewer than 2 finished: stop and repair the start before anything else",
    "C07": "until the puck has passed 119 mm of line (8 cm past the bend).",
    "C08": "Short runs end by the travel budget;",
    "C09": "the puck set by hand into the grey circles (on the line, 7, 12 and 17 mm beside it, at the bend, at the end, far out, forearm turned), the arm held still",
    "C10": "model E, the driver of 11.11, sheet 04 on tape marks at its main recorded place (plans 04-01, 04-02, 04-05), 3 short runs to past the first hairpin; one full run only if the bends are passed",
    "C11": "through the first S-bend in at least 2 of 3 runs: run 6 was mostly start and placement. If not: the same three runs with the step A4 chose, before anything is recorded for sheet 04",
    "C12": "The own length replaces 0.65 cm only if it keeps the line in all 4 runs while 0.65 cm loses it in 2 or more of 5. If 0.65 cm keeps the line in fewer runs than 1.0 cm with the model that never saw the sheet, both steps go to sitting B",
    "C13": "on two pictures in a row,",
    "C14": "against a new reference picture saved on the first day (warn above 4 px, stop above 10 px);",
    "C15": "model E, sheet 01, the driver of 11.11, 3 runs",
    "C16": "| 3 | Go for the tools and the checklist of sitting A, and the day at the robot; the new start move of the arm is part of that checklist | Emranul |",
    "C17": "a snapshot that saves a picture and moves nothing;",
    "C18": "version into the run's summary.",
    "C19": "no refused step; the tip moves 10, 20 and 30 mm, each within 10 %; the puck moves the same within 2 mm and less than 2 mm under the twists. If the 1 mm group fails the driver gets a smallest step of 2 mm",
    "C20": "within 3 mm at 18 of 20 places and never more than 5 mm off;",
    "C21": "or for safety. Every run gets one label: finished, lost,",
    "C22": "picture, a set-up mistake or something in the camera's view.",
    "C23": "ruler in view is taken while the arm waits above the start.",
    "C24": "### Step 1. Tools before sitting A (this session, one to two days, no robot)",
    "C25": "one dry run to a sheet's far end in sitting A",
}
for key, old in OLD.items():
    count = text.count(old)
    on_lines = [number + 1 for number, line in enumerate(lines) if old in line]
    print(f"{key}: found {count} time(s), whole on line(s) {on_lines}  {'OK' if count == 1 and len(on_lines) == 1 else '<-- CHECK'}")
