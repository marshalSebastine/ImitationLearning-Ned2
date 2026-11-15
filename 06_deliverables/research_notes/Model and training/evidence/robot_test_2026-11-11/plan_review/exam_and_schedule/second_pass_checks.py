"""
Plan review (key: exam_and_schedule), second pass: the numbers the first pass had not worked out yet.

Reads the plan and the verifier's stop counts; everything else is calculated on paper from numbers measured by the
first pass (timing_checks.out.txt). Nothing connects to the robot, nothing is trained, no git command is run.
Run:  D:/AI_Project/.venv/Scripts/python.exe second_pass_checks.py > second_pass_checks.out.txt
"""

import datetime
import math
import re
from pathlib import Path

REVIEW_FOLDER = Path(__file__).resolve().parent
NOTES_FOLDER = REVIEW_FOLDER.parents[3]
PLAN_FILE = NOTES_FOLDER / "2026-11-11_plan_after_first_robot_test.md"


def binomial(count: int, successes: int, chance: float) -> float:
    return math.comb(count, successes) * chance ** successes * (1 - chance) ** (count - successes)


def recording_hours() -> None:
    """The plan's "about 48 episodes, about 6 robot hours" under four ways of counting, all from measured unit times."""

    print("=== step 3, new shapes: 12 tracks x 4 episodes (two lay-outs, a solid and a dashed print); the plan says about 6 hours ===")
    episodes = 48
    plain = episodes * 5.8 / 60
    print(f"  48 x 5.8 min (the median gap inside one sheet; no new lay-out, no discarded episode, no break): {plain:.1f} h")
    verifier = (episodes * 5.8 + 24 * 10) / 60
    print(f"  the orientation verifier's own formula (5.8 min an episode plus 10 min a new lay-out; the 10 min is the analyst's guess) with 24 lay-outs: {verifier:.1f} h")
    first_five_means = [20.9, 11.5, 9.9, 9.5, 5.8, 5.8]
    first_five = sum(first_five_means) / len(first_five_means)
    print(f"  each of the 4 episodes of a new shape is among its first five: measured mean of the first five kept episodes of the six sheets "
          f"{first_five:.1f} min -> {episodes * first_five / 60:.1f} h")
    whole_sitting = episodes * 8.13 / 60
    print(f"  48 x 8.13 min (all sittings of 6 to 9.11.26 divided by the kept episodes, breaks and discards inside): {whole_sitting:.1f} h, before any extra lay-out")
    print("  so about 8.5 to 13 hours, not 6 [calculated from measured times; the 10 min a lay-out is a guess]")


def session_2_shares() -> None:
    """How many of session 2's runs measure the exam case (a line the model never trained on)."""

    print("")
    print("=== step 2: which runs of session 2 drive a line the model has not trained on ===")
    blocks = {"A (model E on sheet 03, a sheet it trained on)": (12, 0), "B (the model without sheet 03 on sheet 03; model E on two new tracks)": (12, 12),
              "C (model E on the shapes of sheets 05 and 06, which it trained on)": (8, 0), "D (model E on sheet 02 lying square, recorded six times)": (4, 0)}
    total = sum(count for count, _ in blocks.values())
    unseen = sum(count for _, count in blocks.values())
    for name, (count, unseen_count) in blocks.items():
        print(f"  block {name}: {count} runs, {unseen_count} on an unseen line")
    print(f"  {unseen} of {total} runs ({100 * unseen / total:.0f} %) measure the exam case; only 6 of them use a provided sheet as an unseen one (sheet 03)")


def block_a_undecided() -> None:
    print("")
    print("=== block A: the rule needs fixed 1.0 cm to pass at most 1 of 4 ===")
    for chance in [0.2, 0.33, 0.5]:
        two_or_more = sum(binomial(4, successes, chance) for successes in range(2, 5))
        print(f"  true pass chance of fixed 1.0 cm {chance:.2f}: it passes 2 or more of 4 with chance {two_or_more:.2f}; then no new rule can be taken on, however well it did")
    print("  on 11.11.26 model E with fixed 1.0 cm passed this bend 0 times in 2 (runs 5 and 9); model B passed it once (run 10)")


def stop_on_unseen() -> None:
    """The verifier's counts (stop_and_centring/verify/check_early_and_never_stops.out.txt): never, earlier, within the last 8 pictures."""

    print("")
    print("=== the stop on sheets the model never trained on (two pictures in a row at 0.5), offline on the demonstrators' pictures ===")
    recipes = {"E recipe": {"01": (1, 3, 16), "02": (29, 4, 12), "03": (8, 3, 34), "04": (6, 9, 30), "05": (12, 11, 6), "06": (15, 2, 8)},
               "E recipe, seed 2": {"01": (1, 2, 17), "02": (25, 6, 14), "03": (1, 9, 35), "04": (2, 13, 30), "05": (7, 14, 8), "06": (15, 5, 5)},
               "A recipe": {"01": (2, 1, 17), "02": (29, 4, 12), "03": (13, 4, 28), "04": (11, 9, 25), "05": (7, 12, 10), "06": (20, 0, 5)}}
    for name, table in recipes.items():
        shares = {sheet: right / (never + early + right) for sheet, (never, early, right) in table.items()}
        never_all = sum(never for never, _, _ in table.values())
        early_all = sum(early for _, early, _ in table.values())
        right_all = sum(right for _, _, right in table.values())
        total = never_all + early_all + right_all
        print(f"  {name}: right {right_all} of {total} ({100 * right_all / total:.0f} %), never {100 * never_all / total:.0f} %, too early {100 * early_all / total:.0f} %; "
              f"mean of the six sheets {100 * sum(shares.values()) / 6:.0f} %; by sheet " + ", ".join(f"{sheet}: {100 * share:.0f} %" for sheet, share in shares.items()))
    print("  block B drives only sheet 03 as an unseen provided sheet: the sheet where the unseen stop is second best (76 %); the worst are 05 (21 %), 02 (27 %) and 06 (32 %)")
    print("")
    print("  what the plan's recording adds to the stop, by the fit of report 8.2 (+8.7 points per doubling of stop pictures; frozen features, an extrapolation):")
    stop_rows_now = 259
    for label, new_stop_pictures in [("10 line-end episodes", 10), ("30 line-end episodes", 30), ("all of step 3 at its smallest (48 + 30 + 10)", 88),
                                    ("all of step 3 at its largest (48 + 30 + 30)", 108), ("a doubling of the stop pictures", 259)]:
        points = 8.7 * math.log2((stop_rows_now + new_stop_pictures) / stop_rows_now)
        print(f"    {label}: {points:+.1f} points, {106 + 209 * points / 100:.0f} of 209")
    print("    a doubling needs 259 more episodes with an end; as near-end starts at 2.4 min each (report 8.4) that is about "
          f"{259 * 2.4 / 60:.0f} robot hours")


def professor_questions() -> None:
    """Count what section 7 of the plan asks."""

    text = PLAN_FILE.read_text(encoding = "utf-8")
    section = text.split("## 7. Questions for the professor, to send now")[1].split("## 8.")[0]
    items = re.split(r"\n(?=\d+\. )", section.strip())
    print("")
    print("=== section 7: how much is asked ===")
    total_marks = 0
    total_words = 0
    for item in items:
        marks = item.count("?")
        clauses = marks + item.count(";")
        words = len(item.split())
        total_marks += marks
        total_words += words
        print(f"  question {item.split('.')[0]}: {words} words, {marks} question marks, {clauses} clauses that each want an answer (question marks plus semicolons)")
    print(f"  in all {len(items)} numbered questions, {total_words} words, {total_marks} question marks")
    print("  the team's meetings with the professor are ten minutes long (09:50 to 10:00; claude_memory/user-ai-project-team.md)")


def tool_days() -> None:
    """Step 1's non-robot work, added up from the efforts the analysts and verifiers gave (check_results_complete.md)."""

    print("")
    print("=== step 1: 'about two to three days of this session's work' against the analysts' own efforts (days of 8 hours) ===")
    items = [("the scoring command (analyst: about one day; verifier: optimistic for about 2,000 lines)", 1.0, 2.0),
             ("driver switches with tests (step-length analyst: half a day; twist analyst: 2 to 3 hours)", 0.5, 0.9),
             ("the controlled start: a new robot movement with tests and its own checklist (no estimate anywhere; a guess)", 0.5, 1.0),
             ("print set: new drawing rule and tracks.json (analyst: about an hour), the two prints of sheets 05 and 06 traced from JPEG photos, true-size PDF (a guess)", 0.5, 1.0),
             ("the two checks without the robot (dashed copies of run 7; whole-episode stop report)", 0.25, 0.5),
             ("the checklist and the run table of session 2 (bends analyst: half an hour for 18 runs; here 36 runs in six blocks)", 0.25, 0.5)]
    low = sum(item[1] for item in items)
    high = sum(item[2] for item in items)
    for name, item_low, item_high in items:
        print(f"  {name}: {item_low} to {item_high}")
    print(f"  sum {low:.1f} to {high:.1f} days; and every step is first proposed to Emranul and explained (CLAUDE.md), which the sum leaves out")


def calendar() -> None:
    print("")
    print("=== the calendar ===")
    today = datetime.date(2026, 11, 11)
    for name, day in [("meeting 3", datetime.date(2026, 11, 23)), ("the plan's driver freeze", datetime.date(2026, 12, 6)), ("meeting 4", datetime.date(2026, 12, 7)),
                      ("the plan's model freeze", datetime.date(2026, 12, 13)), ("robot work ends", datetime.date(2026, 12, 20)), ("our freeze", datetime.date(2027, 1, 8)),
                      ("the presentation with the live run", datetime.date(2027, 1, 11))]:
        print(f"  {name}: {day.strftime('%A %d.%m.%Y')}, {(day - today).days} days after 11.11.26")
    print(f"  from the end of robot work to the live run: {(datetime.date(2027, 1, 11) - datetime.date(2026, 12, 20)).days} days without a single run in the plan")
    print(f"  first frozen model in the plan: 13.12.26, {(datetime.date(2026, 12, 13) - today).days} days from now and {(datetime.date(2026, 12, 20) - datetime.date(2026, 12, 13)).days} days before robot work ends")
    print("  a first session that needs no new tool and no print: 4 provided sheets as unseen ones (the models trained without them) x 2 step rules x 3 runs = 24 runs; "
          f"at 3.93 to 5.5 min a run {24 * 3.93 / 60:.1f} to {24 * 5.5 / 60:.1f} h, plus the dry run")


def paired_rule() -> None:
    """A rule for the step length on unseen sheets: 12 pairs of runs, the new setting ahead in at least 9."""

    print("")
    print("=== a paired rule in place of block A: 4 unseen provided sheets x 3 pairs, one run with each setting, the better run of a pair counted ===")
    for chance in [0.5, 0.65, 0.75, 0.8, 0.9]:
        taken_on = sum(binomial(12, successes, chance) for successes in range(9, 13))
        print(f"  chance that the new setting is the better run of a pair {chance:.2f}: ahead in 9 or more of 12 pairs with chance {taken_on:.2f}")
    print("  0.50 is 'no difference': the rule then takes the new setting on 7 times in 100; block A's rule finds a true 0.9 against 0.33 only 57 times in 100")


def run_minutes() -> None:
    """Minutes per run of the first session, from the measured 1.04 s a move and the measured gap between runs."""

    print("")
    print("=== minutes per full run on the four sheets (median demonstrated steps; 1.04 s a move; 3.0 min between the end of a run and the next start, measured 11.11.26) ===")
    steps = {"02": 66, "03": 106, "05": 103.5, "06": 82}
    fixed_total = 0.0
    own_low_total = 0.0
    own_high_total = 0.0
    for sheet, count in steps.items():
        fixed = count * 1.04 / 60 + 3.0
        own_low = count * 1.04 * 1.3 / 60 + 3.0
        own_high = count * 1.04 * 1.54 / 60 + 3.0
        fixed_total += 3 * fixed
        own_low_total += 3 * own_low
        own_high_total += 3 * own_high
        print(f"  sheet {sheet}: fixed 1.0 cm {fixed:.1f} min; the model's own length {own_low:.1f} to {own_high:.1f} min")
    print(f"  24 runs: {(fixed_total + own_low_total) / 60:.1f} to {(fixed_total + own_high_total) / 60:.1f} h; with the dry run (about 20 min) and a photo at each stop (about 1 min each) "
          f"{(fixed_total + own_low_total) / 60 + 0.33 + 0.4:.1f} to {(fixed_total + own_high_total) / 60 + 0.33 + 0.4:.1f} h")


if __name__ == "__main__":
    recording_hours()
    session_2_shares()
    block_a_undecided()
    stop_on_unseen()
    professor_questions()
    tool_days()
    calendar()
    paired_rule()
    run_minutes()
