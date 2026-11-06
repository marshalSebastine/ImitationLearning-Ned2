# How to start the tools yourself

Written 5.11.26 at Emranul's request, so the tools can be started without asking Claude each time.

Type or paste one line into the Terminal panel of the Claude app and press Enter. The lines use full paths, so they
work from any folder, in PowerShell and in Git Bash.

Before a robot tool: only one program may talk to the robot at a time. Close any other tool first (Ctrl-C in its tab).

## The recording page (easiest way to record, since 6.11.26)

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/recording_page.py

It opens a form in the browser (http://127.0.0.1:8765, reachable only from this computer). Fill in the episode, tick the
safety checks, press Start. It starts `record_episodes.py` with the right options, and the tool then moves the robot and
opens its windows as usual. "Check robot" and "Check the newest session" run `check_hardware.py` and `check_episodes.py`.

- All keys still go to the tool's picture window, Enter included. After Start the windows may be behind the browser.
- The page has no stop button, on purpose: use Esc at the "Put the puck at the start" prompt, OPTIONS on the gamepad, or the
  robot's Stop button.
- Choose a sheet and an episode from the list: the "Set up this episode" box then shows the end, kind of start, offset, angle, start
  index and start position, the plan's setup steps, and the note that will be saved. It warns if you change the end or kind of start
  away from the plan. The list starts on the next episode not yet done, in the plan's order: Round 1 (the first 12 of sheets 01, 02
  and 04), then the rest. A discarded episode, a score of 1 and a trial leave the episode open, so it comes up again. Choose "None"
  for a trial or practice.
- A ticked "Trial" is saved with the tool tests in `03_data/raw` (tool `--trial`), so the main data in
  `03_data/raw/main_data` holds only plan episodes; practice runs go to `03_data/raw_simulated`.
- Filled in for you, and you can still change it: operator and mode (your last choice; on a first visit the operator comes from
  the computer's user name), sheet and episode (as above), kind of start and end (from the plan), the taught start
  for that kind of start, and today's light note (not yesterday's, as the light may differ).
- Safety boxes: "Check robot" ticks the calibration box when the robot says it is calibrated, and that lasts 60 minutes. The Stop
  button box must be ticked for every episode. The other two stay ticked until the sheet changes or the page is reloaded.
- A puck that missed the goal fills in score 1 and "Did not finish"; change them if that is wrong. Other scores stay yours to give.
- Each Start records one episode, so each carries its own plan number; then the robot parks. For a finished episode the page asks
  for a score before the next Start: 3 good, 2 usable, 1 bad, with what went wrong. It goes into a new `review.json` in the
  episode's folder and a line in `session_notes.md`. Check it before saving: a saved score cannot be changed from the page.
  A discarded episode asks for no score, because the picture window already asked why.
- Keep the page's program running during a session, for its status and log. If it is closed, a running tool goes on, but start
  nothing else until that tool has finished: the restarted page does not know about it.
- It offers the gamepad and the two hand-guided modes, not the keyboard. The camera check of the first data day and other
  special runs still use the command lines below. Each run's messages are saved in `03_data/recording_page_logs/`.

## Record on track 4 (sheet `path_04_meander`)

Gamepad:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode gamepad --path-id path_04_meander --start-x 0.1707 --start-y 0.0374 --orientation "sheet 04 as on 2.11" --lighting "5.11, camera taped on the forearm cover"

Hand guiding with the exact replay:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode hand_guided_exact --path-id path_04_meander --start-x 0.1707 --start-y 0.0374 --orientation "sheet 04 as on 2.11" --lighting "5.11, camera taped on the forearm cover"

Keyboard teleop:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/record_episodes.py --operator emranul --mode keyboard --path-id path_04_meander --start-x 0.1707 --start-y 0.0374 --orientation "sheet 04 as on 2.11" --lighting "5.11, camera taped on the forearm cover"

What the parts mean:

| Part | Meaning |
|---|---|
| `--operator` | who records: `emranul` or `marshal` |
| `--mode` | `keyboard`, `hand_guided` (replay in 1 cm steps), `hand_guided_exact` (replay of every joint) or `gamepad` |
| `--path-id` | which sheet: `path_01_straight_line`, `path_02_arc`, `path_03_two_curve`, `path_04_meander`, `path_05_alpha_loop`, `path_06_alpha_loop_dashed` |
| `--start-x`, `--start-y` | where the puck starts, in metres from the robot's base; 0.1707 and 0.0374 is track 4's start on the old desk (taught 31.10.26; wrong on any other table). Without them the tool uses the line start in `config.py` |
| `--orientation`, `--lighting` | free notes stored with every episode; change the lighting note when the light changes |
| `--simulated` | add it to practise without the robot |

## The orientation window's puck (since 6.11.26)

The window shows the table from above. The red puck is drawn solid where the camera really sees it (paler when only a
sliver of it shows, so its place is rough), and the green area is what the picture shows on the table; both come from
the camera fitted on 6.11.26 to the episodes of 4 and 5.11. The line under the map says in words what you see ("Puck:
seen, drawn where it really is", "Puck: not found in the picture", ...). The picture window circles the puck it sees.
Nothing the robot does depends on this; it only helps you steer.

A hollow "puck?" is the tool's own guess of where the puck is, worked out from the steps taken. Keyboard forearm turns
(v / b) land back from this guess, so in the keyboard and 1 cm replay views it is shown whenever the camera cannot place
the puck, and also beside the seen puck when the two are more than 1.5 cm apart: then a turn lands from the guess, not
from the real puck. The gamepad view shows it only when the camera cannot place the puck; the exact replay and hand
guiding never show it (they do not use it).

The puck detector does not find a puck held in a hand. If the camera check says the camera moved (over 4 px) and you
record anyway, the map says "rough: camera moved"; the map is then off by a few millimetres or more: tell Claude.

## Check recorded episodes (no robot)

The newest session, or a session or episode folder you name; add `--report <file.md>` to keep the report:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/check_episodes.py
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/check_episodes.py D:/AI_Project/03_data/raw/2026-11-06_session_01_emranul

Tests of the tool without the robot (run them after every change to the code or `config.py`):

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/test_record_episodes.py
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/test_puck_tracking.py
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/test_forearm_camera_model.py
    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/test_recording_page.py

`test_puck_tracking.py` also reads recorded episodes from `03_data/raw`; on a laptop without them those tests say SKIPPED
and the last line counts them.

## Gamepad checks (no robot)

Live display, one line each time a stick, trigger or button changes:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/gamepad_live.py

Guided check, which asks for each stick, trigger and button and compares the numbers with `config.py`:

    D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/gamepad_check.py

## Stopping

- Esc at the "Put the puck at the start" prompt ends the session; the robot goes home.
- Gamepad mode: cross = stop the episode, circle = discard it, OPTIONS = quit.
- Ctrl-C stops any script, but the recording tool then sends the robot home. Use it only when going home is safe;
  otherwise press the robot's Stop button.
