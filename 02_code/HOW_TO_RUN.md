# How to start the tools yourself

Written 5.11.26 at Emranul's request, so the tools can be started without asking Claude each time.

Type or paste one line into the Terminal panel of the Claude app and press Enter. The lines use full paths, so they
work from any folder, in PowerShell and in Git Bash.

Before a robot tool: only one program may talk to the robot at a time. Close any other tool first (Ctrl-C in its tab).

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
| `--start-x`, `--start-y` | where the puck starts, in metres from the robot's base; 0.1707 and 0.0374 is track 4's start, taught on 4.11.26. Without them the tool uses sheet 01's line start |
| `--orientation`, `--lighting` | free notes stored with every episode; change the lighting note when the light changes |
| `--simulated` | add it to practise without the robot |

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
