# Working rules for Claude in this project

Start by reading `HANDOFF_2026-10-26.md` (state of the project) and `README.md` (folder map and rules).

- The user may be a beginner: one step at a time, plain language, propose and explain before doing; ask before anything runs
  on the robot. Write a checklist before any session at the robot.
- Test before any physical change to the robot or its camera. Say what is measured and what is calculated or guessed. Never
  send a drawing of a part that has not been checked against the real geometry (Niryo's STEP and STL files are in
  `06_deliverables/research_notes/Camera mount options/niryo_files/`).
- The camera and its focus ring are not to be touched except as the camera-mount checklist plans. Only the automatic
  calibration after a restart, never manual. Motors off (FreeMotion) before anyone touches the arm; never push on a held arm.
  The robot stays away from the table edge.
- Only one program may be connected to the robot at a time; close any viewer before running another script.
- Git: no Claude attribution of any kind (no Co-Authored-By, no "generated with" lines); commits under the students' own
  names.
- Python follows the `emran-code-style` skill: long names, spaces around "=" in keyword arguments, no import aliases,
  dictionary-based parameters merged over defaults, numpy docstrings, British spelling.
- Settings live in `02_code/config.py` only. Nothing in `01_course` or `03_data/raw` is edited, renamed or deleted, except
  `01_course/moodle_snapshot.md` and a session's `session_notes.md`, which may be appended to.
- Hard constraint from the course: imitation learning only; the robot's motion comes from a model trained on our own
  demonstrations, with the end-effector camera as the only input.

## Shared memory and two Claude accounts

- Emranul and Marshal work with separate Claude accounts. The GitHub repo https://github.com/marshalSebastine/ImitationLearning-Ned2
  is the only channel between their sessions: code, notes and Claude's project memory all live in it.
- Project memory lives in `claude_memory/` (index below). `setup_claude.sh` (macOS) or `setup_claude.ps1` (Windows) links
  Claude's own memory folder for this project to it, so new memories land in git. If the link is missing, write project
  memories into `claude_memory/` directly. Name who said or decided what, never "the user".
- Start of a session: `git pull`. End of a piece of work: commit (memory changes included) and push, after asking.
- When a session ends with something the other person must know (state, decision, open question), add it to the memory
  or to `05_meetings` notes, not only to the chat.

@claude_memory/MEMORY.md
