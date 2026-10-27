# How Team M works together in this repository

Emranul (Windows ThinkPad, `D:\AI_Project`) and Marshal (MacBook, `~/Documents/AI_Project`) each work with their own Claude
account. The two Claude sessions share no chat history: this repository is the only channel between them. Code, notes,
decisions and Claude's project memory all go through git. Claude reads this file at the start of every session (`CLAUDE.md`
points here).

Repository: https://github.com/marshalSebastine/ImitationLearning-Ned2 (public). GitHub users: `marshalSebastine`,
`emrantawsif`.

## Branches

- `main` is the shared state of the project. It should always be in a state the other person can pull and run.
- Work happens on short-lived branches named `<person>/<topic>`, for example `marshal/recording-tool`,
  `emranul/camera-mount-measurements`. One branch per piece of work, lived for days, not weeks.
- A branch goes into `main` through a pull request on GitHub. The author merges it once the other person has had a look,
  or after 24 hours without objection. Small fixes to notes or docs may be merged at once.
- Never force-push to `main`, never rewrite history that is already on GitHub.

Why branches and not both committing straight to `main`: robot code is tested on hardware, and a half-finished script on
`main` could stop the other person's robot session. Branches keep unfinished work apart; pull requests leave a visible
record of who did what, which matters because the course grades each student's identifiable share.

## Rhythm of a session

1. Start: `git switch main`, `git pull`, then either `git switch -c <person>/<topic>` for new work, or switch to your
   branch and run `git merge main` to bring in the other person's work (and their Claude memory).
2. During: commit in small steps with clear messages in English.
3. End (every session, even unfinished): commit and push the branch (`git push -u origin <branch>`). If something is
   finished, open the pull request. Claude asks before pushing or merging.

## Claude's shared memory

- `claude_memory/` is Claude's project memory. `setup_claude.sh` / `setup_claude.ps1` links Claude Code's own memory folder
  for the project to it, so whatever Claude remembers shows up as a changed file in git.
- Memory changes are committed on the branch you are on and reach the other person when that branch is merged. If the
  other person needs a fact sooner (a hardware change, a rule from the professor, a deadline), commit just that memory file
  straight to `main`: `git stash`, `git switch main`, `git pull`, commit and push the file, `git switch -`, `git stash pop`,
  `git merge main`.
- Write memory so the other person's Claude understands it: say who said or decided what ("Marshal said …"), absolute
  dates, no "the user".
- Personal memory that only concerns one person's own tools goes in a file named `claude_memory/personal-<name>-<topic>.md`.
  Git ignores those files; do not add them to `MEMORY.md`.
- Merge conflicts in `MEMORY.md`: keep both sides' lines.

## Who owns what (provisional, from the handoff)

- Emranul: robot sessions, camera-mount measurements, mock-up and Maker Space contact
  (`06_deliverables/research_notes/Camera mount options/`, `05_meetings/`).
- Marshal: recording tool and dataset pipeline (new scripts in `02_code/`), GPU-pool access.
- Both: demonstrations, meeting notes, contribution log.
- `02_code/config.py` is shared by everything; say in the commit message which settings changed and why.

## Rules that do not change

- Commits under your own name and e-mail (`git config user.name`, `git config user.email`); no Claude attribution of any
  kind (no Co-Authored-By, no "generated with" text).
- `03_data/` (recordings) is never in git. It is too large and is backed up separately. How the two of us share raw
  recordings is still open (a shared cloud folder is the likely answer).

## Setting up a new laptop

1. `git clone https://github.com/marshalSebastine/ImitationLearning-Ned2.git <folder>`, then `cd <folder>`.
2. `git config user.name "<Your Name>"` and `git config user.email "<your e-mail>"` (once per laptop, or per repository).
3. Link Claude: `bash setup_claude.sh` on macOS, `powershell -ExecutionPolicy Bypass -File setup_claude.ps1` on Windows.
4. Python environment for the robot code: Python 3.11, `pip install -r 02_code/requirements.txt` into `.venv`.
5. Copy `03_data/` from the backup if you need the recordings.
