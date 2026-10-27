# How we work with Claude: one orchestrator, several workstream sessions

Set up on 27.10.26. The reason: one long session becomes slow, forgetful and overwhelming. Sessions cannot share their
memory, so knowledge lives in files, and every session reads the same files at its start and updates them at its end.

## The three files every session uses

- `CLAUDE.md`: the working rules, loaded automatically in every session. Keep it short.
- `STATUS.md`: where each workstream stands. Read at the start; your section updated at the end.
- The memory folder (`C:\Users\<name>\.claude\projects\D--AI-Project\memory\`): facts and preferences Claude keeps across
  sessions. Updated when Claude learns something that must not be forgotten.

## The sessions

| Session | Job | May change |
| --- | --- | --- |
| Orchestrator | Plans the week, keeps `STATUS.md`, prepares meetings, starts the workstream sessions with a brief, reads their reports | `STATUS.md`, `05_meetings/`, `README.md`, `HOW_WE_WORK.md` |
| Robot and hardware | Camera mount, measurements, robot tests, checklists for robot sessions | test scripts in `02_code/`, `03_data/raw/`, `06_deliverables/research_notes/` |
| Recording and data | The recording tool, demonstration sessions, dataset builds | `02_code/`, `03_data/` |
| Model and training | Training code, runs on the laptop or the GPU pool | `02_code/`, `04_runs/` |
| Evaluation and write-up | Robot trials, numbers, report, slides | `04_runs/`, `06_deliverables/` |
| Questions | Any question, however small; explains, never edits | nothing |

Two sessions never edit the same files at the same time. If a task needs another session's files, it goes through the
orchestrator.

## The ritual

1. Open: the session reads `CLAUDE.md` (automatic) and `STATUS.md`, then the files its brief names.
2. Work: one task per session. When the task is done, the session is closed, not reused for the next unrelated task.
3. Close: before the last message, the session updates its section of `STATUS.md` (done, next, blocked on) and any memory
   fact that changed. The sentence to type is: "Update STATUS.md and memory with what changed, then stop."
4. The orchestrator reads the report and the updated `STATUS.md`, and decides what starts next.

## Token habits that cost nothing

- Point Claude at a file instead of pasting its contents into chat.
- Let Claude use subagents to read widely; the session then sees only the conclusion.
- Ask for `/compact` at a natural break if a session must go on; otherwise start a new one.
- Keep `CLAUDE.md` under 60 lines; facts go into notes, not into the rules.

## For a new team member

Read `HANDOFF_2026-10-26.md` once for the full background, then live from `STATUS.md`.
