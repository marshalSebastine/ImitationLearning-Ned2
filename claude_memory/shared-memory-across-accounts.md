---
name: shared-memory-across-accounts
description: "This memory folder is the repo's claude_memory/, shared through git by two people on two Claude accounts; how to write to it so the other side's sessions stay in sync"
metadata:
  type: feedback
  modified: 2026-10-26T18:00:00.000Z
---

On 26.10.26 Marshal asked that the project-specific Claude memory from every future session be tracked in the GitHub repo, so that Claude sessions on Emranul's and Marshal's separate accounts can collaborate.

**Why:** the two accounts share no chat history; the repo (code, notes, this memory) is the only channel between their sessions.

**How to apply:**
- Claude's memory folder for this project is a link to `claude_memory/` in the repo (made by `setup_claude.sh` / `setup_claude.ps1`), so memory writes land in git automatically. If it is not linked on this machine, write project memory into `claude_memory/` directly anyway.
- Only project facts go here; both people read every file. Say who said or decided something ("Marshal said …"), never "the user", and use absolute dates.
- Workflow decided 27.10.26 (Marshal asked for it, details in `COLLABORATION.md`): work on `<person>/<topic>` branches merged into `main` by pull request; at session start pull `main` and merge it into the branch; at session end commit and push the branch. Memory changes ride along on the branch; a fact the other person needs at once is committed straight to `main`. In `MEMORY.md` keep both sides' lines when merging.
- Personal, one-person memory goes in `personal-<name>-<topic>.md` (git-ignored, not indexed in `MEMORY.md`).
- Commits follow [[feedback-no-claude-in-git]].
