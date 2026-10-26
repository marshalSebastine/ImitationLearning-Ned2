# Shared Claude memory for the AI Project

This folder is Claude Code's memory for this project, shared through git between Emranul's and Marshal's Claude accounts.
`MEMORY.md` is the index Claude loads each session; every other file holds one fact.

Set-up, once per laptop after cloning (and again if the repository folder moves):

- macOS or Linux: `bash setup_claude.sh`
- Windows: `powershell -ExecutionPolicy Bypass -File setup_claude.ps1`

The script links Claude's own memory folder for this project (`~/.claude/projects/<project slug>/memory`) to this folder, and
links `claude_skills/*` into `~/.claude/skills/`. Memory Claude writes from then on lands here, so it goes into git with the
next commit. Any memory already in Claude's folder is copied in first and the old folder is kept as `memory.before_link`.

Workflow: `git pull` before starting a Claude session, commit and push memory changes like any other file. Rules for writing
memory are in `shared-memory-across-accounts.md`. One file about Emranul's own tooling was left out on purpose.
