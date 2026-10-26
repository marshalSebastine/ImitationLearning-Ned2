#!/usr/bin/env bash
# Links this repository's shared Claude memory and skills into Claude Code's own folders (macOS and Linux).
# Run once per laptop after cloning, and again whenever the repository folder moves:  bash setup_claude.sh
# Windows: run setup_claude.ps1 instead.
set -euo pipefail

project_directory="$(cd "$(dirname "$0")" && pwd)"
claude_directory="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
# Claude Code names a project's folder after its path, with every character that is not a letter or digit replaced by "-".
project_slug="$(printf '%s' "$project_directory" | sed 's/[^A-Za-z0-9]/-/g')"
memory_link="$claude_directory/projects/$project_slug/memory"
shared_memory="$project_directory/claude_memory"

mkdir -p "$(dirname "$memory_link")"
if [ -d "$memory_link" ] && [ ! -L "$memory_link" ]; then
    # Keep any memory Claude already wrote here: copy files the repository does not have yet, then swap in the link.
    cp -n "$memory_link"/*.md "$shared_memory"/ 2>/dev/null || true
    mv "$memory_link" "$memory_link.before_link"
    echo "Old memory folder kept as $memory_link.before_link; new files were copied into claude_memory/."
fi
ln -sfn "$shared_memory" "$memory_link"
echo "Memory:  $memory_link -> $shared_memory"

mkdir -p "$claude_directory/skills"
for skill_directory in "$project_directory"/claude_skills/*/; do
    skill_name="$(basename "$skill_directory")"
    ln -sfn "${skill_directory%/}" "$claude_directory/skills/$skill_name"
    echo "Skill:   $claude_directory/skills/$skill_name -> ${skill_directory%/}"
done
