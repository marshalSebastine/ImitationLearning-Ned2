#!/bin/sh
# Unpacks whatever has come back from ki-014 so far (both logins) into results/, copies it into the worktree's
# 04_runs/gpu_ki014 and lists what is there.
S="/c/Users/e7dc/AppData/Local/Temp/claude/1808d3456aab7b9a/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/final"
mkdir -p "$S/results" && cd "$S/results" || exit 1
for archive in ../final_results.tar ../final2_results.tar ../final3_results.tar ../final4_results.tar; do
  [ -s "$archive" ] && tar -x -i -f "$archive" 2>/dev/null
done
mkdir -p /d/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014
cp -r 04_runs/. /d/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/ 2>/dev/null
for run in 04_runs/*; do echo "$(basename "$run"): $(ls "$run" | grep -c '^fold_') folds"; done
