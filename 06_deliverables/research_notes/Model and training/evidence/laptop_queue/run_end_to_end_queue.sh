#!/usr/bin/env bash
# Runs the end-to-end trainings one after another on three held-out sheets, logging each to its own file.
# Usage: bash run_end_to_end_queue.sh run_name [run_name ...]
cd "D:/AI_Project/.claude/worktrees/training-analysis/02_code" || exit 1
LOGS="C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/logs"
FOLDS="${FOLDS:-without_path_03 without_path_05 without_path_06}"
for run_name in "$@"; do
    echo "=== $run_name started $(date +%H:%M)"
    ../../../../.venv/Scripts/python.exe -u train_policy.py "$run_name" --folds $FOLDS > "$LOGS/e2e_$run_name.log" 2>&1
    echo "=== $run_name finished $(date +%H:%M) exit $?"
    grep -E "^$run_name \| without|Traceback|Error" "$LOGS/e2e_$run_name.log" | cut -c1-300
done
