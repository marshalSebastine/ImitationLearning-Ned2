#!/bin/sh
# Waits for the second queue to finish, then runs the third queue's jobs, three at a time.
cd ~/ai_project/02_code || exit 1
while ! grep -q 'queue finished' ../logs/final2_queue.log 2>/dev/null; do sleep 30; done
echo "$(wc -l < ../logs/final3_jobs.txt) jobs, three at a time, started $(date)"
runner='log="../logs/final_${1}_$(echo "$3" | sed "s/without_//").log"; ../.venv/bin/python -u train_policy.py "$@" > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${1}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $*"'
xargs -P 3 -L 1 sh -c "$runner" sh < ../logs/final3_jobs.txt
echo "queue finished $(date)"
