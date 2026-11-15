#!/bin/sh
# Waits for the second queue to finish, then runs the third queue's jobs, three at a time.
cd ~/ai_project/02_code || exit 1
# waits for the second queue's own end line; goes on as well when its runner is gone without one
while ! grep -q 'queue finished' ../logs/final2_queue.log 2>/dev/null && pgrep -u "$(id -un)" -x xargs > /dev/null; do sleep 30; done
while [ "$(pgrep -u "$(id -un)" -f "^[^ ]*python -u train_policy.py" | wc -l)" -ge 3 ]; do sleep 20; done
echo "$(wc -l < ../logs/final3_jobs.txt) jobs, three at a time, started $(date)"
runner='log="../logs/final_${1}_$(echo "$3" | sed "s/without_//").log"; ../.venv/bin/python -u train_policy.py "$@" > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${1}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $*"'
xargs -P 3 -L 1 sh -c "$runner" sh < ../logs/final3_jobs.txt
echo "queue finished $(date)"
