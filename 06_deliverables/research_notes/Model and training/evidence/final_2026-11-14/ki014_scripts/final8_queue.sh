#!/bin/sh
# Waits for the fifth queue to finish, then runs the seventh queue's jobs, three at a time, from 02_code_5.
cd ~/ai_project/02_code_5 || exit 1
# waits for the third queue's own end line; goes on as well when its script and its job runner are both gone without one
while ! grep -q 'queue finished' ../logs/final7_queue.log 2>/dev/null && { pgrep -u "$(id -un)" -f 'final7_queue.sh' > /dev/null || pgrep -u "$(id -un)" -x xargs > /dev/null; }; do sleep 30; done
while [ "$(pgrep -u "$(id -un)" -f "^[^ ]*python -u train_policy" | wc -l)" -ge 3 ]; do sleep 20; done
echo "$(wc -l < ../logs/final8_jobs.txt) jobs, three at a time, started $(date)"
runner='log="../logs/final_${1}_$(echo "$3" | sed "s/without_//").log"; ../.venv/bin/python -u train_policy_emranul.py "$@" > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${1}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $*"'
xargs -P 3 -L 1 sh -c "$runner" sh < ../logs/final8_jobs.txt
echo "queue finished $(date)"
