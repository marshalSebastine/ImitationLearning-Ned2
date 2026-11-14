#!/bin/sh
# Runs the job list of the second login, three at a time, once fewer than three trainings of the first queue are left;
# then the 320-pixel jobs, two at a time when 16 GB of memory or more are free (each holds 5 GB of pictures), else one.
cd ~/ai_project/02_code || exit 1
while [ "$(pgrep -u "$(id -un)" -f "^[^ ]*python -u train_policy.py" | wc -l)" -ge 3 ]; do sleep 20; done
echo "$(wc -l < ../logs/final2_jobs.txt) jobs, three at a time, started $(date)"
runner='log="../logs/final_${1}_$(echo "$3" | sed "s/without_//").log"; ../.venv/bin/python -u train_policy.py "$@" > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${1}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $*"'
xargs -P 3 -L 1 sh -c "$runner" sh < ../logs/final2_jobs.txt
wide_jobs=1
[ "$(free -g | awk '/^Mem|^Speicher/ {print $7}')" -ge 16 ] 2>/dev/null && wide_jobs=2
echo "320-pixel jobs, $wide_jobs at a time, started $(date)"
xargs -P "$wide_jobs" -L 1 sh -c "$runner" sh < ../logs/final2_jobs_wide.txt
echo "queue finished $(date)"
