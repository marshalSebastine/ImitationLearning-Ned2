#!/bin/sh
# Runs the twelfth queue's jobs, up to three at a time, but starts one only when the card has at least 1400 MiB free (one job used 1252 MiB on 15.11.26);
# after a start it waits 90 seconds, so that the new job has taken its memory before the next look.
cd ~/ai_project/02_code_5 || exit 1
echo "$(wc -l < ../logs/final12_jobs.txt) jobs, up to three at a time when the card has room, started $(date)"
while read -r line; do
  [ -z "$line" ] && continue
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    running=$(pgrep -u "$(id -un)" -f "^[^ ]*python -u train_policy" | wc -l)
    [ "$free" -ge 1400 ] && [ "$running" -lt 3 ] && break
    sleep 30
  done
  set -- $line
  name=$1; folds=$3
  log="../logs/final_${name}_$(echo "$folds" | sed "s/without_//").log"
  ( ../.venv/bin/python -u train_policy_emranul.py $line > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${name}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $line" ) &
  echo "job started $(date +%H:%M) with $free MiB free: $line"
  sleep 90
done < ../logs/final12_jobs.txt
wait
echo "queue finished $(date)"
