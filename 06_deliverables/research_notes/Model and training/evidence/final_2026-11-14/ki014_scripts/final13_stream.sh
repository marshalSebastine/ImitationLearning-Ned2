#!/bin/sh
# Thirteenth login, on ki-016: sends every finished result file of ki-016 (eleventh, twelfth and thirteenth queue) back as tar archives
# on standard output, every minute, until both queues have finished (progress goes to standard error). Read with
# "tar -x -i".
cd ~/ai_project || exit 1
sent=logs/final13_sent.txt
: > "$sent"
while :; do
  is_finished=0
  grep -q 'queue finished' logs/final11_queue.log 2>/dev/null && grep -q 'queue finished' logs/final13_queue.log 2>/dev/null && is_finished=1
  [ "$is_finished" = 1 ] && sleep 70
  find 04_runs -type f -mmin +1 \( -path '04_runs/2026-11-14_resnet18_full_*' -o -path '04_runs/2026-11-14_resnet18_model_*_full_*' \) \
       \( -name 'fold_*.json' -o -name 'predictions_*.npz' -o -name 'all_sheets.pt' -o -name 'metrics.json' \) | LC_ALL=C sort > logs/final13_now.txt
  LC_ALL=C comm -23 logs/final13_now.txt "$sent" > logs/final13_new.txt
  if [ -s logs/final13_new.txt ]; then
    tar -cf - -T logs/final13_new.txt
    cat logs/final13_new.txt >> "$sent"
    LC_ALL=C sort -o "$sent" "$sent"
    echo "sent $(wc -l < "$sent") result files from ki-016; jobs ended: eleventh queue $(grep -c 'job ended' logs/final11_queue.log), thirteenth queue $(grep -c "job ended" logs/final13_queue.log 2>/dev/null), $(date +%H:%M)" 1>&2
  fi
  [ "$is_finished" = 1 ] && break
  sleep 60
done
tar -cf - logs/final11_jobs.txt logs/final11_queue.log logs/final12_jobs.txt logs/final12_queue.log logs/final13_jobs.txt logs/final13_queue.log
echo "all results sent $(date)" 1>&2
