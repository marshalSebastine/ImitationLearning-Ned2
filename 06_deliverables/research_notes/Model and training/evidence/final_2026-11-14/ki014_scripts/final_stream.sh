#!/bin/sh
# Sends the finished result files of the final queue back as tar archives on standard output, every minute, until the
# queue has finished (progress goes to standard error). Only files at least a minute old are sent, so none is half
# written. The archives follow each other in one stream: read it with "tar -x -i".
cd ~/ai_project || exit 1
sent=logs/final_sent.txt
: > "$sent"
while :; do
  is_finished=0
  grep -q 'queue finished' logs/final_queue.log 2>/dev/null && is_finished=1
  [ "$is_finished" = 1 ] && sleep 70
  find 04_runs -type f -mmin +1 \( -path '04_runs/2026-11-14_resnet18_full_*' -o -path '04_runs/2026-11-14_resnet18_model_9_full_*' \) \
       \( -name 'fold_*.json' -o -name 'predictions_*.npz' -o -name 'all_sheets.pt' \) | LC_ALL=C sort > logs/final_now.txt
  LC_ALL=C comm -23 logs/final_now.txt "$sent" > logs/final_new.txt
  if [ -s logs/final_new.txt ]; then
    tar -cf - -T logs/final_new.txt
    cat logs/final_new.txt >> "$sent"
    LC_ALL=C sort -o "$sent" "$sent"
    echo "sent $(wc -l < "$sent") result files; $(grep -c 'job ended' logs/final_queue.log) jobs ended, $(date +%H:%M)" 1>&2
  fi
  [ "$is_finished" = 1 ] && break
  sleep 60
done
tar -cf - logs/final_queue.log logs/final_jobs.txt
echo "all results sent $(date)" 1>&2
