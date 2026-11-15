#!/bin/sh
# Fifth login: sends ALL finished result files of all queues back as tar archives on standard output, every minute,
# until the fifth queue has finished (progress goes to standard error). Only files at least a minute old are sent, so
# none is half written. The archives follow each other in one stream: read it with "tar -x -i".
cd ~/ai_project || exit 1
sent=logs/final7_sent.txt
: > "$sent"
while :; do
  is_finished=0
  grep -q 'queue finished' logs/final7_queue.log 2>/dev/null && is_finished=1
  [ "$is_finished" = 1 ] && sleep 70
  find 04_runs -type f -mmin +1 \( -path '04_runs/2026-11-14_resnet18_full_*' -o -path '04_runs/2026-11-14_resnet18_model_*_full_*' \) \
       \( -name 'fold_*.json' -o -name 'predictions_*.npz' -o -name 'all_sheets.pt' -o -name 'metrics.json' \) | LC_ALL=C sort > logs/final7_now.txt
  LC_ALL=C comm -23 logs/final7_now.txt "$sent" > logs/final7_new.txt
  if [ -s logs/final7_new.txt ]; then
    tar -cf - -T logs/final7_new.txt
    cat logs/final7_new.txt >> "$sent"
    LC_ALL=C sort -o "$sent" "$sent"
    echo "sent $(wc -l < "$sent") result files; jobs ended: second queue $(grep -c 'job ended' logs/final2_queue.log), third queue $(grep -c 'job ended' logs/final3_queue.log), seventh queue $(grep -c "job ended" logs/final7_queue.log 2>/dev/null), $(date +%H:%M)" 1>&2
  fi
  [ "$is_finished" = 1 ] && break
  sleep 60
done
tar -cf - logs/final_jobs.txt logs/final2_jobs.txt logs/final3_jobs.txt logs/final5_jobs.txt logs/final7_jobs.txt logs/final7_queue.log logs/final_queue.log logs/final2_queue.log logs/final3_queue.log logs/final5_queue.log
echo "all results sent $(date)" 1>&2
