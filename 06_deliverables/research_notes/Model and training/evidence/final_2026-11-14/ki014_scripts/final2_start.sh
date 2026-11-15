#!/bin/sh
# 14.11.26 night, second login: puts the stop fixes in front of what is left of the first queue, the track-count test
# behind it, and the 320-pixel test last. The first queue's runner is stopped (its three running jobs finish by
# themselves), the jobs it had not started yet are found by their missing log files, and one new runner takes over.
cd ~/ai_project/02_code || exit 1
../.venv/bin/python -c "import config, train_policy; print('import check passed', len(config.TRAINING_RUNS), 'runs')" || exit 1
ls -la ../03_data/processed/v03_training_index/ | grep images
pkill -u "$(id -un)" -x xargs
pkill -u "$(id -un)" -f final_stream.sh
sleep 2
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
{
  echo "resnet18_model_10_full_without_14_02_all_sheets"
  for run in table_as_paper made_up_ends_4 mid_paper_stop_4 made_up_ends_10 made_up_ends_and_table table_and_mid_paper; do
    for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_${run}_new_shapes --folds $folds"; done
  done
  while read -r line; do
    set -- $line
    [ -e "../logs/final_${1}_$(echo "$3" | sed 's/without_//').log" ] || echo "$line"
  done < ../logs/final_jobs.txt
  for count in 12 8 4; do
    for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_${count}_tracks_new_shapes --folds $folds"; done
  done
} > ../logs/final2_jobs.txt
for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_wide_320_new_shapes --folds $folds"; done > ../logs/final2_jobs_wide.txt
free -g | head -2
echo "$(wc -l < ../logs/final2_jobs.txt) jobs in the new list, then $(wc -l < ../logs/final2_jobs_wide.txt) with 320-pixel pictures; still running from the first queue: $(pgrep -u "$(id -un)" -f "^[^ ]*python -u train_policy.py" | wc -l)"
cut -c1-72 ../logs/final2_jobs.txt
(nohup sh final2_queue.sh > ../logs/final2_queue.log 2>&1 < /dev/null &)
