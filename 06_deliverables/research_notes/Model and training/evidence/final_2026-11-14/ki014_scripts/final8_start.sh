#!/bin/sh
# 15.11.26, eighth login: the last 3 steps before an h press at a third of the weight (Emranul's information), alone and
# with the after-h weight. Waits for the seventh queue. Only config.py changed, by additions.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
echo "seventh queue so far: $(grep -c 'job ended' ../logs/final7_queue.log) jobs ended, not exit 0: $(grep 'job ended' ../logs/final7_queue.log | grep -vc 'exit 0')"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
{ for run in resnet18_full_before_h_3_third_after_h_3_new_shapes resnet18_full_before_h_3_third_new_shapes; do
  for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done
done; } > ../logs/final8_jobs.txt
echo "$(wc -l < ../logs/final8_jobs.txt) jobs in the eighth queue; it starts when the seventh has finished"
(nohup sh final8_queue.sh > ../logs/final8_queue.log 2>&1 < /dev/null &)
