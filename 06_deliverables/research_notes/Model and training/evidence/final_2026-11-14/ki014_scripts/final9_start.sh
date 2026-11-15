#!/bin/sh
# 15.11.26, ninth login: every remaining idea added alone to model 15's recipe (after h x5, the approach to the goal x2,
# 24 rounds, two pictures) and two more seeds of the recipe. Waits for nothing that is still running.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
echo "eighth queue: $(grep -c 'job ended' ../logs/final8_queue.log) jobs ended, not exit 0: $(grep 'job ended' ../logs/final8_queue.log | grep -vc 'exit 0'); card: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
{ for idea in approach_2 after_h_5 two_pictures 24_rounds seed_2 seed_3; do
  for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_${idea}_new_shapes --folds $folds"; done
done; } > ../logs/final9_jobs.txt
echo "$(wc -l < ../logs/final9_jobs.txt) jobs in the ninth queue"
(nohup sh final9_queue.sh > ../logs/final9_queue.log 2>&1 < /dev/null &)
