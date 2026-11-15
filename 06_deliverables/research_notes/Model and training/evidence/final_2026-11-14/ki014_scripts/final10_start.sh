#!/bin/sh
# 15.11.26, tenth login: robot models 17 (model 15's recipe without mirror) and 18 (with the last 5 steps of score-3
# episodes three times) first, then the comparison runs: no mirror (two seeds, and on the course sheets), final steps.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
echo "ninth queue: $(grep -c 'job ended' ../logs/final9_queue.log) jobs ended, not exit 0: $(grep 'job ended' ../logs/final9_queue.log | grep -vc 'exit 0'); card: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_17_full_after_h_16_no_mirror_all_sheets"
echo "resnet18_model_18_full_mirror_after_h_16_final_steps_all_sheets"
for folds in "$P" "$Q"; do echo "resnet18_full_after_h_16_no_mirror_course_sheets --folds $folds"; done
for idea in no_mirror final_steps_3 no_mirror_seed_2; do
  for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_${idea}_new_shapes --folds $folds"; done
done; } > ../logs/final10_jobs.txt
echo "$(wc -l < ../logs/final10_jobs.txt) jobs in the tenth queue"
(nohup sh final10_queue.sh > ../logs/final10_queue.log 2>&1 < /dev/null &)
