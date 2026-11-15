#!/bin/sh
# 15.11.26, twelfth login, on ki-016: the direction memory. Robot models 19 (model 15 with the memory) and 20 (model 17,
# no mirror, with the memory) first, then the memory on the 16 new tracks (two seeds) and on the course sheets. It runs
# beside the eleventh queue: each queue starts a job only while fewer than three of ours run and the card has room.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; from training_targets import memory_from_path; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
echo "eleventh queue so far: $(grep -c 'job ended' ../logs/final11_queue.log) ended, not exit 0: $(grep 'job ended' ../logs/final11_queue.log | grep -vc 'exit 0'); card: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_19_full_mirror_after_h_16_memory_all_sheets"
echo "resnet18_model_20_full_after_h_16_no_mirror_memory_all_sheets"
for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_memory_new_shapes --folds $folds"; done
for folds in "$P" "$Q"; do echo "resnet18_full_mirror_after_h_16_memory_course_sheets --folds $folds"; done
for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_memory_seed_2_new_shapes --folds $folds"; done; } > ../logs/final12_jobs.txt
echo "$(wc -l < ../logs/final12_jobs.txt) jobs in the twelfth queue"
(nohup sh final12_queue.sh > ../logs/final12_queue.log 2>&1 < /dev/null &)
