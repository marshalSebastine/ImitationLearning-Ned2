#!/bin/sh
# 15.11.26, eleventh login: the tenth queue again. At 18:26 another person's training (6.4 GB) filled ki-014's card and
# 15 of our 16 jobs ended at once for lack of memory. This queue starts a job only while the card has room for it.
cd ~/ai_project/02_code_5 || exit 1
echo "card now: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_17_full_after_h_16_no_mirror_all_sheets"
echo "resnet18_model_18_full_mirror_after_h_16_final_steps_all_sheets"
for folds in "$P" "$Q"; do echo "resnet18_full_after_h_16_no_mirror_course_sheets --folds $folds"; done
for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_no_mirror_new_shapes --folds $folds"; done
for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_final_steps_3_new_shapes --folds $folds"; done
for folds in "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_no_mirror_seed_2_new_shapes --folds $folds"; done; } > ../logs/final11_jobs.txt
echo "$(wc -l < ../logs/final11_jobs.txt) jobs in the eleventh queue (no_mirror_seed_2 T01 to T04 survived in the tenth)"
(nohup sh final11_queue.sh > ../logs/final11_queue.log 2>&1 < /dev/null &)
