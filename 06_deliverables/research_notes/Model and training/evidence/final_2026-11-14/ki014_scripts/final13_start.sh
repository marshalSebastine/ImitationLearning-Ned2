#!/bin/sh
# 15.11.26, thirteenth login, on ki-016: the recurrent model and the swing weighting; the direction memory again with
# its layer trained (the twelfth queue's memory jobs ran with a memory layer the optimiser never saw, so they are
# stopped here). The eleventh queue goes on.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; from train_policy_emranul import SequencePictureModel; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
pkill -u "$(id -un)" -f "final12_queu[e]"
pkill -u "$(id -un)" -f "train_policy_emranul.py resnet18_.*memor[y]"
sleep 3
echo "twelfth queue stopped; its log:"; cat ../logs/final12_queue.log
echo "eleventh queue so far: $(grep -c 'job ended' ../logs/final11_queue.log) ended, not exit 0: $(grep 'job ended' ../logs/final11_queue.log | grep -vc 'exit 0'); card: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_21_full_mirror_after_h_16_recurrent_all_sheets"
echo "resnet18_model_22_full_mirror_after_h_16_recurrent_swing_3_all_sheets"
echo "resnet18_model_23_full_mirror_after_h_16_swing_3_all_sheets"
echo "resnet18_model_19_full_mirror_after_h_16_memory_all_sheets"
for folds in "$P" "$Q"; do echo "resnet18_full_mirror_after_h_16_recurrent_course_sheets --folds $folds"; done
for idea in recurrent recurrent_swing_3 swing_3 memory; do
  for folds in "$A" "$B" "$C" "$D"; do echo "resnet18_full_mirror_after_h_16_${idea}_new_shapes --folds $folds"; done
done
for folds in "$P" "$Q"; do echo "resnet18_full_mirror_after_h_16_memory_course_sheets --folds $folds"; done; } > ../logs/final13_jobs.txt
echo "$(wc -l < ../logs/final13_jobs.txt) jobs in the thirteenth queue"
(nohup sh final13_queue.sh > ../logs/final13_queue.log 2>&1 < /dev/null &)
