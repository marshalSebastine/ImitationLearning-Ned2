#!/bin/sh
# 14.11.26 evening, first queue of the search for the final model (round 1 plus all of round 2, index v03): model 9 for
# the robot, the reference recipe with each of the 16 new tracks held out (two seeds), the same recipe on round 1 only,
# then one change each: two pictures, score-1 episodes without weight, ResNet-34, 16 rounds. Three jobs at a time, in
# this order, so the most important results come first. The weights of the held-out folds are deleted after each job
# (only their scores and predictions are needed); model 9's weights are kept.
cd ~/ai_project/02_code || exit 1
../.venv/bin/python -c "import config, train_policy; print('import check passed', config.FULL_INDEX_FOLDER)" || exit 1
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
{
  echo "resnet18_model_9_full_all_sheets"
  for run in resnet18_full_new_shapes; do for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done; done
  echo "resnet18_full_round_1_to_new"
  for run in resnet18_full_new_shapes_seed_2 resnet18_full_two_pictures_new_shapes resnet18_full_score_1_off_new_shapes resnet18_full_resnet34_new_shapes resnet18_full_16_rounds_new_shapes; do
    for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done
  done
} > ../logs/final_jobs.txt
df -h . | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "$(wc -l < ../logs/final_jobs.txt) jobs, three at a time, started $(date)"
xargs -P 3 -L 1 sh -c 'log="../logs/final_${1}_$(echo "$3" | sed "s/without_//").log"; ../.venv/bin/python -u train_policy.py "$@" > "$log" 2>&1; status=$?; rm -f ../04_runs/2026-11-14_${1}/checkpoints/without_*.pt; echo "job ended $(date +%H:%M) (exit $status): $*"' sh < ../logs/final_jobs.txt
echo "queue finished $(date)"
