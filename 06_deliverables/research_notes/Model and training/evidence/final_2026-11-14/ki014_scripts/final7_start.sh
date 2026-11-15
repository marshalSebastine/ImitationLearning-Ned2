#!/bin/sh
# 15.11.26, seventh login: the combined recipes (mirror, after-h weight, 16 rounds; the same on the zoomed, turned view),
# robot models 15 and 16 first, and the missing folds of zoom + turned + mirror. Nothing else is running on ki-014.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python -c "import config, train_policy_emranul; print('import check passed', len(config.TRAINING_RUNS), 'runs')" 2>/dev/null || exit 1
echo "--- why jobs of the third queue failed (last lines of three logs):"
for log in final_resnet18_full_resnet50_new_shapes_T01 final_resnet18_full_zoom_turned_mirror_new_shapes_T01 final_resnet18_full_new_shapes_seed_3_T13; do echo "[$log]"; tail -n 3 ../logs/$log.log 2>&1 | cut -c1-300; done
echo "--- card now:"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- training programs running: $(pgrep -u "$(id -un)" -f "python -u train_policy" | wc -l)"
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_15_full_mirror_after_h_16_all_sheets"
echo "resnet18_model_16_full_zoom_turned_mirror_after_h_16_all_sheets"
echo "resnet18_full_zoom_turned_mirror_new_shapes --folds without_T04 without_T06 without_T07 without_T08"
echo "resnet18_full_zoom_turned_mirror_new_shapes --folds without_T09 without_T10 without_T11 without_T12"
for run in resnet18_full_mirror_after_h_16_new_shapes resnet18_full_zoom_turned_mirror_after_h_16_new_shapes; do
  for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done
done
for run in resnet18_full_mirror_after_h_16_course_sheets resnet18_full_zoom_turned_mirror_after_h_16_course_sheets; do
  for folds in "$P" "$Q"; do echo "$run --folds $folds"; done
done; } > ../logs/final7_jobs.txt
echo "$(wc -l < ../logs/final7_jobs.txt) jobs in the seventh queue"
(nohup sh final7_queue.sh > ../logs/final7_queue.log 2>&1 < /dev/null &)
