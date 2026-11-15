#!/bin/sh
# 15.11.26, fifth login: the hand-correction runs (the steps before an h press left out of training, the steps after
# it counted three times; Emranul's question of 15.11.26). They run from their own code folder 02_code_5, so the third
# queue's jobs, which read 02_code, are not touched. The queue waits until the third has finished and starts by itself.
cd ~/ai_project/02_code_5 || exit 1
../.venv/bin/python - <<'EOF' || exit 1
import numpy
import config
from train_policy_emranul import merged_run_settings
from training_probes import training_arrays
from training_targets import load_training_frames

settings = merged_run_settings("resnet18_full_without_before_h_5_after_h_3_new_shapes")
target_parameters = dict(config.TRAINING_TARGET_PARAMETERS)
target_parameters.update({key: value for key, value in settings.items() if key in target_parameters})
settings["chunk_steps"] = target_parameters["chunk_steps"]
data_parameters = dict(config.TRAINING_DATA_PARAMETERS)
data_parameters["processed_folder"] = settings["processed_folder"]
frames = load_training_frames(target_parameters, data_parameters)
changed, _, _ = training_arrays(frames, settings, target_parameters)
usual, _, _ = training_arrays(frames, dict(settings, before_correction_weight = 1.0, after_correction_weight = 1.0), target_parameters)
left_out = int(((changed["sample_weight"] == 0) & (usual["sample_weight"] > 0)).sum())
heavier = int((changed["sample_weight"] > usual["sample_weight"]).sum())
print("import check passed:", len(config.TRAINING_RUNS), "runs,", len(frames), "pictures; before an h press left out:", left_out, "(laptop: 1471); after one counted more:", heavier, "(laptop: 2321)")
if (len(frames), left_out, heavier) != (22646, 1471, 2321):
    print("WARNING: these counts differ from the laptop's; the runs go on, read the results with that in mind")
assert left_out > 1000 and heavier > 1500, "the corrections by hand were not found in the table of pictures"
EOF
# older result streams of a lost connection, if any are left (this login's own stream has another name)
for name in final3_stream final4_stream; do pkill -u "$(id -un)" -f "sh 02_code/$name.sh"; done
echo "third queue so far: $(grep -c 'job ended' ../logs/final3_queue.log 2>/dev/null) jobs ended, of these not with exit 0: $(grep 'job ended' ../logs/final3_queue.log 2>/dev/null | grep -vc 'exit 0')"
grep 'job ended' ../logs/final3_queue.log 2>/dev/null | grep -v 'exit 0' | cut -c1-120
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
P="without_path_01 without_path_02 without_path_03"
Q="without_path_04 without_path_05 without_path_06"
{ echo "resnet18_model_13_full_hand_corrections_all_sheets"
echo "resnet18_model_14_full_hand_corrections_two_pictures_all_sheets"
for folds in "$P" "$Q"; do echo "resnet18_full_without_before_h_5_after_h_3_course_sheets --folds $folds"; done
for run in resnet18_full_without_before_h_5_after_h_3_new_shapes resnet18_full_without_before_h_5_new_shapes resnet18_full_after_h_3_new_shapes resnet18_full_without_before_h_8_after_h_3_new_shapes; do
  for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done
done; } > ../logs/final5_jobs.txt
echo "$(wc -l < ../logs/final5_jobs.txt) jobs in the fifth queue; it starts when the third has finished"
(nohup sh final5_queue.sh > ../logs/final5_queue.log 2>&1 < /dev/null &)
