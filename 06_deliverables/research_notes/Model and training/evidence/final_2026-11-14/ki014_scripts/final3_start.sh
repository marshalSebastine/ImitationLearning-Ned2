#!/bin/sh
# 15.11.26, third login: the third queue (stronger networks, mirrored and turned pictures, the zoomed-in view, far-reach
# pictures counted more, a third seed of the reference). It waits until the second queue has finished and then starts
# by itself. Nothing of the second queue is touched.
cd ~/ai_project/02_code || exit 1
../.venv/bin/python -c "import config, train_policy; print('import check passed', len(config.TRAINING_RUNS), 'runs')" || exit 1
../.venv/bin/python -c "
import timm
for name in ('convnext_tiny.fb_in22k_ft_in1k', 'resnet50.a1_in1k'):
    model = timm.create_model(name, pretrained = True, num_classes = 0)
    print('start weights ready:', name, round(sum(p.numel() for p in model.parameters()) / 1e6, 1), 'million weights')
" || echo "WARNING: the start weights of a larger network could not be fetched; its jobs will fail, the others run"
ls -la ../03_data/processed/v03_training_index/ | grep images
n=final2_stream
pkill -u "$(id -un)" -f "sh 02_code/$n.sh"
echo "second queue so far: $(grep -c 'job ended' ../logs/final2_queue.log) jobs ended, of these not with exit 0: $(grep 'job ended' ../logs/final2_queue.log | grep -vc 'exit 0')"
grep 'job ended' ../logs/final2_queue.log | grep -v 'exit 0' | cut -c1-120
A="without_T01 without_T02 without_T03 without_T04"
B="without_T05 without_T06 without_T07 without_T08"
C="without_T09 without_T10 without_T11 without_T12"
D="without_T13 without_T14 without_T15 without_T16"
for run in resnet18_full_zoom_new_shapes resnet18_full_turned_12_new_shapes resnet18_full_mirror_new_shapes resnet18_full_far_reach_3_new_shapes resnet18_full_new_shapes_seed_3 resnet18_full_convnext_tiny_new_shapes resnet18_full_resnet50_new_shapes resnet18_full_zoom_new_shapes_seed_2 resnet18_full_zoom_turned_mirror_new_shapes resnet18_full_convnext_tiny_zoom_new_shapes; do
  for folds in "$A" "$B" "$C" "$D"; do echo "$run --folds $folds"; done
done > ../logs/final3_jobs.txt
echo "$(wc -l < ../logs/final3_jobs.txt) jobs in the third queue; it starts when the second has finished"
(nohup sh final3_queue.sh > ../logs/final3_queue.log 2>&1 < /dev/null &)
