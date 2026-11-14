#!/usr/bin/env bash
# One night of training on the GPU pool (Marshal, 13.11.26). Steps, in this order:
#   1. train_policy.py --check: GPU, data, ImageNet weights and every policy; stops here if anything is wrong.
#   2. the five policies with seed 1, then all five with seed 2, then with seed 3, so a short night still gives one of each;
#   3. the Optuna search over the factorised classifier (at most 3 hours), then its best settings trained in full;
#   4. the summary table 04_runs/<batch>_summary.md (also rewritten after every run).
# Every finished run is skipped when the script is started again with the same batch name, so a stopped night can be
# continued:   bash run_overnight.sh 2026-11-13_overnight
# The whole output also goes to 04_runs/logs/<batch>.log.

set -u
cd "$(dirname "$0")"
PYTHON="${PYTHON:-../.venv_gpu/bin/python}"
BATCH="${1:-$(date +%F)_overnight}"
MODELS="factorised_classifier exact_step_regressor history_classifier chunk_transformer diffusion_policy"

mkdir -p ../04_runs/logs
exec > >(tee -a "../04_runs/logs/${BATCH}.log") 2>&1
echo "=== batch ${BATCH} started $(date) on $(hostname), python ${PYTHON}"

if ! "$PYTHON" train_policy.py --check; then
    echo "=== the check failed: nothing was trained. Fix the message above, then start again."
    exit 1
fi

for seed in 1 2 3; do
    for model in $MODELS; do
        echo "=== ${model} seed ${seed} started $(date)"
        if ! "$PYTHON" train_policy.py --model "$model" --seed "$seed" --batch "$BATCH" --skip-done; then
            echo "=== FAILED: ${model} seed ${seed}; going on with the next run"
        fi
        "$PYTHON" train_policy.py --summary --batch "$BATCH" > /dev/null || true
    done
done

echo "=== Optuna search started $(date)"
"$PYTHON" tune_policy.py --batch "$BATCH" || echo "=== FAILED: Optuna search"

"$PYTHON" train_policy.py --summary --batch "$BATCH"
echo "=== batch ${BATCH} finished $(date)"
