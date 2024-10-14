#!/bin/bash
set -e

# Remove set -x to reduce verbosity

# Generate a unique identifier for this run
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Redirect all output to a log file
exec > >(tee "train_adaptive_${RUN_ID}.log") 2>&1

echo "Starting adaptive training at $(date)"

python train.py --task_name "run_adaptive_sampling_${RUN_ID}" \
                --batch_size 1 \
                --num_epochs 100 \
                --accumulation_steps 2 \
                --num_workers 1 \
                --lr 8e-5 \
                --click_method random

echo "Adaptive training finished at $(date)"