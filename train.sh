#!/bin/bash
set -e
set -x

echo "Starting training at $(date)"

# Specify the dataset to exclude (e.g., "Task501_HIE")
EXCLUDE_DATASET="Task501_HIE"

python train.py --task_name run_4_exclude_${EXCLUDE_DATASET} \
                --batch_size 1 \
                --num_epochs 100 \
                --accumulation_steps 2 \
                --num_workers 1 \
                --lr 8e-5 \
                --exclude_dataset ${EXCLUDE_DATASET}

echo "Training finished at $(date)"