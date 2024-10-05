#!/bin/bash
set -e
set -x

echo "Starting adaptive training at $(date)"

python train.py --task_name run_adaptive_sampling \
                --batch_size 1 \
                --num_epochs 100 \
                --accumulation_steps 2 \
                --num_workers 1 \
                --lr 8e-5 \
                --click_method random

echo "Adaptive training finished at $(date)"