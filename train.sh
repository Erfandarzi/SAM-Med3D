#!/bin/bash
set -e
set -x

echo "Starting training at $(date)"

python train.py --task_name run_3 --batch_size 1 --num_epochs 100 --accumulation_steps 2 --num_workers 1 --lr 8e-5

echo "Training finished at $(date)"