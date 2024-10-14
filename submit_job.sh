#!/bin/bash
#SBATCH --partition=bch-gpu-pe
#SBATCH --time=05-10:00:00
#SBATCH --job-name=sam_validate_adaptive
#SBATCH --output=slurm_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:Titan_RTX:1

set -e
set -x

echo "Job started at $(date)"

module load anaconda3
module load cuda

source ~/.bashrc

conda activate pytorchuse

# Debug information
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -l

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

JOB_ID=${SLURM_JOB_ID:-$$}

# # Run training with adaptive sampling
# echo "Starting training at $(date)"
# ./train_adaptive.sh > "train_adaptive_${JOB_ID}.log" 2>&1
# if [ $? -ne 0 ]; then
#     echo "Error: train_adaptive.sh failed"
#     exit 1
# fi
# echo "Training finished at $(date)"

# # Check if log file was created
# echo "Checking for training log file:"
# ls -l "train_adaptive_${JOB_ID}.log"

# Run validation for adaptive sampling model
echo "Starting validation at $(date)"
./validate_adaptive.sh > "validate_adaptive_${JOB_ID}.log" 2>&1
if [ $? -ne 0 ]; then
    echo "Error: validate_adaptive.sh failed"
    exit 1
fi

echo "Checking for validation log file:"
ls -l "validate_adaptive_${JOB_ID}.log"

echo "Job finished at $(date)"

# # Run validation after training
# echo "Running validation.sh"
# ./validation.sh > validation_${SLURM_JOB_ID}.log 2>&1

# # Specify the dataset to exclude
# # export EXCLUDE_DATASET="Task501_HIE"

# # Run training
# # echo "Running train.sh"
# # ./train.sh > train_${SLURM_JOB_ID}.log 2>&1