#!/bin/bash
#SBATCH --partition=bch-gpu-pe
#SBATCH --time=05-10:00:00
#SBATCH --job-name=test_unet
#SBATCH --output=slurm_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:Titan_RTX:1

set -e
set -x

echo "Job started at $(date)"

module load anaconda3
module load cuda

source ~/.bashrc

conda activate pytorchuse

# echo "Running valiation.sh"
# ./validation.sh > validation_${SLURM_JOB_ID}.log 2>&1
./train.sh > train_${SLURM_JOB_ID}.log 2>&1

echo "Job finished at $(date)"