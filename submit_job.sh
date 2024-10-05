#!/bin/bash
#SBATCH --partition=bch-gpu-pe
#SBATCH --time=05-10:00:00
#SBATCH --job-name=sam_train_adaptive
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

# Run training with adaptive sampling
echo "Running train_adaptive.sh"
./train_adaptive.sh > train_adaptive_${SLURM_JOB_ID}.log 2>&1

# Run validation after training
echo "Running validation.sh"
./validation.sh > validation_${SLURM_JOB_ID}.log 2>&1

echo "Job finished at $(date)"


# #!/bin/bash
# #SBATCH --partition=bch-gpu-pe
# #SBATCH --time=05-10:00:00
# #SBATCH --job-name=sam_train
# #SBATCH --output=slurm_%j.out
# #SBATCH --nodes=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=10G
# #SBATCH --gres=gpu:Titan_RTX:1

# set -e
# set -x

# echo "Job started at $(date)"

# module load anaconda3
# module load cuda

# source ~/.bashrc

# conda activate pytorchuse

# # Specify the dataset to exclude
# export EXCLUDE_DATASET="Task501_HIE"

# # Run training
# echo "Running train.sh"
# ./train.sh > train_${SLURM_JOB_ID}.log 2>&1

# # Run validation after training
# echo "Running validation.sh"
# ./validation.sh > validation_${SLURM_JOB_ID}.log 2>&1

# echo "Job finished at $(date)"

