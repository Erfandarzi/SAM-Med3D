#!/bin/bash
set -e
set -x

echo "Starting validation of adaptive sampling model at $(date)"

RUN_NAME="run_adaptive_sampling"
CHECKPOINT_PATH="./work_dir/${RUN_NAME}/sam_model_latest.pth"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint file not found: ${CHECKPOINT_PATH}"
    exit 1
fi

rm -rf "./results/${RUN_NAME}_validation"

echo "Available memory before validation:"
free -h

echo "GPU information:"
nvidia-smi

echo "Python and CUDA versions:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"

# Run validation script
python validation.py \
    --test_data_path './data/validation' \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --save_name './results/sam_run_adaptive_sampling_validation.py' \
    --num_clicks 5 \
    --dim 3

echo "Validation completed for adaptive sampling model at $(date)"

echo "Available memory after validation:"
free -h