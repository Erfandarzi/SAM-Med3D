#!/bin/bash
set -e
set -x

# Function to run validation
run_validation() {
    local RUN_NAME=$1
    local CHECKPOINT_PATH=$2
    local EXCLUDE_DATASET=$3

    # Define all datasets
    ALL_DATASETS=("Task501_HIE" "Task502_BCHUNC" "Task503_INRIA" "Task504_ATLAS" "Task505_ISLES" "Task506_NWMH" "Task507_JHU" "Task508_ISLES2022")

    # Remove previous validation results
    rm -rf "./results/${RUN_NAME}_validation"

    # Validate on all datasets
    for dataset in "${ALL_DATASETS[@]}"
    do
        echo "Validating on $dataset"
        python validation.py --seed 2023 \
         -vp "./results/${RUN_NAME}_validation/${dataset}" \
         -cp "${CHECKPOINT_PATH}" \
         -tdp "./data/validation/${dataset}" -nc 5 \
         --save_name "./results/sam_${RUN_NAME}_validation_${dataset}.py"
    done

    echo "Validation completed for all datasets"
}

# Validate adaptive sampling model
echo "Validating adaptive sampling model"
run_validation "run_adaptive_sampling" "./work_dir/run_adaptive_sampling/sam_model_latest.pth" ""

# Validate model with excluded dataset
EXCLUDE_DATASET="Task501_HIE"
echo "Validating model with excluded dataset: ${EXCLUDE_DATASET}"
run_validation "run_4_exclude_${EXCLUDE_DATASET}" "./work_dir/run_4_exclude_${EXCLUDE_DATASET}/sam_model_latest.pth" "${EXCLUDE_DATASET}"

# Optionally, validate run3 if needed
# echo "Validating run3"
# run_validation "run_3" "./work_dir/run_3/sam_model_latest.pth" ""