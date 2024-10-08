#!/bin/bash
set -e
set -x

# Specify the excluded dataset
EXCLUDE_DATASET="Task501_HIE"

# Define all datasets
ALL_DATASETS=("Task501_HIE" "Task502_BCHUNC" "Task503_INRIA" "Task504_ATLAS" "Task505_ISLES" "Task506_NWMH" "Task507_JHU" "Task508_ISLES2022")

# Remove previous validation results
rm -rf ./results/run_4_exclude_${EXCLUDE_DATASET}_validation

# Validate on all datasets
for dataset in "${ALL_DATASETS[@]}"
do
    echo "Validating on $dataset"
    python validation.py --seed 2023 \
     -vp "./results/run_4_exclude_${EXCLUDE_DATASET}_validation/${dataset}" \
     -cp "./work_dir/run_4_exclude_${EXCLUDE_DATASET}/sam_model_latest.pth" \
     -tdp "./data/validation/${dataset}" -nc 5 \
     --save_name "./results/sam_run_4_exclude_${EXCLUDE_DATASET}_validation_${dataset}.py"
done

echo "Validation completed for all datasets"

# Optionally, you can keep the validation on run3 if needed
# echo "Validating run3"
# rm -rf ./results/run_3_validation
# python validation.py --seed 2023\
#  -vp ./results/run_3_validation\
#  -cp ./work_dir/run_3/sam_model_latest.pth \
#  -tdp ./data/validation -nc 5 \
#  --save_name ./results/sam_run_3_validation.py