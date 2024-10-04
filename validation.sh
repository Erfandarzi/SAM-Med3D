
#!/bin/bash
set -e
set -x
# #Validating run1
# rm -rf ./results/run_2_validation

# python validation.py --seed 2023\
#  -vp ./results/run_2_validation\
#  -cp ./work_dir/run_2/sam_model_loss_best.pth \
#  -tdp ./data/validation -nc 5 \
#  --save_name ./results/sam_run_2_validation.py


#Validating run3

rm -rf ./results/run_3_validation

python validation.py --seed 2023\
 -vp ./results/run_3_validation\
 -cp ./work_dir/run_3/sam_model_latest.pth \
 -tdp ./data/validation -nc 5 \
 --save_name ./results/sam_run_3_validation.py


# #Validating sam_med3d
# rm -rf ./results/sam_med3d_validation

# python validation.py --seed 2023\
#  -vp ./results/sam_med3d_validation\
#  -cp ./ckpt/sam_med3d.pth \
#  -tdp ./data/validation -nc 5 \
#  --save_name ./results/sam_med3d_validation.py





