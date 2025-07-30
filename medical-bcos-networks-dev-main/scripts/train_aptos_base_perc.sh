#!/bin/bash

PERC=("0.5" "0.1" "0.01")

for p in "${PERC[@]}"
do
    echo $p
    p_name="${p/./"_"}"
    cmd="DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 \
    python B-cos-v2-main/train.py \
    --dataset APTOS \
    --base_network baseline_densenet121_${p_name} \
    --experiment_name baseline_densenet121 \
    --wandb_logger \
    --wandb_project medical-bcos-final \
    --wandb_name aptos2019_baseline_densenet121_${p} \
    --base_directory "new_experiments/baseline_densenet_${p}" "
    eval $cmd
done

