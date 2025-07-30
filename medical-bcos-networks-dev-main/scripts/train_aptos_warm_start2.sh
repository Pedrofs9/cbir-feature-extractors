#!/bin/bash

PERC=("1.0" "0.5" "0.1" "0.01")
BASELINE="/home/icrto/Documents/medical-bcos-networks-dev/new_experiments/baseline_densenet121_1e-4/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt"


for p in "${PERC[@]}"
do
    echo $p
    p_name="${p/./"_"}"
    # cmd="DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 \
    # python B-cos-v2-main/train.py \
    # --dataset APTOS \
    # --base_network bcos_densenet121_tl_${p_name} \
    # --experiment_name densenet121 \
    # --wandb_logger \
    # --wandb_project medical-bcos \
    # --wandb_name aptos2019_bcos_densenet121_warm_${p} \
    # --base_directory "/home/icrto/Documents/medical-bcos-networks-dev/experiments/bcos_densenet_warm_${p}" \
    # --pretrained_weights $BASELINE"

    # eval $cmd

    cmd="DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=1 \
    python B-cos-v2-main/train.py \
    --dataset APTOS \
    --base_network bcos_densenet121_tl_${p_name} \
    --experiment_name densenet121 \
    --wandb_logger \
    --wandb_project medical-bcos-final \
    --wandb_name aptos2019_bcos_densenet121_warm_${p}_feats_only_new_base \
    --base_directory "new_experiments/bcos_densenet_warm_${p}_feats_only_new_base" \
    --pretrained_weights $BASELINE
    --features_only"

    eval $cmd
done

