#!/bin/bash

PERC=("1.0")
BASELINE="/home/icrto/Documents/medical-bcos-networks-dev/experiments/baseline_densenet121_lr1e-5_aug/CXR8/baseline_densenet121/baseline_densenet121/best.ckpt"


for p in "${PERC[@]}"
do
    echo $p
    p_name="${p/./"_"}"
    cmd="DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=0 \
    python B-cos-v2-main/train.py \
    --dataset CXR8 \
    --base_network bcos_densenet121_tl_${p_name} \
    --experiment_name densenet121 \
    --wandb_logger \
    --wandb_project medical-bcos \
    --wandb_name cxr8_bcos_densenet121_warm_${p} \
    --base_directory "/home/icrto/Documents/medical-bcos-networks-dev/experiments/bcos_densenet_warm_${p}" \
    --pretrained_weights $BASELINE"

    eval $cmd

    cmd="DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=0 \
    python B-cos-v2-main/train.py \
    --dataset CXR8 \
    --base_network bcos_densenet121_tl_${p_name} \
    --experiment_name densenet121 \
    --wandb_logger \
    --wandb_project medical-bcos \
    --wandb_name cxr8_bcos_densenet121_warm_${p}_feats_only \
    --base_directory "/home/icrto/Documents/medical-bcos-networks-dev/experiments/bcos_densenet_warm_${p}_feats_only" \
    --pretrained_weights $BASELINE \
    --features_only"

    eval $cmd
done

