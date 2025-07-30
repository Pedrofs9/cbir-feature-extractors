#!/bin/bash

METHODS=("IxG" "Grad")
echo "CXR8 Database | XAI Metrics Analysis | Started"


# METHODS+=("Ours")
echo "B-cos Networks V2"
for m in "${METHODS[@]}"
do
    echo $m
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights "./experiments/bcos_densenet121_lr1e-4_aug/CXR8/bcos_densenet121/densenet121/best.ckpt"
    # DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/bcos_densenet121_lr1e-4_aug/CXR8/bcos_densenet121/densenet121/best.ckpt" --analysis_config 10_2x20.0 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/bcos_densenet121_lr1e-4_aug/CXR8/bcos_densenet121/densenet121/"
done
echo "CXR8 Database | XAI Metrics Analysis | Finished"
