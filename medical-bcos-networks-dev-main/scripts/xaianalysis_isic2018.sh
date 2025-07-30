#!/bin/bash

METHODS=("GCam" "IntGrad" "Grad" "IxG")
echo "ISIC2018 Database | XAI Metrics Analysis | Started"

echo "Baseline DenseNet121"
for m in "${METHODS[@]}"
do
    echo $m
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights "./experiments/baseline_densenet121_lr1e-5_aug/ISIC2018/baseline_densenet121/baseline_densenet121/best.ckpt"
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/baseline_densenet121_lr1e-5_aug/ISIC2018/baseline_densenet121/baseline_densenet121/best.ckpt" --analysis_config 50_2x20.5 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/baseline_densenet121_lr1e-5_aug/ISIC2018/baseline_densenet121/baseline_densenet121/"
done

METHODS+=("Ours")
echo "B-cos Networks V2"
for m in "${METHODS[@]}"
do
    echo $m
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights "./experiments/bcos_densenet121_lr1e-4_aug/ISIC2018/bcos_densenet121/densenet121/best.ckpt"
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/bcos_densenet121_lr1e-4_aug/ISIC2018/bcos_densenet121/densenet121/best.ckpt" --analysis_config 50_2x20.5 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/bcos_densenet121_lr1e-4_aug/ISIC2018/bcos_densenet121/densenet121/"
done
echo "ISIC2018 Database | XAI Metrics Analysis | Finished"
