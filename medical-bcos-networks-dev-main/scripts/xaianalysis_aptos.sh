#!/bin/bash

METHODS=("IntGrad" "Grad" "IxG")
echo "APTOS2019 Database | XAI Metrics Analysis | Started"

# echo "Baseline DenseNet121"
# DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name "GCam" --weights "/home/icrto/Documents/medical-bcos-networks-dev/new_experiments/baseline_densenet121_1e-4/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt" --imgs_file "new_experiments/aptos_correct_imgs.json" --batch_size 1

# for m in "${METHODS[@]}"
# do
#     echo $m
#     DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights "/home/icrto/Documents/medical-bcos-networks-dev/new_experiments/baseline_densenet121_1e-4/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt" --imgs_file "new_experiments/aptos_correct_imgs.json"
#     # DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/baseline_densenet121_lr1e-4_aug/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt" --analysis_config 15_2x20.5 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/baseline_densenet121_lr1e-4_aug/APTOS/baseline_densenet121/baseline_densenet121/"
# done

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/xai_eval.py --explainer_name "GCam" --weights "new_experiments/bcos_densenet121/APTOS/bcos_densenet121/densenet121/best.ckpt" --imgs_file "new_experiments/aptos_correct_imgs.json" --batch_size 1

METHODS+=("Ours")
echo "B-cos Networks V2"
for m in "${METHODS[@]}"
do
    echo $m
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights "new_experiments/bcos_densenet121/APTOS/bcos_densenet121/densenet121/best.ckpt" --imgs_file "new_experiments/aptos_correct_imgs.json"
    # DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/bcos_densenet121_lr1e-4_aug/APTOS/bcos_densenet121/densenet121/best.ckpt" --analysis_config 15_2x20.5 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/bcos_densenet121_lr1e-4_aug/APTOS/bcos_densenet121/densenet121/"
done
echo "APTOS2019 Database | XAI Metrics Analysis | Finished"
