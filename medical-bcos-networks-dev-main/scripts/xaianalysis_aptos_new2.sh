#!/bin/bash
# METHODS=("IntGrad" "Grad" "IxG")
METHODS=("Ours")

echo "APTOS2019 Database | XAI Metrics Analysis | Started"
BASELINES=(
    # "/media/TOSHIBA6T/ICRTO/medical-bcos/new_experiments/baseline_densenet121_1e-4/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt" 
    # "/media/TOSHIBA6T/ICRTO/medical-bcos/new_experiments/bcos_densenet121/APTOS/bcos_densenet121/densenet121/best.ckpt" 
    # "/media/TOSHIBA6T/ICRTO/medical-bcos/new_experiments/bcos_densenet_warm_1.0_new_base/APTOS/bcos_densenet121_tl_1_0/densenet121/best.ckpt" 
    # "/media/TOSHIBA6T/ICRTO/medical-bcos/pretrained_experiments/baseline_densenet121/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt" 
    "/media/TOSHIBA6T/ICRTO/medical-bcos/pretrained_experiments/bcos_densenet121_tl_1_0_bcos_imagenet/APTOS/bcos_densenet121_tl_1_0/densenet121/best.ckpt" 
    "/media/TOSHIBA6T/ICRTO/medical-bcos/pretrained_experiments/bcos_densenet121_tl_1_0_aptos_baseline_imagenet/APTOS/bcos_densenet121_tl_1_0/densenet121/best.ckpt" 
)
IMGS_FILE="results/aptos_correct_imgs.json"

for b in "${BASELINES[@]}"
do
    echo "Baseline DenseNet121"
    echo "GCam"
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/xai_eval.py --explainer_name "GCam" --weights $b --batch_size 1 --num_workers 8 --imgs_file $IMGS_FILE

    for m in "${METHODS[@]}"
    do
        echo $m
        DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights $b --batch_size 8 --num_workers 8  --imgs_file $IMGS_FILE
        # DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/interpretability/analyses/localisation.py --reload weights --weights "./experiments/baseline_densenet121_lr1e-5_aug/ISIC2018/baseline_densenet121/baseline_densenet121/best.ckpt" --analysis_config 50_2x20.5 --explainer_name $m --smooth 15 --batch_size 1 --save_path "./experiments/baseline_densenet121_lr1e-5_aug/ISIC2018/baseline_densenet121/baseline_densenet121/"
    done

    echo "APTOS2019 Database | XAI Metrics Analysis | Finished"
done
