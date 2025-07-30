#!/bin/bash
METHODS=("Ours" "IntGrad" "Grad" "IxG")

BASELINE="/media/TOSHIBA6T/ICRTO/medical-bcos/pretrained_experiments/Herlev/bcos_densenet121/densenet121/best.ckpt"
IMGS_FILE="results/herlev_correct_imgs.json"

# echo "GCam"
DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/HERLEV/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name "GCam" --weights $BASELINE --batch_size 1 --num_workers 8 --imgs_file $IMGS_FILE

for m in "${METHODS[@]}"
do
    echo $m
    DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/HERLEV/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/xai_eval.py --explainer_name $m --weights $BASELINE --batch_size 8 --num_workers 8  --imgs_file $IMGS_FILE
done

