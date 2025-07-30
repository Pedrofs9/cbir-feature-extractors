#!/bin/bash

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.01/ISIC2018/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.01

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.1/ISIC2018/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.1

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.5/ISIC2018/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.5

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet121_lr1e-4_aug/ISIC2018/bcos_densenet121/densenet121/best.ckpt --base_directory experiments/bcos_densenet121_lr1e-4_aug

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.01/ISIC2018/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.01

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.1/ISIC2018/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.1

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.5/ISIC2018/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.5

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_1.0/ISIC2018/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_1.0

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.01_feats_only/ISIC2018/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.01_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.1_feats_only/ISIC2018/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.1_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.5_feats_only/ISIC2018/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.5_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/ISIC2018/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_1.0_feats_only/ISIC2018/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_1.0_feats_only