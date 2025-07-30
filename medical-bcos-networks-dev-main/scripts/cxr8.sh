#!/bin/bash

# DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.01/CXR8/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.01

# DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.1/CXR8/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.1

# DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_0.5/CXR8/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_0.5

# DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet121_lr1e-4_aug/CXR8/bcos_densenet121/densenet121/best.ckpt --base_directory experiments/bcos_densenet121_lr1e-4_aug

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.01/CXR8/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.01

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.1/CXR8/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.1

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.5/CXR8/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.5

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_1.0/CXR8/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_1.0

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.01_feats_only/CXR8/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.01_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.1_feats_only/CXR8/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.1_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_0.5_feats_only/CXR8/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_0.5_feats_only

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/CXR8/" CUDA_VISIBLE_DEVICES=1 python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights experiments/bcos_densenet_warm_1.0_feats_only/CXR8/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory experiments/bcos_densenet_warm_1.0_feats_only