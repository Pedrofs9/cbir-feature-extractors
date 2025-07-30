#!/bin/bash

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_0.01/APTOS/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_0.01

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_0.1/APTOS/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_0.1

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_0.5/APTOS/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_0.5

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet121/APTOS/bcos_densenet121/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet121

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.01_new_base/APTOS/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.01_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.1_new_base/APTOS/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.1_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.5_new_base/APTOS/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.5_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_1.0_new_base/APTOS/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_1.0_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_01 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.01_feats_only_new_base/APTOS/bcos_densenet121_tl_0_01/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.01_feats_only_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_1 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.1_feats_only_new_base/APTOS/bcos_densenet121_tl_0_1/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.1_feats_only_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_0_5 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_0.5_feats_only_new_base/APTOS/bcos_densenet121_tl_0_5/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_0.5_feats_only_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network bcos_densenet121_tl_1_0 --experiment_name densenet121 --reload weights --weights new_experiments/bcos_densenet_warm_1.0_feats_only_new_base/APTOS/bcos_densenet121_tl_1_0/densenet121/best.ckpt --base_directory new_experiments/bcos_densenet_warm_1.0_feats_only_new_base

DATA_ROOT="/media/TOSHIBA6T/ICRTO/DATASETS/APTOS2019/" CUDA_VISIBLE_DEVICES=0 python B-cos-v2-main/evaluate.py --dataset APTOS --base_network baseline_densenet121 --experiment_name baseline_densenet121 --reload weights --weights new_experiments/baseline_densenet121_1e-4/APTOS/baseline_densenet121/baseline_densenet121/best.ckpt --base_directory new_experiments/baseline_densenet121_1e-4
