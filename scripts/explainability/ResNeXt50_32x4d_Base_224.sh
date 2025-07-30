#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=ResNeXt50_32x4d_Base_224.out
#SBATCH --error=ResNeXt50_32x4d_Base_224.err

VIS_DIR="visualizations/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p $VIS_DIR 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=0  

python src/main_image.py \
 --visualizations_path "$VIS_DIR" \
 --gpu_id 0 \
 --pickles_path 'pickles/E' \
 --verbose \
 --train_or_test 'test' \
 --visualize_all \
 --max_visualizations 1000 \
 --generate_xai \
 --results_path 'results' \
 --checkpoint_path 'results/2025-06-19_06-57-17,ResNext_E' \
 --xai_batch_size 1
echo "Finished"
