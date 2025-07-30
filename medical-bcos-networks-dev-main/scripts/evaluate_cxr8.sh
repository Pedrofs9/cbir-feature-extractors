#!/bin/bash
#
#SBATCH -p rtx2080ti_11GB              # Partition
#SBATCH --qos=rtx2080ti                # QOS
#SBATCH --job-name=medbcos_eval_cxr8         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CXR8 Database | Evaluation | Started"

echo "Baseline DenseNet121"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/" python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network baseline_densenet121 --experiment_name baseline_densenet121 --reload weights --weights /nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/CXR8/baseline_densenet121/baseline_densenet121/best.ckpt

echo "B-cos Networks V2"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/" python B-cos-v2-main/evaluate.py --dataset CXR8 --base_network bcos_densenet121 --experiment_name densenet121 --reload weights --weights /nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/CXR8/bcos_densenet121/densenet121/best.ckpt

echo "CXR8 Database | Evaluation | Finished"
