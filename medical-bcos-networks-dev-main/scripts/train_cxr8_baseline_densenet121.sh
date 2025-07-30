#!/bin/bash
#
#SBATCH -p rtx2080ti_11GB              # Partition
#SBATCH --qos=rtx2080ti                # QOS
#SBATCH --job-name=bas_cxr8_densenet121         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CXR8 Database | Baseline DenseNet121 | Started"

DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/" python B-cos-v2-main/train.py --dataset CXR8 --base_network baseline_densenet121 --experiment_name baseline_densenet121 --wandb_logger --wandb_project medical-bcos --wandb_name cxr8_baseline_densenet121

echo "CXR8 Database | Baseline DenseNet121 | Finished"
