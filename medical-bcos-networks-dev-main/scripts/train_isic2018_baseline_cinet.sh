#!/bin/bash
#
#SBATCH -p a100_80GB              # Partition
#SBATCH --qos=a100                # QOS
#SBATCH --job-name=isic2018_baseline_cinet         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "ISIC2018 Database | Baseline CINet | Started"

DATA_ROOT="/nas-ctm01/homes/icrto/ISIC2018/" python B-cos-v2-main/train.py --dataset 'ISIC2018CINet' --base_network 'baseline_cinet' --experiment_name 'baseline_cinet' --wandb_logger --wandb_project medical-bcos --wandb_name isic2018_baseline_cinet --base_directory "/nas-ctm01/homes/icrto/medical-bcos-networks-dev/experiments/baseline_cinet" \

echo "ISIC2018 Database | Baseline CINet | Finished"
