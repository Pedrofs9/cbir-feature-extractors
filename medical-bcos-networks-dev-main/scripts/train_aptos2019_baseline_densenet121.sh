#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=aptos2019_baseline_densenet121         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "APTOS2019 Database | Baseline DenseNet121 | Started"

DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/aptos-2019-db/" python B-cos-v2-main/train.py --dataset APTOS --base_network baseline_densenet121 --experiment_name baseline_densenet121 --wandb_logger --wandb_project medical-bcos --wandb_name aptos2019_baseline_densenet121

echo "APTOS2019 Database | Baseline DenseNet121 | Finished"
