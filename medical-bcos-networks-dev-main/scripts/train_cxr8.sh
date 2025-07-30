#!/bin/bash
#
#SBATCH -p v100_32GB              # Partition
#SBATCH --qos=v100                # QOS
#SBATCH --job-name=bcosv2_cxr8_densenet121         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CXR8 Database | B-cos Networks V2 | Started"

DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/" python B-cos-v2-main/train.py --dataset CXR8 --base_network bcos_densenet121 --experiment_name densenet121 --wandb_logger --wandb_project medical-bcos --wandb_name cxr8_bcos_densenet121

echo "CXR8 Database | B-cos Networks V2 | Finished"
