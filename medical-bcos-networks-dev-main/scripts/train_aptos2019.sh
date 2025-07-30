#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=bcosv2_aptos2019_densenet121         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "APTOS2019 Database | B-cos Networks V2 | Started"

DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/aptos-2019-db/" python B-cos-v2-main/train.py --dataset APTOS --base_network bcos_densenet121 --experiment_name densenet121 --wandb_logger --wandb_project medical-bcos --wandb_name aptos2019_bcos_densenet121

echo "APTOS2019 Database | B-cos Networks V2 | Finished"
