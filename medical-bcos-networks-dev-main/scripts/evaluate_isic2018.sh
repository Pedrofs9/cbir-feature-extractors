#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti              # QOS
#SBATCH --job-name=medbcos_eval_isic2018         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "ISIC2018 Database | Evaluation | Started"

echo "Baseline DenseNet121"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/isic-2018-db/" python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network baseline_densenet121 --experiment_name baseline_densenet121 --reload weights --weights /nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/ISIC2018/baseline_densenet121/baseline_densenet121/best.ckpt

echo "B-cos Networks V2"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/isic-2018-db/" python B-cos-v2-main/evaluate.py --dataset ISIC2018 --base_network bcos_densenet121 --experiment_name densenet121 --reload weights --weights /nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/ISIC2018/bcos_densenet121/densenet121/best.ckpt

echo "ISIC2018 Database | Evaluation | Finished"
