#!/bin/bash
#
#SBATCH -p debug              # Partition
#SBATCH --qos=debug              # QOS
#SBATCH --job-name=medbcos_genexpl_isic2018         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "ISIC2018 Database | Evaluation | Started"

echo "Baseline DenseNet121"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/isic-2018-db/" python B-cos-v2-main/generate_visual_explanations.py --reload weights --weights "/nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/ISIC2018/baseline_densenet121/baseline_densenet121/best.ckpt" --explainer_name OursRelative --batch_size 1 --save_path "experiments/ISIC2018/baseline_densenet121/baseline_densenet121/"

echo "B-cos Networks V2"
DATA_ROOT="/nas-ctm01/datasets/public/MEDICAL/isic-2018-db/" python B-cos-v2-main/generate_visual_explanations.py --reload weights --weights "/nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/ISIC2018/bcos_densenet121/densenet121/best.ckpt" --explainer_name OursRelative --batch_size 1 --save_path "experiments/ISIC2018/bcos_densenet121/densenet121/"

echo "ISIC2018 Database | Evaluation | Finished"
