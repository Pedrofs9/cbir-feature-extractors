#!/bin/bash
#
#SBATCH -p debug              # Partition
#SBATCH --qos=debug              # QOS
#SBATCH --job-name=medbcos_plotexpl_cxr8         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CXR8 Database | Plot Visual Explanations | Started"

echo "Baseline DenseNet121"
python B-cos-v2-main/plot_visual_explanations.py --model_type baseline --explanations_path "/nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/CXR8/baseline_densenet121/baseline_densenet121/visual_explanations/epoch_49/OursRelative/default"

echo "B-cos Networks V2"
python B-cos-v2-main/plot_visual_explanations.py --model_type bcos --explanations_path "/nas-ctm01/homes/tgoncalv/medical-bcos-networks-dev/experiments/CXR8/bcos_densenet121/densenet121/visual_explanations/epoch_40/OursRelative/default"

echo "CXR8 Database | Plot Visual Explanations | Finished"
