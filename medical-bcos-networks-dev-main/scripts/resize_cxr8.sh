#!/bin/bash
#
#SBATCH -p gtx1080ti_11GB              # Partition
#SBATCH --qos=gtx1080ti                # QOS
#SBATCH --job-name=bcosv2_cxr8         # Job name
#SBATCH -o slurm.%N.%j.out             # STDOUT
#SBATCH -e slurm.%N.%j.err             # STDERR



echo "CXR8 Database | Resize | Started"

python code/data_resize.py --original_path /nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/images/ --new_path /nas-ctm01/datasets/public/MEDICAL/chestx-ray8-db/images-resized/ --new_height 448

echo "CXR8 Database | Resize | Finished"
