#!/bin/bash
#
#SBATCH -p gtx1080_8GB               # Partition
#SBATCH --qos=gtx1080                # QOS
#SBATCH --job-name=resize_isic2018   # Job name
#SBATCH -o slurm.%N.%j.out           # STDOUT
#SBATCH -e slurm.%N.%j.err           # STDERR



echo "ISIC2018 Database | Resize | Started"

python code/data_resize.py --original_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images/task3/ISIC2018_Task3_Test_Input --new_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images-resized/task3/ISIC2018_Task3_Test_Input --new_height 448
python code/data_resize.py --original_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images/task3/ISIC2018_Task3_Training_Input --new_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images-resized/task3/ISIC2018_Task3_Training_Input --new_height 448
python code/data_resize.py --original_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images/task3/ISIC2018_Task3_Validation_Input --new_path /nas-ctm01/datasets/public/MEDICAL/isic-2018-db/images-resized/task3/ISIC2018_Task3_Validation_Input --new_height 448

echo "ISIC2018 Database | Resize | Finished"
