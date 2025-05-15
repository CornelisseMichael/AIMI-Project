#!/bin/bash
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

#virtual environment
source /vol/csedu-nobackup/course/IMC037_aimi/group01/baseline/bin/activate

cd /vol/csedu-nobackup/course/IMC037_aimi/group01/luna25-vit


python train.py
