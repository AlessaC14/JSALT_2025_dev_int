#!/bin/bash -l
#SBATCH --job-name=SAE
#SBATCH --time=12:00:00
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --output=train_logs/SAE_%j.out
#SBATCH --error=train_logs/SAE_%j.err

#SBATCH -A enalisn1_gpu



# Load Conda Environment
conda activate mechanistic_int

mkdir -p train_logs

python /home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int/SAE/new_SAE.py
     


