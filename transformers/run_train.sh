#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --time=24:00:00
#SBATCH --mem=500G
#SBATCH --output=train_logs/transformer%j.out
#SBATCH --error=train_logs/transformer%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4


# Print node and GPU info
echo "Running on node: $(hostname)"
echo "Available GPUs:"
nvidia-smi || echo "WARNING: nvidia-smi failed"

mkdir -p train_


source ~/.bashrc
conda activate mechanistic_int

python /home/acarbol1/scratchenalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int/transformers/train.py