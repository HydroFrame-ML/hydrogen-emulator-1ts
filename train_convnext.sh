#!/bin/bash
#SBATCH --job-name=train-convnext
#SBATCH --output=hydrogen-train-convnext_%j.out
#SBATCH --error=hydrogen-train-convnext_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

# Load necessary modules here
python ./emulator-1ts/main.py \
    --mode train \
    --config convnext_config.yaml \
    --log-level verbose