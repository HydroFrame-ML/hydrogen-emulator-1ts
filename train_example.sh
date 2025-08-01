#!/bin/bash
#SBATCH --job-name=train-1ts
#SBATCH --output=hydrogen-train-1ts_%j.out
#SBATCH --error=hydrogen-train-1ts_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

# Load necessary modules here
python ./emulator-1ts/main.py \
    --mode train \
    --config example_config.yaml \
    --log-level verbose