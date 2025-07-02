#!/bin/bash
#SBATCH --job-name=train-ucrb
#SBATCH --output=hydrogen-train-ucrb_%j.out
#SBATCH --error=hydrogen-train-ucrb_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

# Load necessary modules here
module load hydrogen-shared

python ./emulator-1ts/main.py \
    --mode test \
    --config /home/ga6/workspace/model_training/UpperEel_box.wy2003_config.yaml \
    --log-level verbose \
    --save_inputs
