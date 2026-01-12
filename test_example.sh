#!/bin/bash
#SBATCH --job-name=test_multistep
#SBATCH --output=hydrogen-test-multistep_%j.out
#SBATCH --error=hydrogen-test-multistep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

# Load necessary modules here
module load hydrogen-shared

python -m emulator_1ts.main \
    --mode test \
    --config /home/ga6/hydrogen-emulator-1ts/runs/UpperEel_box.wy2003_config.yaml \
    --log-level verbose \
    --save_inputs
