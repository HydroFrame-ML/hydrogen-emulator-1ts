#!/bin/bash
#SBATCH --job-name=train-multistep
#SBATCH --output=train-multistep_%j.out
#SBATCH --error=train-multistep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00

# Load necessary modules here
echo "Starting multi-timestep autoregressive training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -m emulator_1ts.main \
    --mode train \
    --config convnext_multistep_config.yaml \
    --log-level verbose

echo "Multi-timestep training completed."