#!/bin/bash
#SBATCH --job-name=train-multistep_pfclm_tb
#SBATCH --output=train-multistep_pfclm_tb_%j.out
#SBATCH --error=train-multistep_pfclm_tb_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Load necessary modules here
echo "Starting multi-timestep autoregressive training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

module load hydrogen-shared

python -m emulator_1ts.main \
    --mode train \
    --config convnext_multistep_config_subsurface.yaml \
    --log-level verbose

echo "Multi-timestep training completed."
