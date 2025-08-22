#!/bin/bash
#SBATCH --job-name=test-multistep
#SBATCH --output=test-multistep_%j.out
#SBATCH --error=test-multistep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00

# Load necessary modules here
echo "Starting multi-timestep autoregressive training test..."
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODEID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -m emulator_1ts.main \
    --mode train \
    --config test_multistep_config.yaml \
    --log-level verbose

echo "Multi-timestep training test completed."