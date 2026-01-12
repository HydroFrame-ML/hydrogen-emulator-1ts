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

module load hydrogen-shared

python -m emulator_1ts.main \
    --mode test \
    --config /home/ga6/hydrogen-emulator-1ts/runs/PFCLM_3D_TB_34335_1_1_config.yaml \
    --log-level verbose \
    --save_inputs

echo "Multi-timestep training test completed."
