#!/bin/bash
#SBATCH -p GPU
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=train-weight
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=zcqsfhu@ucl.ac.uk
#SBATCH --mail-type=ALL

set -euo pipefail

echo "===== [1/7] Job started ====="
date
echo "Running on host: $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $SLURMD_NODENAME"
echo "============================================="

# Move to the directory containing the script + Python file
cd "$SLURM_SUBMIT_DIR"

# Ensure logs directory exists
mkdir -p logs

echo "===== [2/7] Sourcing environment ====="
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Environment sourced successfully."
echo "============================================="

echo "===== [3/7] Thread configuration ====="
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "============================================="

echo "===== [4/7] Checking GPU visibility ====="
nvidia-smi || echo "WARNING: No NVIDIA GPU found!"
echo "============================================="

echo "===== [5/7] Python environment ====="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
echo "============================================="

echo "===== [6/7] Starting training script ====="
echo "Command: python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml"
start_time=$(date +%s)

python weight_ONLY_TRAINS.py configs/config_ONLY_TRAIN.yaml

end_time=$(date +%s)
echo "Training script finished successfully."
echo "Duration: $(( (end_time - start_time) / 60 )) minutes"
echo "============================================="

echo "===== [7/7] Job completed ====="
date

