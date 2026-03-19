#!/bin/bash

# =================================================================
# SLURM DIRECTIVES (Resource Requests)
# =================================================================

# Job name
#SBATCH --job-name=qg_make_data

# Specify the partition/queue (gpu is usually safe for DIAS)
#SBATCH -p RCIF

# Request one node and one GPU
#SBATCH -N1
#SBATCH --gres=gpu:1

# Keep environment variables and shell settings
#SBATCH --export=ALL

# Request 4 CPUs and 35GB of memory (conservative, safe estimate)
#SBATCH -n4
#SBATCH --mem=35G

# Output and Error logs (redirected to your analysis folder)
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/slurm-%j.out
#SBATCH --error=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/slurm-%j.err

# Email notifications
#SBATCH --mail-user=[zcqsfhu@ucl.ac.uk]
#SBATCH --mail-type=BEGIN,END,FAIL

# =================================================================
# JOB EXECUTION
# =================================================================

# 1. Change to the directory where your script and configs are located
cd /home/xzcqsfhu/lundtoptagger
echo "Working directory: $(pwd)"

# 2. Activate your Conda/Miniforge environment (THIS IS THE CRUCIAL PART)
echo "Activating Python environment..."

# Source the Miniforge initialization script
source /share/data1/xucaphue/setup.sh

# Activate your specific project environment by name
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126

echo "Conda env: $CONDA_DEFAULT_ENV"

# 3. Print status
echo "Hostname: $(hostname)"
echo "Running on GPU: $CUDA_VISIBLE_DEVICES"

# 4. Run the data creation script
echo "Starting data creation with make_data_SRJ.py..."
python Make_data_SRJ.py configs/config_make_data_SRJ.yaml

echo "Job finished."
