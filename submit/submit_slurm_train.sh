#!/bin/bash

# ------------------------------------------------------------
#  SLURM config
# ------------------------------------------------------------

#SBATCH --job-name=QLund_train
#SBATCH -p GPU                     # GPU partition
#SBATCH -N1                        # single node
#SBATCH -n8                        # 8 CPU cores
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1               # request 1 GPU

#SBATCH --mem=100G                  # 50 GB RAM

# Email notification (optional)
#SBATCH --mail-user=zcqsfhu@ucl.ac.uk
#SBATCH --mail-type=ALL

# Log files
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/jet2m_ParT/logs/slurm_train-%j.%x.out

# Submit by sbatch submit_slurm_train.sh

# ------------------------------------------------------------
#  Environment setup
# ------------------------------------------------------------

echo "===> Current directory before cd:"
pwd

# Move into SRJ project folder
cd /home/xzcqsfhu/lundtoptagger
echo "===> Switched directory to:"
pwd

echo "===> Hostname:"
hostname

echo "===> Activating conda environment..."
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Activated env: $CONDA_DEFAULT_ENV"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ------------------------------------------------------------
#  Run training
# ------------------------------------------------------------

echo ""
echo "===> Starting SRJ training..."
python weight_ONLY_TRAINS_SRJ.py configs/config_ONLY_TRAIN_SRJ.yaml

echo "===> Job finished."