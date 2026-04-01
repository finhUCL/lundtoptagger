#!/bin/bash

# ------------------------------------------------------------
#  sbatch submit/submit_slurm_scores_combiner.sh
# ------------------------------------------------------------

#SBATCH --job-name=SRJ_Combiner_score  # Job name
#SBATCH --time=24:00:00                # Time limit hrs:min:sec
#SBATCH -p GPU                         # GPU partition
#SBATCH -N1                            # single node
#SBATCH -n8                            # 8 CPU cores
#SBATCH --gres=gpu:1                   # request 1 GPU
#SBATCH --mem=50G                      # 50 GB RAM

# Email notification (optional)
#SBATCH --mail-user=zcqsfhu@ucl.ac.uk
#SBATCH --mail-type=ALL

# Log files
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/jet2m_ParT/logs/slurm-%j.%x.out

# ------------------------------------------------------------
#  Environment setup
# ------------------------------------------------------------

echo "===> Current directory before cd:"
pwd

# Move into SRJ project folder
cd /home/xzcqsfhu/lundtoptagger/
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
#  Run combined scoring
# ------------------------------------------------------------

CONFIG_FILE="configs/config_make_scores_combinerV2_SRJ.yaml"

echo ""
echo "===> Starting SRJ combined scoring..."
echo "Using config: $CONFIG_FILE"

python make_scores_combiner_SRJ.py "$CONFIG_FILE"

echo "===> Job finished."