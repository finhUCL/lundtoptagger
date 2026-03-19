#!/bin/bash
#SBATCH --job-name=50%_preprocess
#SBATCH --partition=COMPUTE       # CPU 分区
# exclude nodes that do NOT mount /share/lustre properly

#SBATCH -N 1
#SBATCH --cpus-per-task=16                # 16 CPU cores
#SBATCH --mem=256G              # 50GB 内存（KDE 用得上）
#SBATCH --export=ALL
#SBATCH --time=96:00:00         
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/jet2m_ParT/logs/slurm_stat_preprocess-%j.out
#SBATCH --error=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/jet2m_ParT/logs/slurm_stat_preprocess-%j.err 
#SBATCH --mail-user=zcqsfhu@ucl.ac.uk
#SBATCH --mail-type=ALL
# submit by sbatch submit_preprocess.sh

echo "===> Hostname:"
hostname

echo "===> Activating environment"
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Using environment: $CONDA_DEFAULT_ENV"

echo "===> Running CPU preprocess script..."
cd /home/xzcqsfhu/lundtoptagger

python preprocess_SRJ_CPU.py configs/config_preprocess_SRJ.yaml

echo "===> Job finished."