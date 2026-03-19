#!/bin/bash

# Submission command:   sbatch submit_slurm_make_data.sh

#SBATCH --job-name=50%makedata
#SBATCH --array=0-0 # for 1 slice
#SBATCH -p COMPUTE
#SBATCH -n 4
#SBATCH --time=96:00:00
#SBATCH --mem=256G

#SBATCH --mail-user=zcqsfhu@ucl.ac.uk
#SBATCH --mail-type=ALL

# Note: Added %a to distinguish log files for each slice in the array
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/jet2m_ParT/logs/slurm-makedata-%j-%a.out

# ------------------------------------------
# Environment and Paths
# ------------------------------------------
cd /home/xzcqsfhu/lundtoptagger
echo "Now in $(pwd)"

# Source the Miniforge initialization script
source /share/data1/xucaphue/setup.sh

# Activate your specific project environment
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126

# ------------------------------------------
# Parameters
# ------------------------------------------
main_config="configs/config_make_data_SRJ.yaml"
signal_config="configs/config_signal_SRJ.yaml"

# This pulls the index (0, 1, 2, 3, or 4) from the SBATCH array
event_fraction_idx=${SLURM_ARRAY_TASK_ID}

# ------------------------------------------
# Run MakeData
# ------------------------------------------
# Ensure the filename below matches your actual file case (e.g., make_data_SRJ.py)
python Make_data_SRJ.py "$main_config" --override \
    signal_config_file="$signal_config" \
    signal="srj" \
    event_fraction_idx="$event_fraction_idx"

echo "Job done for slice index $event_fraction_idx."