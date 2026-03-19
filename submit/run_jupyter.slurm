#!/bin/bash
#SBATCH -p COMPUTE
#SBATCH --job-name=jupyter-plotting
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/xzcqsfhu/lundtoptagger/qg_analysis_data/slurm_jupyter-%j.out

# set -euo pipefail

echo "===== [1/4] Job start ====="
date
echo "Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Node: ${SLURMD_NODENAME:-UNKNOWN}"
mkdir -p logs

# Work from the directory where you submitted the job
cd "$SLURM_SUBMIT_DIR"
echo "CWD: $(pwd)"
echo "==========================="

echo "===== [2/4] Environment ====="
# Your usual DIAS environment
source /share/data1/xucaphue/setup.sh
conda activate /share/data1/xucaphue/envs/pytorch_py39_cu126
echo "Python: $(python -V 2>&1)"
echo "which python: $(which python)"
echo "============================="

# Pick a port that is likely free on the node
PORT=8891

echo "===== [3/4] Jupyter instructions ====="
echo ""
echo "When this job is RUNNING on a node, do this on your local machine:"
echo ""
echo "  ssh -L ${PORT}:<node>:${PORT} xzcqsfhu@dias.hpc.phys.ucl.ac.uk"
echo ""
echo "where <node> is the node name from squeue (e.g. compute-gpu-0-1.local)."
echo ""
echo "Then open in your browser:"
echo "  http://localhost:${PORT}"
echo ""
echo "Jupyter will now start; token/URL printed below:"
echo "==============================================="

# Start Jupyter Lab (or notebook) WITHOUT opening a browser on the node
jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}"

echo "===== [4/4] Jupyter stopped, job ending ====="
date
