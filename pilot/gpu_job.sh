#!/bin/bash
# Submitted by the dashboard: run_task.py on a GPU node.
#   sbatch gpu_job.sh <task> <argfile>
#SBATCH --job-name=fltwin
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/FLTwin/pilot/work/slurm-%j.log

# H100 only: the installed torch build has no kernels for the V100s (sm_70),
# so a plain --gres=gpu:1 can land on a node where CUDA fails at model load.
BASE=/lustre/fs1/home/mu290328/github-repos/FLTwin
cd "$BASE/pilot" || exit 1
export MPLBACKEND=Agg
export YOLO_CONFIG_DIR="$BASE/pilot/work"
nvidia-smi --query-gpu=name --format=csv,noheader
exec "$BASE/.venv/bin/python" run_task.py "$1" "$2"
