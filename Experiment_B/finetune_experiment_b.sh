#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_%j.out
#SBATCH --error=/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_%j.err
#SBATCH --job-name=experiment_b

set -e

cd /scratch/rayarvid/Experiments/experiment_b
mkdir -p logs

LOG_FILE="/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Job $SLURM_JOB_ID started: $(date) ==="
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rayarvid/hf_cache
export HF_HUB_CACHE=/scratch/rayarvid/hf_cache/hub
export TRANSFORMERS_CACHE=/scratch/rayarvid/hf_cache/hub

nvidia-smi

echo "[1/2] Generating dataset..."
/scratch/rayarvid/.venv/bin/python generate_training_data_experiment_b.py

echo "[2/2] Fine-tuning..."
/scratch/rayarvid/.venv/bin/python finetune_experiment_b.py

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
