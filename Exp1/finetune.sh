#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/exp05_%j.out
#SBATCH --error=logs/exp05_%j.err
#SBATCH --job-name=exp05

set -e

cd /scratch/rayarvid/festo_llama3.2_exp05
mkdir -p logs

echo "=== Job $SLURM_JOB_ID started: $(date) ==="
echo "Node: $(hostname)"

export CUDA_VISIBLE_DEVICES=MIG-50e7ea79-5c28-5c08-8703-7df708730fd8
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rayarvid/hf_cache
export HF_HUB_CACHE=/scratch/rayarvid/hf_cache/hub
export TRANSFORMERS_CACHE=/scratch/rayarvid/hf_cache/hub

nvidia-smi

echo "[1/3] Generating dataset..."
/scratch/rayarvid/.venv/bin/python generate_dataset.py

echo "[2/3] Training..."
/scratch/rayarvid/.venv/bin/python train.py

echo "[3/3] Offline evaluation..."
/scratch/rayarvid/.venv/bin/python eval_offline.py --n 200

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
