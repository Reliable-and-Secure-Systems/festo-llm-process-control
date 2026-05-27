#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=job_%j.log

set -e
cd /scratch/rayarvid/cpp_sim
mkdir -p /scratch/rayarvid/experiment_a/festo_live

echo "Starting simulation..."
./simulated_festo &
SIM_PID=$!

echo "Starting LLM supervisor..."
/scratch/rayarvid/.venv/bin/python -u llm_supervisor_experiment_a.py

kill $SIM_PID 2>/dev/null || true
