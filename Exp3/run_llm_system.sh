#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=job_%j.log

export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep -oE "MIG-[0-9a-f-]+" | head -1)
echo "CVD=$CUDA_VISIBLE_DEVICES"

echo "========== PYTORCH CUDA CHECK =========="
/scratch/rayarvid/.venv/bin/python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
"
echo

echo "========== BITSANDBYTES CHECK =========="
/scratch/rayarvid/.venv/bin/python -m bitsandbytes
echo

set -e
cd /scratch/rayarvid/Experiments/cpp_sim
mkdir -p /scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live

echo "Starting simulation..."
./simulated_festo &
SIM_PID=$!

echo "Starting LLM supervisor (v3_cot)..."
PROMPT_VARIANT=v3_cot /scratch/rayarvid/.venv/bin/python -u /scratch/rayarvid/Experiments/Experiment_2_equal_cycles/llm_supervisor_exp07_equal_cycles.py

echo "Starting LLM supervisor (v4_fewshot)..."
PROMPT_VARIANT=v4_fewshot /scratch/rayarvid/.venv/bin/python -u /scratch/rayarvid/Experiments/Experiment_2_equal_cycles/llm_supervisor_exp07_equal_cycles.py

kill $SIM_PID 2>/dev/null || true