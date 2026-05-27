#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:student:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_supervisor_%j.out
#SBATCH --error=/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_supervisor_%j.err
#SBATCH --job-name=experiment_b_supervisor

set -e

cd /scratch/rayarvid/Experiments/experiment_b
mkdir -p logs
mkdir -p festo_live

LOG_FILE="/scratch/rayarvid/Experiments/experiment_b/logs/experiment_b_supervisor_${SLURM_JOB_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Job $SLURM_JOB_ID started: $(date) ==="
echo "Node: $(hostname)"

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch/rayarvid/hf_cache
export HF_HUB_CACHE=/scratch/rayarvid/hf_cache/hub
export TRANSFORMERS_CACHE=/scratch/rayarvid/hf_cache/hub

nvidia-smi

# ── Step 1: Start C++ simulation in background ──
echo "=== Starting C++ simulation... ==="
/scratch/rayarvid/Experiments/cpp_sim/simulated_festo &
CPP_SIM_PID=$!
echo "C++ sim PID: $CPP_SIM_PID"

# ── Step 2: Wait until sensor data is being written ──
echo "Waiting for sensor data to appear..."
SENSOR_FILE="/scratch/rayarvid/Experiments/experiment_b/festo_live/json_data.txt"
until [ -f "$SENSOR_FILE" ] && [ -s "$SENSOR_FILE" ]; do
    sleep 1
done
echo "Sensor data detected. Starting supervisor variants..."
sleep 3  # small buffer to let sim stabilise

# ── Step 3: Run all 4 variants sequentially ──
for VARIANT in v1_plain v2_system v3_cot v4_fewshot; do
    echo "=== Starting variant: $VARIANT at $(date) ==="
    PROMPT_VARIANT=$VARIANT /scratch/rayarvid/.venv/bin/python -u llm_supervisor_experiment_b.py &
    SUPERVISOR_PID=$!
    # 125 decisions x ~21s/decision (16s inference + 5s interval) + 60s model load = ~2685s
    # sleep 3200 is a safety-net kill; supervisor exits naturally via MAX_DECISIONS=125
    sleep 3200
    kill $SUPERVISOR_PID 2>/dev/null || true
    wait $SUPERVISOR_PID 2>/dev/null || true
    echo "=== Finished variant: $VARIANT at $(date) ==="
    sleep 10
done

# ── Step 4: Stop C++ simulation ──
echo "Stopping C++ simulation..."
kill $CPP_SIM_PID 2>/dev/null || true
wait $CPP_SIM_PID 2>/dev/null || true

echo "=== All variants complete. Job $SLURM_JOB_ID finished: $(date) ==="
