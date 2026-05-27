# Experiment A — Baseline Fine-Tuning (No Predictive Context)

## Overview

Experiment A is the baseline. A Llama 3.2-3B-Instruct model is fine-tuned using QLoRA on 11,975 training records generated from a physics-informed simulator of the Festo MPS PA dual-tank process. No predictive horizon is used. The model receives the current sensor state and recent history, and outputs a structured JSON control action. This experiment establishes the performance ceiling of the fine-tuned model under the full cumulative prompt setup (system prompt + chain of thought + few-shot examples) without any predictive context.

**Paper reference**: Experiment A in Table I.

---

## System Description

The controlled system is a simulated Festo MPS PA dual-tank workstation with two control objectives:

- **Level control**: Keep upper tank level at 45% (acceptable band: 40–50%)
- **Temperature control**: Keep water temperature at 35°C (acceptable band: 32–38°C)

The LLM receives sensor readings every 5 seconds and outputs a JSON action with four fields:
- `pump_power` (integer 0–100)
- `upper_valve_open` (boolean)
- `heater_on` (boolean)
- `reason` (short string)

A safety validation layer checks every action before applying it to the simulator. If the action violates hardware constraints, the safety fallback intervenes.

---

## Model

| Property | Value |
|----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Double quantization | Enabled |
| Compute dtype | float16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| rsLoRA | Enabled |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Training records | 11,975 |
| Epochs | 5 |
| Learning rate | 2e-4 |
| Batch size (per device) | 1 |
| Warmup steps | 100 |
| Save steps | 400 |
| Eval steps | 200 |
| Best model selection | Disabled (load_best_model_at_end=False) |
| max_new_tokens (inference) | 180 |

---

## Dataset

- **Total records**: 11,975
- **File**: `festo_live/llm_training_dataset_experiment_a.jsonl`
- Covers scenarios including: on-target maintenance, level critical low, level above target, temperature cold/warm/overshoot, combined faults, float switch edge cases, and anti-hallucination correction pairs
- No predictive horizon context in training data

---

## Prompting Strategy

### Fine-Tuning Prompt
**SP+CoT+FS** — System Prompt + Chain of Thought + Few-Shot examples (cumulative, all three components)

### Inference Prompt
**SP+CoT+FS** — same as fine-tuning prompt

The variants build cumulatively:
- **SP** (System Prompt): role definition, output schema, control targets
- **+CoT** (Chain of Thought): step-by-step reasoning instruction (assess level → assess temperature → decide)
- **+FS** (Few-Shot): 3 worked examples showing correct JSON output

No predictive context (P) is used in this experiment.

---

## Evaluation

- **Decisions evaluated**: 185
- **Decision interval**: 5 seconds
- **Level hit condition**: upper tank within ±5% of 45% target (i.e. 40–50%)
- **Temperature hit condition**: water temperature within ±3°C of 35°C target (i.e. 32–38°C)
- **Overall accuracy**: percentage of steps where both level AND temperature conditions are satisfied simultaneously
- **Format Hallucination (Fmt H)**: model returns invalid or unparseable JSON
- **Content Hallucination (Cnt H)**: model returns valid JSON but commands unsafe values requiring safety fallback

---

## Results

| FT Prompt | Inf Prompt | Level% | Temp% | Overall% | Level MAE | Temp MAE | Fmt H% | Cnt H% |
|-----------|-----------|--------|-------|----------|-----------|----------|--------|--------|
| SP+CoT+FS | SP+CoT+FS | 65.4 | 85.9 | 60.0 | 11.57% | 1.83°C | 0.0 | 0.0 |

Additional metrics:
- Average inference latency: 14,392 ms
- Time to target (cycles): 3
- Longest stable run (cycles): 15

---

## Key Observations

1. Zero format and content hallucination across all 185 decisions — the model reliably produces valid, safe JSON under the full cumulative prompt.
2. Temperature accuracy (85.9%) significantly exceeds level accuracy (65.4%) — the model handles thermal control more reliably without predictive context.
3. Overall accuracy of 60.0% with 0% safety overrides is the strongest safety performance across all three experiments.
4. This result serves as the baseline. Experiments B and C introduce predictive horizon context to investigate whether forward-looking prompting improves control.
