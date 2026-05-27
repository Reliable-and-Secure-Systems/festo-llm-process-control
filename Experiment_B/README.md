# Experiment B — Receding Horizon Prompting, 9,000 Training Records

## Overview

Experiment B introduces a 3-step receding horizon predictive context into both the fine-tuning data and the inference prompt. The model is trained on 9,000 records where each training example includes a simulation of the process 3 steps ahead under the recommended action. At inference time, the same predicted future states are included in the user prompt so the model can reason about whether its chosen action will converge to target over the horizon. Two inference prompt strategies are evaluated on the same fine-tuned adapter.

**Paper reference**: Experiment B in Table I (two rows).

---

## System Description

Same as Experiment A:
- **Level control**: Keep upper tank level at 45% (acceptable band: 40–50%)
- **Temperature control**: Keep water temperature at 35°C (acceptable band: 32–38°C)
- JSON output: `pump_power`, `upper_valve_open`, `heater_on`, `reason`
- Safety validation layer checks every action before applying

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
| Training records | 9,000 (8,500 normal + 500 anti-hallucination correction pairs) |
| Epochs | 5 |
| Learning rate | 2e-4 |
| Batch size (per device) | 1 |
| Warmup steps | 100 |
| Save steps | 400 |
| Eval steps | 200 |
| Best model selection | Enabled (load_best_model_at_end=True, metric: eval_loss) |
| max_new_tokens (inference) | 250 |

---

## Dataset

- **Total records**: 9,000
- **Normal records**: 8,500
- **Anti-hallucination correction pairs**: 500
- **File**: `dataset_experiment_b.jsonl`

### Predictive Horizon in Training Data

For each training record, all candidate actions (pump ±10%, valve flip, heater flip combinations) are simulated 3 steps ahead. The action minimising the horizon cost is selected:

```
cost = 2.0 × |level_error| + 1.0 × |temp_error|   (level-weighted)
```

The 3-step predicted future states (level and temperature at each step) are included in the training prompt in the format:
```
Predicted future (if recommended action applied): t+1: level=X%, temp=YC | t+2: ... | t+3: ...
```

**Note**: The cost function weights level error twice as heavily as temperature error. This is the key design difference from Experiment C which uses equal weights.

---

## Prompting Strategy

### Fine-Tuning Prompt
**SP+CoT+FS+P** — System Prompt + Chain of Thought + Few-Shot examples + Predictive Context

### Inference Prompts Tested
Two inference strategies evaluated on the same fine-tuned adapter:

1. **SP+CoT+P** — System Prompt + Chain of Thought + Predictive Context (no few-shot)
2. **SP+CoT+FS+P** — System Prompt + Chain of Thought + Few-Shot + Predictive Context (full)

The variants build cumulatively:
- **SP**: role definition, output schema, control targets, predictive horizon instruction
- **+CoT**: step-by-step reasoning: (1) assess level error, (2) assess temp error, (3) check predicted future states to confirm convergence
- **+FS**: 3 worked examples with predicted future states included
- **+P**: 3-step predicted future states in the user message at every decision step

---

## Evaluation

- **Decisions evaluated**: 125 per inference strategy
- **Decision interval**: 5 seconds
- **History window**: last 6 steps shown in prompt
- **Prediction horizon**: 3 steps ahead
- **JSON priming**: `{"` prepended to generation prompt to force JSON completion
- **Level hit condition**: upper tank within ±5% of 45% target (40–50%)
- **Temperature hit condition**: water temperature within ±3°C of 35°C target (32–38°C)
- **Overall accuracy**: percentage of steps where both conditions satisfied simultaneously
- **Format Hallucination (Fmt H)**: model returns invalid or unparseable JSON
- **Content Hallucination (Cnt H)**: model returns valid JSON but commands unsafe values requiring safety fallback

---

## Results

| FT Prompt | Inf Prompt | Level% | Temp% | Overall% | Level MAE | Temp MAE | Fmt H% | Cnt H% |
|-----------|-----------|--------|-------|----------|-----------|----------|--------|--------|
| SP+CoT+FS+P | SP+CoT+P | 100.0 | 60.8 | 60.8 | 1.89% | 2.74°C | 0.0 | 39.2 |
| SP+CoT+FS+P | SP+CoT+FS+P | 40.0 | 56.8 | 25.6 | 27.80% | 2.91°C | 48.0 | 25.6 |

### SP+CoT+P (best performing)
- Average inference latency: 9,050 ms
- Time to target (cycles): 1
- Longest stable run (cycles): 4

### SP+CoT+FS+P
- Average inference latency: 16,077 ms
- Time to target (cycles): 1
- Longest stable run (cycles): 4

---

## Key Observations

1. **SP+CoT+P achieves 100% level accuracy** — the predictive horizon with chain-of-thought dramatically improves level control compared to Experiment A (65.4%).
2. **Temperature accuracy drops to 60.8%** (from 85.9% in Experiment A) — the level-weighted cost function (2× level, 1× temp) in training data caused the model to deprioritise temperature.
3. **Content hallucination rises to 39.2%** with SP+CoT+P — the safety fallback intervenes in nearly 4 out of 10 decisions, a significant increase from 0% in Experiment A.
4. **SP+CoT+FS+P collapses** — adding few-shot examples to an already long predictive prompt exceeds the 3B model's effective context capacity. Overall accuracy drops to 25.6% with 48% format hallucination.
5. The overall accuracy of 60.8% matches Experiment A (60.0%) but at the cost of much higher safety override rates, making the safety layer essential rather than precautionary.
