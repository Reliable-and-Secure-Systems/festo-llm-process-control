# Experiment C — Equal-Weight Multivariable MPC, 3,000 Training Records

## Overview

Experiment C addresses the structural flaw identified in Experiment B: the training cost function weighted level error twice as heavily as temperature error, causing the model to systematically sacrifice temperature control. Experiment C uses an equal-weight cost function (1.5× level, 1.5× temp) so the training data reflects actions that optimise both variables equally over the 3-step horizon. Training data is reduced to 3,000 records to isolate the effect of the cost function change from data quantity. The same two inference strategies as Experiment B are evaluated.

**Paper reference**: Experiment C in Table I (two rows).

---

## System Description

Same as Experiments A and B:
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
| Training records | 3,000 (2,750 normal + 250 anti-hallucination correction pairs) |
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

- **Total records**: 3,000
- **Normal records**: 2,750
- **Anti-hallucination correction pairs**: 250
- **File**: `dataset_experiment_c.jsonl`

### Key Change from Experiment B — Equal-Weight Cost Function

The horizon cost function used to select the best training action is changed to equal weight:

**Experiment B (level-weighted):**
```
cost = 2.0 × |level_error| + 1.0 × |temp_error|
```

**Experiment C (equal-weight):**
```
cost = 1.5 × |level_error| + 1.5 × |temp_error|
```

This means training actions are selected by equally minimising both level and temperature errors over the 3-step horizon.

### Additional Changes from Experiment B

1. **Few-shot examples updated** — predicted future states now explicitly label both level and temperature at each step:
   ```
   t+1: level=13.3%, temp=29.8C | t+2: level=16.6%, temp=30.0C | t+3: level=19.9%, temp=30.2C
   ```
   (Experiment B only showed level percentages in abbreviated form)

2. **Reason labels updated** — training labels now explicitly report horizon convergence for both variables:
   - `"both converging"` when both level and temperature are approaching target
   - `"level converging, temp stable"` when only level is converging

3. **CoT instruction updated** — step 3 now reads:
   > check the predicted future states for BOTH level AND temperature convergence

---

## Prompting Strategy

### Fine-Tuning Prompt
**SP+CoT+FS+P** — System Prompt + Chain of Thought + Few-Shot examples + Predictive Context (equal-weight MPC)

### Inference Prompts Tested
Two inference strategies evaluated on the same fine-tuned adapter:

1. **SP+CoT+P** — System Prompt + Chain of Thought + Predictive Context (no few-shot)
2. **SP+CoT+FS+P** — System Prompt + Chain of Thought + Few-Shot + Predictive Context (full)

The variants build cumulatively:
- **SP**: role definition, output schema, control targets, predictive horizon instruction
- **+CoT**: step-by-step reasoning: (1) assess level error, (2) assess temp error, (3) check predicted future states for BOTH level AND temperature convergence
- **+FS**: 3 worked examples with labeled level+temp predictions at each horizon step
- **+P**: 3-step predicted future states in the user message at every decision step

---

## Evaluation

- **Decisions evaluated**: 116–118 per inference strategy (slight variation due to timing)
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
| SP+CoT+FS+P | SP+CoT+P | 84.7 | 67.8 | 57.6 | 4.10% | 2.59°C | 0.0 | 32.2 |
| SP+CoT+FS+P | SP+CoT+FS+P | 86.2 | 61.2 | 52.6 | 4.06% | 2.46°C | 0.0 | 38.8 |

### SP+CoT+P (best performing)
- Average inference latency: 21,957 ms
- Time to target (cycles): 1
- Longest stable run (cycles): 8

### SP+CoT+FS+P
- Average inference latency: 22,307 ms
- Time to target (cycles): 1
- Longest stable run (cycles): 4

---

## Key Observations

1. **Best temperature accuracy across all experiments** — SP+CoT+P achieves 67.8% temperature accuracy, up from 60.8% in Experiment B. Equal cost weighting directly improved temperature control.
2. **Zero format hallucination across both inference strategies** — the model consistently produces valid, parseable JSON.
3. **Content hallucination remains elevated** — 32.2% with SP+CoT+P, confirming that predictive context increases safety fallback intervention regardless of cost function design.
4. **SP+CoT+FS+P again degrades performance** — adding few-shot examples consistently reduces accuracy across Experiments B and C, confirming that the combined length of predictive context and few-shot examples exceeds the effective reasoning capacity of the 3B model.
5. **Temperature MAE improved across the board** — 2.46–2.59°C compared to 2.74–2.91°C in Experiment B, confirming the equal-weight cost function reduced temperature errors even where accuracy did not change dramatically.
6. **Level accuracy reduced vs Experiment B** — 84.7% vs 100%, expected since level no longer receives 2× priority in the training objective.
