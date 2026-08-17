````markdown
# Exp2 — Predictive Look-Ahead (SP + CoT + FS + P)

## Overview

Exp2 evaluates the effect of adding **predictive look-ahead** to the LLM closed-loop controller.

The experiment uses the fine-tuned Llama 3.2 3B model and adds a **3-step predictive horizon** to the prompt. The predicted future process states are provided to the LLM so that it can use forward-looking information when selecting its control action.

**Paper reference:** Exp2, Table I.

### Prompting strategy

The experiment preserves the four cumulative prompting variants from the baseline and adds predictive look-ahead (**P**) to all of them:

| Variant | Strategy |
|---|---|
| `v1_plain` | P |
| `v2_system` | SP + P |
| `v3_cot` | SP + CoT + P |
| `v4_fewshot` | SP + CoT + FS + P |

For the reported Exp2 result, the active configuration is the corresponding variant selected through `PROMPT_VARIANT`.

---

## 1. System

The controlled system is a simulated Festo MPS PA dual-tank process.

| Variable | Target | Acceptable range |
|---|---:|---:|
| Upper tank level | 45% | 40–50% |
| Water temperature | 35°C | 32–38°C |

The controller receives the current process state and a **6-step history**.

The control loop operates at a **5-second decision interval**.

Hardware safety constraints are enforced independently of the LLM:

- Heater OFF when temperature > 38°C
- Pump OFF and valve OPEN when level > 70%
- No simultaneous pump + valve operation

---

## 2. Predictive Look-Ahead

Exp2 introduces a **3-step prediction horizon**.

For every control decision, the supervisor simulates three future states using the nominal process dynamics and includes them in the LLM prompt:

```text
Predicted future (if recommended action applied):
t+1: level=..., temp=...
t+2: level=..., temp=...
t+3: level=..., temp=...
````

The prediction uses the configured process parameters:

```text
Prediction horizon: 3 steps
Fill rate: 3.27
Drain rate: 1.58
Heat rate: 0.074
Cool rate: 0.0173
Thermal inertia: 3
Inertia factor: 0.3
Float cutoff: 47%
```

The predictive horizon is added to **all four prompting variants**.

---

## 3. Repository Contents

```text
Exp2/
├── README.md
├── ...
└── festo_live/
```

The experiment's Slurm `.sh` files, where present, are BTU-specific execution wrappers. They are not required for reproducing the experiment on another system.

---

# Reproduction Procedure

## 4. Environment Setup

Use a Python environment containing:

* PyTorch
* Transformers
* PEFT
* TRL
* BitsAndBytes
* Datasets

A CUDA-capable GPU is required.

The base model is:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Hugging Face access is required.

Store the Hugging Face token locally in `.env`:

```text
HUGGING_FACE_KEY=<your-token>
```

Do **not** commit `.env` to GitHub.

---

## 5. Fine-Tuned Model

Exp2 loads the fine-tuned Llama 3.2 adapter from:

```text
/scratch/rayarvid/Experiments/exp07/festo_llama3.2_finetuned_exp07
```

The supervisor automatically searches this directory for `checkpoint-*` directories and selects the checkpoint with the highest checkpoint number.

The base model is loaded using:

* 4-bit NF4 quantization
* Double quantization
* FP16 computation

The adapter is loaded using PEFT.

---

## 6. Start the Festo Simulator

Start the Festo simulation:

```bash
./simulated_festo
```

Keep the simulator running.

The simulator provides the live sensor state consumed by the LLM supervisor.

---

## 7. Select the Prompting Variant

The supervisor supports:

```text
v1_plain
v2_system
v3_cot
v4_fewshot
```

For example:

```bash
PROMPT_VARIANT=v4_fewshot python -u llm_supervisor_exp07.py
```

The default is:

```text
v4_fewshot
```

The predictive horizon is automatically added regardless of which variant is selected.

---

## 8. Run the LLM Controller

From a second terminal:

```bash
PROMPT_VARIANT=v4_fewshot python -u llm_supervisor_exp07.py
```

For every decision, the supervisor:

1. Reads the current sensor state.
2. Maintains the most recent 6-step history.
3. Computes a 3-step future-state prediction.
4. Adds the prediction to the prompt.
5. Queries the fine-tuned Llama model.
6. Parses the generated JSON control action.
7. Applies the hardware-safety constraints.
8. Sends the resulting actuator command to the simulator.
9. Records the decision and metrics.
10. Repeats after 5 seconds.

The experiment terminates automatically after:

**250 decisions.**

---

## 9. Inference Configuration

The LLM uses:

| Parameter          |     Value |
| ------------------ | --------: |
| Max new tokens     |       250 |
| Sampling           |  Disabled |
| Temperature        |       1.0 |
| Top-p              |       1.0 |
| History window     |   6 steps |
| Prediction horizon |   3 steps |
| Decision interval  | 5 seconds |

Greedy decoding is used with:

```text
do_sample = False
```

The JSON generation is primed with an opening `{"` before generation.

---

## 10. Evaluation

Evaluation is performed through the **live closed-loop simulation**.

### Hit conditions

Level:

```text
40% ≤ level ≤ 50%
```

Temperature:

```text
32°C ≤ temperature ≤ 38°C
```

Overall accuracy requires both conditions to be satisfied simultaneously.

The supervisor records:

* Level control accuracy
* Temperature control accuracy
* Overall accuracy
* Level MAE
* Temperature MAE
* Time to target
* Longest stable run
* Average inference latency
* Safety override rate
* Hallucination rate
* Control hallucination rate
* Text hallucination rate
* Hallucination types

---

# 11. Results

The reported Exp2 run contains **250 decisions**.

| Metric                     |    Result |
| -------------------------- | --------: |
| Decisions                  |       250 |
| Level accuracy             |     86.8% |
| Temperature accuracy       |     59.2% |
| Overall accuracy           |     51.2% |
| Level MAE                  |     4.73% |
| Temperature MAE            |    2.75°C |
| Time to target             |  7 cycles |
| Longest stable run         |  5 cycles |
| Average latency            | 10,465 ms |
| Safety override rate       |     37.6% |
| Hallucination rate         |    100.0% |
| Control hallucination rate |      5.6% |
| Text hallucination rate    |     96.8% |

### Hallucination breakdown

| Type          | Count |
| ------------- | ----: |
| Parse error   |     8 |
| Missing field |     0 |
| Wrong type    |     6 |
| Out of range  |     0 |
| Wrong key     |     0 |
| Extra content |   242 |

---

## 12. Output Files

Summary statistics and unified logs are separated by prompting variant.

For example, a `v4_fewshot` run produces:

```text
festo_live/
├── json_data.txt
├── llm_control.json
├── llm_summary_stats_v4_fewshot.json
├── llm_unified_log_v4_fewshot.jsonl
└── sensor_log.json
```

The possible variant-specific summary files are:

```text
llm_summary_stats_v1_plain.json
llm_summary_stats_v2_system.json
llm_summary_stats_v3_cot.json
llm_summary_stats_v4_fewshot.json
```

The corresponding unified logs are:

```text
llm_unified_log_v1_plain.jsonl
llm_unified_log_v2_system.jsonl
llm_unified_log_v3_cot.jsonl
llm_unified_log_v4_fewshot.jsonl
```

---

## 13. Reproduction Summary

```text
1. Set up Python/CUDA environment
        ↓
2. Configure Hugging Face access in .env
        ↓
3. Ensure the fine-tuned Exp2 adapter is available
        ↓
4. Start simulated_festo
        ↓
5. Select PROMPT_VARIANT
        ↓
6. Run llm_supervisor_exp07.py
        ↓
7. Allow the controller to execute 250 decisions
        ↓
8. Inspect the corresponding summary_stats JSON
        ↓
9. Inspect the corresponding unified_decision log
```

This reproduces **Exp2 — Predictive Look-Ahead**, with a **3-step prediction horizon added to the selected prompting strategy**.

```
```
