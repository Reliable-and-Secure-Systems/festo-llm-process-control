# Exp1 — Baseline Fine-Tuning (SP + CoT + FS)

## Overview

Exp1 is the **baseline closed-loop control experiment** for the Festo MPS PA dual-tank system.

The experiment fine-tunes `meta-llama/Llama-3.2-3B-Instruct` using QLoRA on **11,975 training records**.

**Paper reference:** Exp1, Row 1 in Table I.

### Prompting Strategy

**SP + CoT + FS**

* **SP:** System Prompt
* **CoT:** Chain-of-Thought
* **FS:** Few-Shot examples
* **P:** Not used

---

## 1. System

The controlled system is a simulated Festo MPS PA dual-tank process.

| Variable          | Target | Acceptable Range |
| ----------------- | -----: | ---------------: |
| Upper tank level  |    45% |           40–50% |
| Water temperature |   35°C |          32–38°C |

The controller receives the current sensor state and a **6-step history** and produces a structured JSON control action.

The control loop operates at a **5-second decision interval**.

A hardware-safety layer enforces:

* Heater OFF when temperature > 38°C
* Pump OFF and valve OPEN when level > 70%
* No simultaneous pump + valve operation

---

## 2. Dataset

Exp1 uses **11,975 training records**:

* **11,475 normal records**
* **500 anti-hallucination correction records**
* **11,975 total records**

The training data contains current process-state information and **6-step history**.

No predictive horizon is used in Exp1.

The dataset is stored as:

```text
dataset.jsonl
```

---

## 3. Model and Fine-Tuning

Base model:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Fine-tuning method:

```text
QLoRA
```

### Training Configuration

| Parameter               |       Value |
| ----------------------- | ----------: |
| Training records        |      11,975 |
| Quantization            |   4-bit NF4 |
| Double quantization     |     Enabled |
| Compute dtype           |        FP16 |
| LoRA rank               |          16 |
| LoRA alpha              |          32 |
| LoRA dropout            |        0.05 |
| rsLoRA                  |     Enabled |
| Optimizer               | AdamW 8-bit |
| Epochs                  |           5 |
| Learning rate           |        2e-4 |
| Batch size              |           1 |
| Gradient accumulation   |          16 |
| Warmup steps            |         100 |
| Maximum sequence length |         512 |
| Random seed             |        3407 |

The fine-tuned adapter is stored under:

```text
festo_llama3.2_finetuned_exp05/
```

---

## 4. Prompting Strategy

Exp1 corresponds to:

```text
SP + CoT + FS
```

The corresponding inference configuration is:

```text
PROMPT_VARIANT=v4_fewshot
```

### SP — System Prompt

Defines the controller role, process targets, actuator constraints, and required JSON output format.

### CoT — Chain-of-Thought

Instructs the model to assess the process state and reason about the appropriate control action before producing the final action.

### FS — Few-Shot

Provides worked examples demonstrating the expected control behaviour and structured JSON response.

### Predictive Horizon

**Not used in Exp1.**

Therefore:

```text
Exp1 = Row 1 = SP + CoT + FS
```

---

# Reproduction Procedure

## 5. Environment Setup

Use a Python environment containing:

* PyTorch
* Transformers
* PEFT
* TRL
* BitsAndBytes
* Datasets

A CUDA-capable GPU is required for QLoRA fine-tuning.

The base model is:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Configure Hugging Face access using a local `.env` file:

```text
HUGGING_FACE_KEY=<your-token>
```

**Do not commit `.env` or the Hugging Face token to GitHub.**

---

## 6. Generate Training Data

From the `Exp1` directory:

```bash
python generate_training_data.py
```

The generator uses random seed:

```text
3407
```

The generated dataset should contain:

```text
11,475 normal records
+ 500 anti-hallucination records
= 11,975 total records
```

Output:

```text
festo_live/dataset.jsonl
```

---

## 7. Fine-Tune the Model

Run:

```bash
python finetune.py
```

The resulting LoRA adapter is saved under:

```text
festo_llama3.2_finetuned_exp05/
```

`finetune.sh` is a BTU/Slurm-specific wrapper and is not required when reproducing the experiment on another system.

---

## 8. Start the Festo Simulator

Start the simulator:

```bash
./simulated_festo
```

Keep the simulator running.

The simulator continuously updates the process state used by the LLM controller.

---

## 9. Run Exp1

Select the Exp1 prompting strategy:

```text
PROMPT_VARIANT=v4_fewshot
```

Run:

```bash
PROMPT_VARIANT=v4_fewshot python -u llm_supervisor.py
```

This corresponds to:

```text
SP + CoT + FS
```

For each control cycle, the supervisor:

1. Reads the current sensor state.
2. Maintains the 6-step history.
3. Builds the SP + CoT + FS prompt.
4. Generates the LLM control action.
5. Parses the JSON response.
6. Applies the hardware-safety layer.
7. Sends the control command to the simulator.
8. Records the decision and metrics.
9. Repeats every 5 seconds.

The current supervisor executes a maximum of **250 decisions**.

---

## 10. Inference Configuration

| Parameter          |             Value |
| ------------------ | ----------------: |
| Prompt strategy    | **SP + CoT + FS** |
| Prompt variant     |      `v4_fewshot` |
| Predictive horizon |          Not used |
| History window     |           6 steps |
| Decision interval  |         5 seconds |
| Maximum decisions  |               250 |
| Max new tokens     |               180 |
| Sampling           |          Disabled |
| Temperature        |   Greedy decoding |

The controller uses:

```text
do_sample = False
```

---

# 11. Evaluation

Evaluation is performed through the **live closed-loop simulation**.

### Level Hit Condition

```text
40% ≤ level ≤ 50%
```

### Temperature Hit Condition

```text
32°C ≤ temperature ≤ 38°C
```

### Overall Accuracy

A decision is counted as an overall hit only when both level and temperature are within their respective acceptable ranges.

The experiment records:

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

# 12. Exp1 Results

The reported Exp1 run contains **250 live closed-loop decisions**.

| Metric                     |        Result |
| -------------------------- | ------------: |
| Training records           |        11,975 |
| Decisions                  |           250 |
| Level accuracy             |         70.4% |
| Temperature accuracy       |         86.4% |
| Overall accuracy           |     **65.2%** |
| Level MAE                  |        10.69% |
| Temperature MAE            |        1.75°C |
| Time to target             |      4 cycles |
| Longest stable run         | **20 cycles** |
| Average latency            |      6,937 ms |
| Safety override rate       |      **0.0%** |
| Hallucination rate         |        100.0% |
| Control hallucination rate |      **0.0%** |
| Text hallucination rate    |        100.0% |

### Hallucination Breakdown

| Type          | Count |
| ------------- | ----: |
| Parse error   |     0 |
| Missing field |     0 |
| Wrong type    |     0 |
| Out of range  |     0 |
| Wrong key     |     0 |
| Extra content |   250 |

The reported **100% hallucination rate** is entirely due to `extra_content` being detected in all 250 responses. The **control hallucination rate is 0%**, and the **safety override rate is also 0%**.

---

# 13. Output Files

The actual Exp1 `festo_live` directory contains:

```text
festo_live/
├── data.bin
├── dataset.jsonl
├── job_1335_exp05.log
├── job_1344_exp05.log
├── job_2490.log
├── json_data.txt
├── llm_control.json
├── sensor_log.json
├── summary_stats.json
└── unified_log.jsonl
```

### Output Description

| File                 | Purpose                          |
| -------------------- | -------------------------------- |
| `data.bin`           | Simulator data file              |
| `dataset.jsonl`      | Exp1 training dataset            |
| `job_1335_exp05.log` | Slurm execution log              |
| `job_1344_exp05.log` | Slurm execution log              |
| `job_2490.log`       | Slurm execution log              |
| `json_data.txt`      | Live/current process state       |
| `llm_control.json`   | Latest actuator command          |
| `sensor_log.json`    | Recorded sensor data             |
| `summary_stats.json` | Final Exp1 evaluation metrics    |
| `unified_log.jsonl`  | Unified closed-loop decision log |

The **primary result files** for analysis are:

```text
summary_stats.json
unified_log.jsonl
sensor_log.json
```

The `job_*.log` files are execution/Slurm logs.

---

# 14. Reproduction Summary

```text
1. Set up Python/CUDA environment
        ↓
2. Configure HUGGING_FACE_KEY in local .env
        ↓
3. Run generate_training_data.py
        ↓
4. Verify 11,975 training records
        ↓
5. Run finetune.py
        ↓
6. Start simulated_festo
        ↓
7. Set PROMPT_VARIANT=v4_fewshot
        ↓
8. Run llm_supervisor.py
        ↓
9. Execute up to 250 live closed-loop decisions
        ↓
10. Inspect summary_stats.json
        ↓
11. Inspect unified_log.jsonl and sensor_log.json
```

## Experiment Definition

**Exp1 = Row 1 = SP + CoT + FS**

**Training data:** 11,975 records
**Live evaluation:** 250 decisions
**Predictive horizon:** Not used
**Overall accuracy:** 65.2%
**Safety override rate:** 0.0%
**Control hallucination rate:** 0.0%
