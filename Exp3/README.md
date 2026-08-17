````markdown
# Exp3 — Predictive Look-Ahead (SP + CoT + FS + P)

## Overview

Exp3 corresponds to **Row 3 in Table I** and evaluates the addition of predictive look-ahead to the full cumulative prompting strategy.

The fine-tuned Llama 3.2 3B controller receives the current process state, a 6-step history, and a **3-step predictive horizon**.

**Experiment strategy: SP + CoT + FS + P**

- **SP:** System Prompt
- **CoT:** Chain-of-Thought
- **FS:** Few-Shot examples
- **P:** 3-step predictive look-ahead

The experiment uses **9,000 training records** and **250 live closed-loop evaluation decisions**.

---

## 1. System

The controlled system is a simulated Festo MPS PA dual-tank process.

| Variable | Target | Acceptable Range |
|---|---:|---:|
| Upper tank level | 45% | 40–50% |
| Water temperature | 35°C | 32–38°C |

The controller operates at a **5-second decision interval** and maintains a **6-step history window**. :contentReference[oaicite:0]{index=0}

### Hardware Safety Limits

Three hardware safety constraints cannot be overridden by the LLM:

1. Temperature > 38°C → heater OFF
2. Level > 70% → pump OFF and valve OPEN
3. Pump + valve simultaneously → pump OFF :contentReference[oaicite:1]{index=1}

---

## 2. Dataset

Exp3 uses **9,000 training records** for fine-tuning.

Training records and live evaluation decisions are separate:

| Data | Count |
|---|---:|
| Training records | 9,000 |
| Live evaluation decisions | 250 |

---

## 3. Prompting Strategy

Exp3 uses the **full cumulative prompting configuration with predictive look-ahead**:

```text
SP + CoT + FS + P
````

### Components

**SP — System Prompt**

Defines the controller role, required JSON schema, actuator constraints, and control targets.

**CoT — Chain-of-Thought**

Instructs the model to assess the level, assess the temperature, check the predicted future states, and then produce the control action.

**FS — Few-Shot**

Provides three worked control examples demonstrating the expected JSON output and use of the predictive horizon.

**P — Predictive Look-Ahead**

Provides three predicted future process states:

```text
t+1
t+2
t+3
```

The corresponding implementation is `v4_fewshot`. The supervisor defines this variant as the cumulative `SP + CoT + FS` prompt with the predictive horizon added. 

---

## 4. Predictive Look-Ahead

The prediction horizon is **3 steps**.

For each control decision, future process states are calculated using the nominal process model and supplied to the LLM.

Configuration:

| Parameter          |   Value |
| ------------------ | ------: |
| Prediction horizon | 3 steps |
| Fill rate          |    3.27 |
| Drain rate         |    1.58 |
| Heat rate          |   0.074 |
| Cool rate          |  0.0173 |
| Thermal inertia    |       3 |
| Inertia factor     |     0.3 |
| Float cutoff       |     47% |



---

## 5. Model

Base model:

```text
meta-llama/Llama-3.2-3B-Instruct
```

The fine-tuned PEFT adapter is loaded from the Exp3 model directory, with the latest available checkpoint selected automatically. 

The base model uses:

* 4-bit NF4 quantization
* Double quantization
* FP16 computation



---

# Reproduction Procedure

## 6. Environment Setup

Use a CUDA-capable Python environment with the required PyTorch, Transformers, PEFT, and BitsAndBytes dependencies.

Configure Hugging Face access using a local `.env` file:

```text
HUGGING_FACE_KEY=<your-token>
```

**Do not commit `.env` to GitHub.**

---

## 7. Ensure the Fine-Tuned Model Is Available

Ensure the Exp3 fine-tuned adapter and its checkpoints are available in the experiment directory.

The supervisor automatically selects the latest `checkpoint-*` directory.

---

## 8. Start the Festo Simulator

Start the simulator:

```bash
./simulated_festo
```

Keep the simulator running.

---

## 9. Run Exp3

Exp3 uses:

```text
PROMPT_VARIANT=v4_fewshot
```

Run:

```bash
PROMPT_VARIANT=v4_fewshot python -u llm_supervisor_exp07.py
```

The controller then:

1. Reads the live sensor state.
2. Maintains the latest 6-step history.
3. Computes the 3-step predictive horizon.
4. Builds the **SP + CoT + FS + P** prompt.
5. Generates the LLM control action.
6. Parses the JSON response.
7. Applies hardware safety constraints.
8. Writes the actuator command.
9. Records the decision and metrics.
10. Waits 5 seconds before the next decision.



The run terminates automatically after **250 decisions**. 

---

## 10. Inference Configuration

| Parameter          |                 Value |
| ------------------ | --------------------: |
| Prompt strategy    | **SP + CoT + FS + P** |
| Prompt variant     |          `v4_fewshot` |
| History window     |               6 steps |
| Prediction horizon |               3 steps |
| Decision interval  |             5 seconds |
| Maximum decisions  |                   250 |
| Max new tokens     |                   250 |
| Sampling           |              Disabled |
| Temperature        |                   1.0 |
| Top-p              |                   1.0 |

Greedy decoding is used:

```text
do_sample = False
```



---

# 11. Evaluation

Evaluation is performed through the **live closed-loop simulation**.

### Hit Conditions

Level:

```text
40% ≤ level ≤ 50%
```

Temperature:

```text
32°C ≤ temperature ≤ 38°C
```

Overall accuracy requires both variables to be within their respective target bands.

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
* Overall hallucination rate
* Control hallucination rate
* Text hallucination rate
* Hallucination types



---

# 12. Results

The reported **Exp3 (SP + CoT + FS + P)** run contains **250 live closed-loop decisions**.

| Metric                     |    Result |
| -------------------------- | --------: |
| Training records           |     9,000 |
| Decisions                  |       250 |
| Level accuracy             |     58.4% |
| Temperature accuracy       |     62.0% |
| Overall accuracy           |     39.6% |
| Level MAE                  |    18.15% |
| Temperature MAE            |    2.78°C |
| Time to target             |  9 cycles |
| Longest stable run         |  7 cycles |
| Average latency            | 16,595 ms |
| Safety override rate       |     31.6% |
| Hallucination rate         |     97.6% |
| Control hallucination rate |     28.0% |
| Text hallucination rate    |     69.6% |

### Hallucination Breakdown

| Type          | Count |
| ------------- | ----: |
| Parse error   |    70 |
| Missing field |     0 |
| Wrong type    |     0 |
| Out of range  |     0 |
| Wrong key     |     0 |
| Extra content |   174 |

---

# 13. Output Files

The actual `festo_live` output directory contains:

```text
festo_live/
├── data.bin
├── json_data.txt
├── llm_control.json
├── sensor_log.json
├── summary_stats.json
└── unified_log.jsonl
```

| File                 | Purpose                          |
| -------------------- | -------------------------------- |
| `data.bin`           | Simulator data file              |
| `json_data.txt`      | Current/live sensor state        |
| `llm_control.json`   | Latest actuator command          |
| `sensor_log.json`    | Recorded sensor data             |
| `summary_stats.json` | Experiment summary metrics       |
| `unified_log.jsonl`  | Unified closed-loop decision log |

---

# 14. Reproduction Summary

```text
1. Set up Python/CUDA environment
        ↓
2. Configure Hugging Face access in .env
        ↓
3. Ensure the 9,000-record fine-tuned Exp3 model is available
        ↓
4. Start simulated_festo
        ↓
5. Set PROMPT_VARIANT=v4_fewshot
        ↓
6. Run llm_supervisor_exp07.py
        ↓
7. Execute 250 live closed-loop decisions
        ↓
8. Inspect summary_stats.json
        ↓
9. Inspect unified_log.jsonl and sensor_log.json
```

**Exp3 = Row 3 = SP + CoT + FS + P.**

```
```
