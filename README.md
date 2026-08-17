````markdown
# Large Language Models as Autonomous Controllers for Multivariable Industrial Processes

Implementation and experimental artifacts accompanying the IEEE IECON 2026 paper:

> **"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes"**  
> **V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner**

---

## Overview

This repository provides the implementation of a **closed-loop LLM-based control framework for simultaneous multivariable industrial process control**, demonstrated on a Festo MPS PA dual-tank fluid process.

The framework uses a fine-tuned **Llama 3.2-3B-Instruct** model with **QLoRA** to autonomously control the process. At each control cycle, the LLM receives the current process state and generates a structured control action:

```text
pump_power
upper_valve_open
heater_on
reason
````

The repository contains:

* Fine-tuning and training-data generation
* LLM inference supervisors
* Closed-loop evaluation results
* Fine-tuned LoRA adapters
* Five experimental configurations
* The C++ process simulation

The experiments investigate the effect of **cumulative prompting strategies** and **3-step predictive look-ahead** on closed-loop control performance and hallucination behaviour.

[^1]: **Associated publication:** V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner, *"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes,"* in **Proceedings of the 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON 2026)**, Doha, Qatar, Oct. 18–21, 2026, in press. *(Paper link will be added once publicly available.)*

---

## Control Task

The controller regulates two process variables simultaneously:

| Variable          |   Target | Acceptable Range |
| ----------------- | -------: | ---------------: |
| Upper tank level  |  **45%** |           40–50% |
| Water temperature | **35°C** |          32–38°C |

The controller operates the process through:

* Pump power
* Upper-tank valve
* Heater

Safety constraints are enforced independently of the LLM controller.

---

# Experimental Results

All experiments use **250 live closed-loop evaluation decisions**.

| Experiment | Training Records | Prompt Configuration             | Level Accuracy | Temperature Accuracy | Overall Accuracy |
| :--------: | ---------------: | -------------------------------- | -------------: | -------------------: | ---------------: |
|  **Exp1**  |           11,975 | SP + CoT + FS                    |      **70.4%** |            **86.4%** |        **65.2%** |
|  **Exp2**  |            9,000 | SP + CoT + FS + P → SP + CoT + P |      **86.8%** |            **59.2%** |        **51.2%** |
|  **Exp3**  |            9,000 | SP + CoT + FS + P                |          58.4% |                62.0% |            39.6% |
|  **Exp4**  |            3,000 | SP + CoT + P                     |          67.2% |                66.8% |            47.6% |
|  **Exp5**  |            3,000 | SP + CoT + FS + P                |      **84.0%** |                66.4% |        **56.8%** |

### Prompt notation

| Abbreviation | Meaning                      |
| ------------ | ---------------------------- |
| **SP**       | System Prompt                |
| **CoT**      | Chain-of-Thought             |
| **FS**       | Few-Shot examples            |
| **P**        | 3-step Predictive Look-Ahead |

---

# Experiments

## [Exp1 — Baseline Fine-Tuning](./Exp1/)

**Prompt:** `SP + CoT + FS`
**Training data:** 11,975 records
**Predictive look-ahead:** No

Exp1 establishes the baseline closed-loop controller using system prompting, Chain-of-Thought, and few-shot examples.

| Metric                     |    Result |
| -------------------------- | --------: |
| Level accuracy             | **70.4%** |
| Temperature accuracy       | **86.4%** |
| Overall accuracy           | **65.2%** |
| Level MAE                  |    10.69% |
| Temperature MAE            |    1.75°C |
| Time to target             |  4 cycles |
| Longest stable run         | 20 cycles |
| Average latency            |  6,937 ms |
| Safety override rate       |  **0.0%** |
| Control hallucination rate |  **0.0%** |

**Model:** `festo_llama3.2_finetuned_exp05`
**Checkpoint:** [`Exp1/festo_llama3.2_finetuned_exp05/checkpoint-1200/`](./Exp1/festo_llama3.2_finetuned_exp05/checkpoint-1200/)

---

## [Exp2 — Predictive Look-Ahead](./Exp2/)

**Training data:** 9,000 records
**Predictive horizon:** 3 steps

Exp2 investigates the effect of adding predictive process states to the controller prompt. The predictive horizon is evaluated across the cumulative prompting configurations.

| Metric                     |    Result |
| -------------------------- | --------: |
| Level accuracy             | **86.8%** |
| Temperature accuracy       |     59.2% |
| Overall accuracy           | **51.2%** |
| Level MAE                  |     4.73% |
| Temperature MAE            |    2.75°C |
| Time to target             |  7 cycles |
| Longest stable run         |  5 cycles |
| Average latency            | 10,465 ms |
| Safety override rate       |     37.6% |
| Hallucination rate         |    100.0% |
| Control hallucination rate |      5.6% |
| Text hallucination rate    |     96.8% |

**Model:** `festo_llama3.2_finetuned_exp07`
**Checkpoint used:** [`Exp2/festo_llama3.2_finetuned_exp07/checkpoint-2755/`](./Exp2/festo_llama3.2_finetuned_exp07/checkpoint-2755/)

---

## [Exp3 — Full Prompting + Predictive Look-Ahead](./Exp3/)

**Prompt:** `SP + CoT + FS + P`
**Training data:** 9,000 records
**Predictive horizon:** 3 steps

Exp3 evaluates the full cumulative prompting strategy together with predictive look-ahead.

| Metric                     |    Result |
| -------------------------- | --------: |
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

**Model:** `festo_llama3.2_finetuned_exp07`
**Checkpoint used:** [`Exp3/festo_llama3.2_finetuned_exp07/checkpoint-2755/`](./Exp3/festo_llama3.2_finetuned_exp07/checkpoint-2755/)

---

## [Exp4 — Reduced Training Data + Predictive Look-Ahead](./Exp4/)

**Prompt:** `SP + CoT + P`
**Training data:** 3,000 records
**Predictive horizon:** 3 steps
**Few-shot examples:** Not used

Exp4 evaluates predictive look-ahead with a reduced training dataset and without few-shot examples.

| Metric                     |    Result |
| -------------------------- | --------: |
| Level accuracy             |     67.2% |
| Temperature accuracy       |     66.8% |
| Overall accuracy           |     47.6% |
| Level MAE                  |     4.59% |
| Temperature MAE            |    2.64°C |
| Time to target             |  5 cycles |
| Longest stable run         |  6 cycles |
| Average latency            | 25,518 ms |
| Safety override rate       |     32.4% |
| Hallucination rate         |    100.0% |
| Control hallucination rate |  **0.0%** |
| Text hallucination rate    |    100.0% |

**Model:** `festo_llama3.2_finetuned_exp08`
**Checkpoint used:** [`Exp4/festo_llama3.2_finetuned_exp08/checkpoint-915/`](./Exp4/festo_llama3.2_finetuned_exp08/checkpoint-915/)

---

## [Exp5 — Full Prompting + Predictive Look-Ahead](./Exp5/)

**Prompt:** `SP + CoT + FS + P`
**Training data:** 3,000 records
**Predictive horizon:** 3 steps

Exp5 evaluates the full cumulative prompting strategy with predictive look-ahead using the reduced 3,000-record training dataset.

| Metric                         |    Result |
| ------------------------------ | --------: |
| Level accuracy                 | **84.0%** |
| Temperature accuracy           |     66.4% |
| Overall multivariable accuracy | **56.8%** |
| Level MAE                      |     4.07% |
| Temperature MAE                |    2.81°C |
| Control hallucination rate     |  **0.0%** |

The complete experiment-specific evaluation metrics and hallucination breakdown are documented in the [Exp5 README](./Exp5/README.md).

**Model:** `festo_llama3.2_finetuned_exp08`
**Checkpoint used:** [`Exp5/festo_llama3.2_finetuned_exp08/checkpoint-915/`](./Exp5/festo_llama3.2_finetuned_exp08/checkpoint-915/)

---

# Fine-Tuned Models

The fine-tuned LoRA adapters used in the experiments are included directly in this repository.

| Experiment | Model Directory                  | Evaluation Checkpoint                                                       |
| ---------- | -------------------------------- | --------------------------------------------------------------------------- |
| **Exp1**   | `festo_llama3.2_finetuned_exp05` | [`checkpoint-1200`](./Exp1/festo_llama3.2_finetuned_exp05/checkpoint-1200/) |
| **Exp2**   | `festo_llama3.2_finetuned_exp07` | [`checkpoint-2755`](./Exp2/festo_llama3.2_finetuned_exp07/checkpoint-2755/) |
| **Exp3**   | `festo_llama3.2_finetuned_exp07` | [`checkpoint-2755`](./Exp3/festo_llama3.2_finetuned_exp07/checkpoint-2755/) |
| **Exp4**   | `festo_llama3.2_finetuned_exp08` | [`checkpoint-915`](./Exp4/festo_llama3.2_finetuned_exp08/checkpoint-915/)   |
| **Exp5**   | `festo_llama3.2_finetuned_exp08` | [`checkpoint-915`](./Exp5/festo_llama3.2_finetuned_exp08/checkpoint-915/)   |

Each checkpoint contains the LoRA adapter weights and associated tokenizer/configuration files required to load the fine-tuned model.

---

# Model Configuration

| Property            | Configuration                                                               |
| ------------------- | --------------------------------------------------------------------------- |
| Base model          | `meta-llama/Llama-3.2-3B-Instruct`                                          |
| Fine-tuning         | QLoRA                                                                       |
| Quantization        | 4-bit NF4                                                                   |
| Double quantization | Enabled                                                                     |
| Compute dtype       | FP16                                                                        |
| LoRA rank           | 16                                                                          |
| LoRA alpha          | 32                                                                          |
| LoRA dropout        | 0.05                                                                        |
| rsLoRA              | Enabled                                                                     |
| Optimizer           | AdamW 8-bit                                                                 |
| Target modules      | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

---

# Repository Structure

```text
Paper_Experiments/
│
├── Exp1/
│   ├── README.md
│   ├── finetune.py
│   ├── finetune.sh
│   ├── generate_training_data.py
│   ├── llm_supervisor.py
│   ├── run_llm_system.sh
│   ├── festo_live/
│   └── festo_llama3.2_finetuned_exp05/
│       └── checkpoint-1200/
│
├── Exp2/
│   ├── README.md
│   ├── finetune.py
│   ├── finetune.sh
│   ├── generate_training_data.py
│   ├── llm_supervisor.py
│   ├── run_llm_system.sh
│   ├── festo_live/
│   └── festo_llama3.2_finetuned_exp07/
│       └── checkpoint-2755/
│
├── Exp3/
│   ├── README.md
│   ├── finetune.py
│   ├── finetune.sh
│   ├── generate_training_data.py
│   ├── llm_supervisor.py
│   ├── run_llm_system.sh
│   ├── festo_live/
│   └── festo_llama3.2_finetuned_exp07/
│       └── checkpoint-2755/
│
├── Exp4/
│   ├── README.md
│   ├── finetune.py
│   ├── finetune.sh
│   ├── generate_training_data.py
│   ├── llm_supervisor.py
│   ├── run_llm_system.sh
│   ├── festo_live/
│   └── festo_llama3.2_finetuned_exp08/
│       └── checkpoint-915/
│
├── Exp5/
│   ├── README.md
│   ├── finetune.py
│   ├── finetune.sh
│   ├── generate_training_data.py
│   ├── llm_supervisor.py
│   ├── run_llm_system.sh
│   ├── festo_live/
│   └── festo_llama3.2_finetuned_exp08/
│       └── checkpoint-915/
│
└── cpp_sim/
    ├── main.cpp
    ├── SimulatedFesto.cpp
    ├── SimulatedFesto.hpp
    ├── Data.cpp
    ├── Data.hpp
    ├── DataFilter.cpp
    ├── DataFilter.hpp
    ├── ErrorDetector.cpp
    ├── ErrorDetector.hpp
    ├── ConvertAndPrepareData.cpp
    ├── ConvertAndPrepareData.hpp
    └── commands.txt
```

---

# Requirements

```bash
pip install transformers peft trl bitsandbytes accelerate huggingface_hub
```

The fine-tuning scripts additionally require access to the Llama 3.2 model through Hugging Face.

---

# Reproducing the Experiments

Each experiment directory contains its own documentation and execution scripts.

Start with the experiment-specific README:

* [Exp1 documentation](./Exp1/README.md)
* [Exp2 documentation](./Exp2/README.md)
* [Exp3 documentation](./Exp3/README.md)
* [Exp4 documentation](./Exp4/README.md)
* [Exp5 documentation](./Exp5/README.md)

The `cpp_sim/` directory contains the process simulation used by the closed-loop experiments.

---

# Citation

If you use this repository in your research, experiments, publications, or derivative work, please cite our paper.

**Large Language Models as Autonomous Controllers for Multivariable Industrial Processes**
V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner
*Proceedings of the 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON 2026), Doha, Qatar, Oct. 18–21, 2026, In Press.*

**BibTeX:**

```bibtex
@inproceedings{rayar2026llmcontrollers,
  title     = {Large Language Models as Autonomous Controllers for Multivariable Industrial Processes},
  author    = {Rayar, Vidyashree and Taheri, Mahdi and Herglotz, Christian and Thomas, Peter and M{\"o}ller, Stefan and H{\"u}bner, Michael},
  booktitle = {Proceedings of the 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON)},
  address   = {Doha, Qatar},
  month     = oct,
  year      = {2026},
  note      = {In Press}
}
```

---

# Acknowledgements

* Llama 3.2
* Hugging Face Transformers
* PEFT (QLoRA)
* BitsAndBytes
* PyTorch

```
```

[1]: https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories?utm_source=chatgpt.com "Best practices for repositories - GitHub Docs"
