````markdown
# Large Language Models as Autonomous Controllers for Multivariable Industrial Processes

This repository contains the implementation, training data generation scripts, inference supervisors, and evaluation results accompanying the IEEE IECON 2026 paper:

> **"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes"**  
> **V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner**

---

## Overview

This repository presents a **closed-loop fully open-sourced LLM-based control framework for simultaneous multi-variable industrial process control, demonstrated on a fluid process system**[^1].

The repository contains the complete implementation, training data generation scripts, fine-tuning pipelines, inference supervisors, and evaluation results accompanying our IEEE IECON 2026 paper. We fine-tune a Llama 3.2-3B-Instruct model using QLoRA to autonomously control a simulated Festo MPS PA dual-tank workstation. The model receives sensor readings every 5 seconds and outputs structured JSON control actions (`pump_power`, `upper_valve_open`, `heater_on`, `reason`). Five experiments evaluate different cumulative prompting configurations and the addition of predictive look-ahead to study its effect on closed-loop control accuracy and hallucination rates.

[^1]: **Associated publication:** V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner, *"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes,"* in **Proceedings of the 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON 2026)**, Doha, Qatar, Oct. 18–21, 2026, in press. *(Paper link will be added once publicly available.)*

### Control Objectives

- **Level:** Maintain the upper tank level at **45%** (acceptable range: **40–50%**)
- **Temperature:** Maintain the water temperature at **35°C** (acceptable range: **32–38°C**)

---

## Experiments

| Experiment | Training Records | Fine-Tuning Prompt | Inference Prompt | Level % | Temp % | Overall % |
|------------|-----------------:|--------------------|------------------|--------:|-------:|----------:|
| [Exp1](./Exp1/) | 11,975 | SP + CoT + FS | SP + CoT + FS | 70.4 | 86.4 | **65.2** |
| [Exp2](./Exp2/) | 9,000 | SP + CoT + FS + P | SP + CoT + P | 86.8 | 59.2 | **51.2** |
| [Exp3](./Exp3/) | 9,000 | SP + CoT + FS + P | SP + CoT + FS + P | 58.4 | 62.0 | **39.6** |
| [Exp4](./Exp4/) | 3,000 | SP + CoT + P | SP + CoT + P | 67.2 | 66.8 | **47.6** |
| [Exp5](./Exp5/) | 3,000 | SP + CoT + FS + P | SP + CoT + FS + P | 84.0 | 66.4 | **56.8** |

**SP** = System Prompt  **CoT** = Chain of Thought  **FS** = Few-Shot  **P** = 3-step Predictive Look-Ahead

All experiments use **250 live closed-loop evaluation decisions**.

---

## Experiment Descriptions

### [Exp1](./Exp1/) — Baseline Fine-Tuning

- **11,975 training records**.
- Prompting strategy: **SP + CoT + FS**.
- No predictive look-ahead is used.
- **70.4% level accuracy**.
- **86.4% temperature accuracy**.
- **65.2% overall accuracy**.
- Safety override rate: **0.0%**.
- Control hallucination rate: **0.0%**.

### [Exp2](./Exp2/) — Predictive Look-Ahead

- **9,000 training records**.
- Adds a **3-step predictive horizon** to the prompting strategy.
- Predictive look-ahead is evaluated across the cumulative prompting variants.
- **86.8% level accuracy**.
- **59.2% temperature accuracy**.
- **51.2% overall accuracy**.
- Safety override rate: **37.6%**.
- Control hallucination rate: **5.6%**.

### [Exp3](./Exp3/) — Full Prompting with Predictive Look-Ahead

- **9,000 training records**.
- Uses the full **SP + CoT + FS + P** strategy.
- Combines system prompting, Chain-of-Thought, few-shot examples, and a **3-step predictive horizon**.
- **58.4% level accuracy**.
- **62.0% temperature accuracy**.
- **39.6% overall accuracy**.
- Safety override rate: **31.6%**.
- Control hallucination rate: **28.0%**.

### [Exp4](./Exp4/) — Reduced Training Data with Predictive Look-Ahead

- **3,000 training records**.
- Uses **SP + CoT + P** without few-shot examples.
- Uses a **3-step predictive horizon**.
- **67.2% level accuracy**.
- **66.8% temperature accuracy**.
- **47.6% overall accuracy**.
- Safety override rate: **32.4%**.
- Control hallucination rate: **0.0%**.

### [Exp5](./Exp5/) — Full Prompting with Predictive Look-Ahead

- **3,000 training records**.
- Uses the full **SP + CoT + FS + P** strategy.
- Combines system prompting, Chain-of-Thought, few-shot examples, and a **3-step predictive horizon**.
- **84.0% level accuracy**.
- **66.4% temperature accuracy**.
- **56.8% overall multivariable accuracy**.
- Safety override rate: **not listed in the extracted Exp5 summary**.
- Control hallucination rate: **0.0%**.

---

## Fine-Tuned Models

The fine-tuned LoRA adapters used for the experiments are included directly in this repository.

| Experiment | Fine-Tuned Model | Adapter / Checkpoint |
|------------|------------------|----------------------|
| Exp1 | `festo_llama3.2_finetuned_exp05` | [`Exp1/festo_llama3.2_finetuned_exp05/checkpoint-1200/`](./Exp1/festo_llama3.2_finetuned_exp05/checkpoint-1200/) |
| Exp2 | `festo_llama3.2_finetuned_exp07` | [`Exp2/festo_llama3.2_finetuned_exp07/checkpoint-2755/`](./Exp2/festo_llama3.2_finetuned_exp07/checkpoint-2755/) |
| Exp3 | `festo_llama3.2_finetuned_exp07` | [`Exp3/festo_llama3.2_finetuned_exp07/checkpoint-2755/`](./Exp3/festo_llama3.2_finetuned_exp07/checkpoint-2755/) |
| Exp4 | `festo_llama3.2_finetuned_exp08` | [`Exp4/festo_llama3.2_finetuned_exp08/checkpoint-915/`](./Exp4/festo_llama3.2_finetuned_exp08/checkpoint-915/) |
| Exp5 | `festo_llama3.2_finetuned_exp08` | [`Exp5/festo_llama3.2_finetuned_exp08/checkpoint-915/`](./Exp5/festo_llama3.2_finetuned_exp08/checkpoint-915/) |

Each checkpoint contains the LoRA adapter weights and associated tokenizer/configuration files required to load the fine-tuned model.

---

## Model

| Property | Value |
|----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Fine-tuning | QLoRA |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| rsLoRA | Enabled |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Repository Structure

```text
Exp1/
├── README.md
├── finetune.py
├── finetune.sh
├── generate_training_data.py
├── llm_supervisor.py
├── run_llm_system.sh
├── festo_live/
└── festo_llama3.2_finetuned_exp05/
    
Exp2/
├── README.md
├── finetune.py
├── finetune.sh
├── generate_training_data.py
├── llm_supervisor.py
├── run_llm_system.sh
├── festo_live/
└── festo_llama3.2_finetuned_exp07/

Exp3/
├── README.md
├── finetune.py
├── finetune.sh
├── generate_training_data.py
├── llm_supervisor.py
├── run_llm_system.sh
├── festo_live/
└── festo_llama3.2_finetuned_exp07/

Exp4/
├── README.md
├── finetune.py
├── finetune.sh
├── generate_training_data.py
├── llm_supervisor.py
├── run_llm_system.sh
├── festo_live/
└── festo_llama3.2_finetuned_exp08/

Exp5/
├── README.md
├── finetune.py
├── finetune.sh
├── generate_training_data.py
├── llm_supervisor.py
├── run_llm_system.sh
├── festo_live/
└── festo_llama3.2_finetuned_exp08/

cpp_sim/
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
````

Each experiment directory contains:

* Experiment-specific documentation
* Fine-tuning scripts
* Training-data generation scripts
* LLM inference supervisor
* Fine-tuned LoRA adapter/checkpoint
* Closed-loop evaluation results
* Reproduction scripts

The `cpp_sim/` directory contains the C++ process simulation used for the closed-loop experiments.

---

## Requirements

```bash
pip install transformers peft trl bitsandbytes accelerate huggingface_hub
```

---

## Citation

If you use this repository in your research, experiments, publications, or derivative work, please cite ourpaper.

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

## Acknowledgements

* Llama 3.2
* Hugging Face Transformers
* PEFT (QLoRA)
* BitsAndBytes
* PyTorch

```
```
