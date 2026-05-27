# LLMs as Autonomous Controllers for Multivariable Industrial Fluid Processes

This repository contains the code, training data generation scripts, inference supervisors, and evaluation results for the experiments presented in the paper:

> **"LLMs as Autonomous Controllers for Multivariable Industrial Fluid Processes"**
> Vidyashree Rayar — BTU Cottbus-Senftenberg

---

## Overview

We fine-tune a Llama 3.2-3B-Instruct model using QLoRA to act as a closed-loop controller for a simulated Festo MPS PA dual-tank workstation. The model receives sensor readings every 5 seconds and outputs a structured JSON control action (`pump_power`, `upper_valve_open`, `heater_on`, `reason`). Three experiments progressively introduce Model Predictive Control (MPC) and evaluate the effect on control accuracy and hallucination rates.

### Control Objectives
- **Level**: Keep upper tank level at 45% (acceptable band: 40–50%)
- **Temperature**: Keep water temperature at 35°C (acceptable band: 32–38°C)

---

## Experiments

| Experiment | Records | Fine-Tune Prompt | Inference Prompt | Level% | Temp% | Overall% |
|------------|---------|-----------------|-----------------|--------|-------|----------|
| [A (12k)](./Experiment_A/) | 11,975 | SP+CoT+FS | SP+CoT+FS | 65.4 | 85.9 | 60.0 |
| [B (9k)](./Experiment_B/) | 9,000 | SP+CoT+FS+P | SP+CoT+P | 100.0 | 60.8 | 60.8 |
| [B (9k)](./Experiment_B/) | 9,000 | SP+CoT+FS+P | SP+CoT+FS+P | 40.0 | 56.8 | 25.6 |
| [C (3k)](./Experiment_C/) | 3,000 | SP+CoT+FS+P | SP+CoT+P | 84.7 | 67.8 | 57.6 |
| [C (3k)](./Experiment_C/) | 3,000 | SP+CoT+FS+P | SP+CoT+FS+P | 86.2 | 61.2 | 52.6 |

**SP**=System Prompt, **CoT**=Chain of Thought, **FS**=Few Shot, **P**=Predictive Context

---

## Experiment Descriptions

### [Experiment A](./Experiment_A/) — Baseline Fine-Tuning
- **11,975 training records**, no predictive horizon
- Fine-tuned with full cumulative prompt: SP+CoT+FS
- Zero format and content hallucination — strongest safety performance across all experiments
- Establishes the baseline: 60.0% overall accuracy, 0% safety overrides

### [Experiment B](./Experiment_B/) — Receding Horizon MPC (9,000 Records)
- **9,000 training records** with 3-step predictive horizon context
- Level-biased MPC cost function: `2.0 × |level_error| + 1.0 × |temp_error|`
- SP+CoT+P achieves 100% level accuracy but temperature drops to 60.8%
- Content hallucination rises to 39.2% — safety layer becomes essential

### [Experiment C](./Experiment_C/) — Equal-Weight MPC (3,000 Records)
- **3,000 training records** with equal-weight MPC cost function
- Equal-weight cost function: `1.5 × |level_error| + 1.5 × |temp_error|`
- Best temperature accuracy across all experiments: 67.8%
- Isolates the effect of cost function design from data quantity

---

## Model

| Property | Value |
|----------|-------|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| rsLoRA | Enabled |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Model Weights (Hugging Face Hub)

Fine-tuned LoRA adapters are available on Hugging Face Hub:

| Experiment | Model |
|------------|-------|
| Experiment A | [vid1203/festo-llm-experiment-a](https://huggingface.co/vid1203/festo-llm-experiment-a) |
| Experiment B | [vid1203/festo-llm-experiment-b](https://huggingface.co/vid1203/festo-llm-experiment-b) |
| Experiment C | [vid1203/festo-llm-experiment-c](https://huggingface.co/vid1203/festo-llm-experiment-c) |

---

## Repository Structure

```
├── Experiment_A/
│   ├── README.md
│   ├── dataset/                  # Training dataset (11,975 records)
│   ├── results/                  # Evaluation results
│   ├── logs/                     # SLURM training logs
│   ├── finetune_experiment_a.py/.sh
│   ├── generate_training_data_experiment_a.py
│   └── llm_supervisor_experiment_a.py
│
├── Experiment_B/
│   ├── README.md
│   ├── dataset/                  # Training dataset (9,000 records)
│   ├── results/
│   │   ├── sp_cot_p/             # SP+CoT+P inference results
│   │   └── sp_cot_fs_p/          # SP+CoT+FS+P inference results
│   ├── logs/
│   ├── finetune_experiment_b.py/.sh
│   ├── generate_training_data_experiment_b.py
│   └── llm_supervisor_experiment_b.py
│
└── Experiment_C/
    ├── README.md
    ├── dataset/                  # Training dataset (3,000 records)
    ├── results/
    │   ├── sp_cot_p/             # SP+CoT+P inference results
    │   └── sp_cot_fs_p/          # SP+CoT+FS+P inference results
    ├── logs/
    ├── finetune_experiment_c.py/.sh
    ├── generate_training_data_experiment_c.py
    └── llm_supervisor_experiment_c.py
```

---

## Requirements

```bash
pip install transformers peft trl bitsandbytes accelerate huggingface_hub
```

---

## Citation

If you use this code or results, please cite:

```
@inproceedings{rayar2026llm,
  title        = {LLMs as Autonomous Controllers for Multivariable Industrial Fluid Processes},
  author       = {Rayar, Vidyashree and H{\"u}bner, Michael and Taheri, Mahdi},
  booktitle    = {Proceedings of [Conference Name]},
  year         = {2026},
  organization = {Brandenburg University of Technology Cottbus-Senftenberg},
  address      = {Cottbus, Germany}
}
```
