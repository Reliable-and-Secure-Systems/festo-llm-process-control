# Large Language Models as Autonomous Controllers for Multivariable Industrial Processes

This repository contains the implementation, training data generation scripts, inference supervisors, and evaluation results accompanying the IEEE IECON 2026 paper:

> **"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes"**  
> **V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner**

---

## Overview

This repository presents a **closed-loop fully open-sourced LLM-based control framework for simultaneous multi-variable industrial process control, demonstrated on a fluid process system**[^1].

The repository contains the complete implementation, training data generation scripts, fine-tuning pipelines, inference supervisors, and evaluation results accompanying our IEEE IECON 2026 paper. We fine-tune a Llama 3.2-3B-Instruct model using QLoRA to autonomously control a simulated Festo MPS PA dual-tank workstation. The model receives sensor readings every 5 seconds and outputs structured JSON control actions (`pump_power`, `upper_valve_open`, `heater_on`, `reason`). Three experiments progressively introduce Model Predictive Control (MPC) to study the effect of predictive context on control accuracy and hallucination rates.

[^1]: **Associated publication:** V. Rayar, M. Taheri, C. Herglotz, P. Thomas, S. Möller, and M. Hübner, *"Large Language Models as Autonomous Controllers for Multivariable Industrial Processes,"* in **Proceedings of the 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON 2026)**, Doha, Qatar, Oct. 18–21, 2026, in press. *(Paper link will be added once publicly available.)*

### Control Objectives

- **Level:** Maintain the upper tank level at **45%** (acceptable range: **40–50%**)
- **Temperature:** Maintain the water temperature at **35°C** (acceptable range: **32–38°C**)

---

## Experiments

| Experiment | Records | Fine-Tune Prompt | Inference Prompt | Level % | Temp % | Overall % |
|------------|---------|-----------------|------------------|--------:|-------:|----------:|
| [A (12k)](./Experiment_A/) | 11,975 | SP+CoT+FS | SP+CoT+FS | 65.4 | 85.9 | 60.0 |
| [B (9k)](./Experiment_B/) | 9,000 | SP+CoT+FS+P | SP+CoT+P | 100.0 | 60.8 | 60.8 |
| [B (9k)](./Experiment_B/) | 9,000 | SP+CoT+FS+P | SP+CoT+FS+P | 40.0 | 56.8 | 25.6 |
| [C (3k)](./Experiment_C/) | 3,000 | SP+CoT+FS+P | SP+CoT+P | 84.7 | 67.8 | 57.6 |
| [C (3k)](./Experiment_C/) | 3,000 | SP+CoT+FS+P | SP+CoT+FS+P | 86.2 | 61.2 | 52.6 |

**SP** = System Prompt  **CoT** = Chain of Thought  **FS** = Few-Shot  **P** = Predictive Context

---

## Experiment Descriptions

### [Experiment A](./Experiment_A/) — Baseline Fine-Tuning

- **11,975 training records**, no predictive horizon.
- Fine-tuned using **SP + CoT + FS**.
- Zero format and content hallucinations.
- Establishes the baseline with **60.0%** overall accuracy.

### [Experiment B](./Experiment_B/) — Receding Horizon MPC (9,000 Records)

- **9,000 training records** with a 3-step predictive horizon.
- Level-biased MPC cost function: `2.0 × |level_error| + 1.0 × |temp_error|`.
- SP+CoT+P achieves **100% level accuracy**.
- Demonstrates the importance of the safety supervisor due to increased hallucination rates.

### [Experiment C](./Experiment_C/) — Equal-Weight MPC (3,000 Records)

- **3,000 training records**.
- Equal-weight MPC cost function: `1.5 × |level_error| + 1.5 × |temp_error|`.
- Best temperature performance among all experiments.
- Isolates the impact of MPC cost-function design from dataset size.

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

## Model Weights (Hugging Face Hub)

Fine-tuned LoRA adapters are available on Hugging Face Hub:

| Experiment | Model |
|------------|-------|
| Experiment A | https://huggingface.co/vid1203/festo-llm-experiment-a |
| Experiment B | https://huggingface.co/vid1203/festo-llm-experiment-b |
| Experiment C | https://huggingface.co/vid1203/festo-llm-experiment-c |

---

## Repository Structure

```
Experiment_A/
Experiment_B/
Experiment_C/
```

Each experiment directory contains:

- Training dataset
- Fine-tuning scripts
- Inference supervisor
- Evaluation results
- Training logs
- Experiment-specific documentation

---

## Requirements

```bash
pip install transformers peft trl bitsandbytes accelerate huggingface_hub
```

---

## Citation

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

## Acknowledgements

- Llama 3.2
- Hugging Face Transformers
- PEFT (QLoRA)
- BitsAndBytes
- PyTorch
