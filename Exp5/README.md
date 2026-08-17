Exp5 — Predictive Look-Ahead with Few-Shot Prompting (SP + CoT + FS + P)

Overview

Exp5 corresponds to Row 5 in Table I and evaluates the full cumulative prompting strategy with predictive look-ahead.

The experiment uses a fine-tuned Llama 3.2 3B controller in a live closed-loop simulation of the Festo MPS PA dual-tank process.

Experiment strategy: SP + CoT + FS + P

SP: System Prompt

CoT: Chain-of-Thought

FS: Few-Shot examples

P: 3-step predictive look-ahead

Exp5 uses 3,000 training records and 250 live closed-loop evaluation decisions.

1. Controlled System

The controlled system is a simulated Festo MPS PA dual-tank hydraulic process with two control objectives:

Variable

Target

Acceptable Range

Upper tank level

45%

40–50%

Water temperature

35°C

32–38°C

The controller receives live sensor readings every 5 seconds and maintains a 6-step history window.

The LLM generates control actions for:

Pump power

Upper valve state

Heater state

Short control reason

A hardware-safety layer is applied after LLM generation.

Hardware Safety Limits

The following constraints cannot be overridden by the LLM:

Temperature > 38°C → heater OFF

Level > 70% → pump OFF and valve OPEN

Pump + valve simultaneously → pump OFF

2. Dataset

Exp5 uses:

3,000 fine-tuning records

The live evaluation set is separate from the training data.

Data

Count

Training records

3,000

Live evaluation decisions

250

The experiment evaluates the fine-tuned model through live closed-loop interaction with the simulator rather than through an offline test dataset.

3. Prompting Strategy

Exp5 corresponds to Row 5:

SP + CoT + FS + P

The corresponding prompt implementation is:

v4_fewshot

SP — System Prompt

The system prompt defines the controller role, required JSON schema, actuator constraints, and control targets.

CoT — Chain-of-Thought

The model is instructed to reason through:

Level error

Temperature error

Predicted future states for both variables

Final control action

FS — Few-Shot Examples

Three worked examples are included to demonstrate the expected control behaviour and JSON output format. The examples are also updated to include predictive-horizon information.

P — Predictive Look-Ahead

A 3-step predictive horizon is supplied with the current process state. The model uses the predicted states to assess whether the control action is moving both variables toward their targets.

Therefore:

Exp5 = SP + CoT + FS + P
     = v4_fewshot

The attached supervisor implementation defines v4_fewshot as the cumulative System Prompt + Chain-of-Thought + Few-Shot configuration, with the 3-step predictive horizon included in the prompt. It also confirms the 3-step horizon and 6-step history configuration.

4. Predictive Look-Ahead

For each decision, the controller generates a nominal 3-step future trajectory:

t+1: level=...%, temp=...C
t+2: level=...%, temp=...C
t+3: level=...%, temp=...C

The prediction uses the process simulation parameters:

Parameter

Value

Prediction horizon

3 steps

Fill rate

3.27

Drain rate

1.58

Heat rate

0.074

Cool rate

0.0173

Thermal inertia

3

Inertia factor

0.3

Float cutoff

47%

Temperature hysteresis low

34°C

Temperature hysteresis high

36°C

5. Model

Base model:

meta-llama/Llama-3.2-3B-Instruct

The fine-tuned model is loaded as a PEFT/LoRA adapter.

Inference uses:

4-bit NF4 quantization

Double quantization

FP16 computation

CUDA acceleration

Greedy decoding

The attached supervisor selects the latest available fine-tuning checkpoint.

Reproduction Procedure

6. Environment Setup

Use the configured CUDA/Python environment containing the required:

PyTorch

Transformers

PEFT

BitsAndBytes

Hugging Face authentication is required for the base model.

Create a local .env file containing:

HUGGING_FACE_KEY=<your-token>

Do not commit .env or the Hugging Face token to GitHub.

7. Prepare the Fine-Tuned Model

Ensure the Exp5 fine-tuned adapter, including its checkpoints, is available in the experiment directory.

The latest checkpoint-* directory is selected automatically by the supervisor.

8. Start the Festo Simulator

Start the simulated Festo process using the project simulator:

./simulated_festo

Keep the simulator running while the LLM supervisor is executing.

9. Run Exp5

Select the Exp5 prompting configuration:

PROMPT_VARIANT=v4_fewshot

Run the Exp5 supervisor with the experiment's configured Python environment.

The selected variant must be:

v4_fewshot

which corresponds to:

SP + CoT + FS + P

The run terminates automatically after 250 decisions.

10. Closed-Loop Execution

For each decision cycle, the supervisor:

Reads the current live sensor state.

Maintains the latest 6-step process history.

Computes the 3-step predictive horizon.

Builds the SP + CoT + FS + P prompt.

Generates the LLM control action.

Parses the generated JSON.

Applies the hardware-safety constraints.

Writes the actuator command for the simulator.

Records control and hallucination metrics.

Waits 5 seconds before the next decision.

The attached implementation confirms the 5-second interval, 250-decision stopping condition, 6-step history, 3-step prediction horizon, and v4_fewshot prompt variant.

11. Inference Configuration

Parameter

Value

Prompt strategy

SP + CoT + FS + P

Prompt variant

v4_fewshot

Few-shot examples

Yes — 3

History window

6 steps

Prediction horizon

3 steps

Decision interval

5 seconds

Maximum decisions

250

Max new tokens

250

Sampling

Disabled

Temperature

1.0

Top-p

1.0

The model uses greedy decoding:

do_sample = False

12. Evaluation

Evaluation is performed through live closed-loop simulation.

Level Hit Condition

40% ≤ level ≤ 50%

Temperature Hit Condition

32°C ≤ temperature ≤ 38°C

Overall Accuracy

A decision is counted as an overall hit only when both level and temperature are within their respective acceptable bands.

The experiment records:

Level control accuracy

Temperature control accuracy

Overall accuracy

Level MAE

Temperature MAE

Time to target

Longest stable run

Average inference latency

Safety override rate

Overall hallucination rate

Control hallucination rate

Text hallucination rate

Hallucination types

13. Results

The reported Exp5 run contains 250 live closed-loop decisions.

Metric

Result

Training records

3,000

Decisions

250

Level accuracy

77.2%

Temperature accuracy

61.6%

Overall accuracy

48.0%

Level MAE

4.305%

Temperature MAE

2.633°C

Time to target

2 cycles

Longest stable run

4 cycles

Average latency

25,210 ms

Safety override rate

38.4%

Hallucination rate

100.0%

Control hallucination rate

0.0%

Text hallucination rate

100.0%

Hallucination Breakdown

Type

Count

Parse error

0

Missing field

0

Wrong type

0

Out of range

0

Wrong key

0

Extra content

250

The 100% overall hallucination rate is therefore entirely attributable to extra content after the valid JSON response. The reported control hallucination rate is 0%, meaning none of the 250 decisions were classified as control hallucinations.

14. Output Files

The Exp5 festo_live directory contains:

festo_live/
├── data.bin
├── json_data.txt
├── llm_control.json
├── sensor_log.json
├── summary_stats.json
└── unified_log.jsonl

File

Purpose

data.bin

Simulator data file

json_data.txt

Current live sensor state

llm_control.json

Latest actuator command sent to the simulator

sensor_log.json

Recorded sensor data

summary_stats.json

Experiment summary metrics

unified_log.jsonl

Unified closed-loop decision log

15. Reproduction Summary

1. Set up the CUDA/Python environment
        ↓
2. Configure HUGGING_FACE_KEY in local .env
        ↓
3. Ensure the 3,000-record Exp5 fine-tuned adapter is available
        ↓
4. Start simulated_festo
        ↓
5. Set PROMPT_VARIANT=v4_fewshot
        ↓
6. Run the Exp5 LLM supervisor
        ↓
7. Execute 250 live closed-loop decisions
        ↓
8. Inspect summary_stats.json
        ↓
9. Inspect unified_log.jsonl and sensor_log.json

Environment-Specific Paths

The current supervisor contains hardcoded paths referring to exp08 and Experiment_3_equal_cycles. These reflect the execution environment used for the recorded experiment and are not the canonical Exp5 folder names.

When reproducing Exp5 on another system, adapt these environment-specific paths to the local Exp5 directory structure as required. The experimental configuration and reported results remain unchanged.

Experiment Definition

Exp5 = Row 5 = SP + CoT + FS + P

This experiment therefore evaluates the full cumulative prompting strategy together with the 3-step predictive look-ahead in live closed-loop simulation.