# Prompting Strategy Examples

Concrete examples of how each prompting strategy contributes to the prompt sent to the language model. Strategies are combined incrementally (Zero -> SP -> CoT -> FS) and paired with the predictive context (P) as described in Table I of the paper.

## Zero Prompt

No system instruction, no reasoning guidance, no examples. Only the task input.

```
Upper level: 30.0%, Temperature: 32.0C, Target level: 45%, Target temp: 35C
Return ONLY valid JSON: {"pump_power": 0, "heater_on": false, "valve_open": false, "reason": "short reason"}
```

## System Prompt (SP)

A dedicated message defining the model's role and the expected output format, sent before the task input.

```
You are a Festo water tank controller. PRIMARY: keep level at 45% (range 40-50%). SECONDARY: keep temperature at 35C (range 32-38C). Output ONLY a single JSON object with these exact keys: pump_power (integer 0-100), valve_open (boolean true/false), heater_on (boolean true/false), reason (short string). NEVER use strings for boolean fields. NEVER add extra keys.
```

## Chain of Thought (CoT)

Instructs the model to reason step by step before producing its final output. Appended to the system message.

```
Think step by step about level error first, then temperature, then output ONLY a single JSON object.
```

## Few-shot Prompting (FS)

A small set of worked input/output examples appended after the system message, giving the model a concrete reference for expected behavior.

```
Examples:
Input: level=10%, temp=30C -> {"pump_power": 100, "valve_open": false, "heater_on": true, "reason": "level critical low, pump max; temp low, heater on"}
Input: level=48%, temp=36C -> {"pump_power": 0, "valve_open": true, "heater_on": false, "reason": "level near float switch, drain; temp within band"}
Input: level=42%, temp=33C -> {"pump_power": 30, "valve_open": false, "heater_on": true, "reason": "level slightly low, gentle fill; temp low, heater on"}
```

## Combined SP+CoT+FS example

The full system message used in experiments combining all three strategies:

```
You are a Festo water tank controller. PRIMARY: keep level at 45% (range 40-50%). SECONDARY: keep temperature at 35C (range 32-38C). Output ONLY a single JSON object with these exact keys: pump_power (integer 0-100), valve_open (boolean true/false), heater_on (boolean true/false), reason (short string). NEVER use strings for boolean fields. NEVER add extra keys.

Think step by step about level error first, then temperature, then output ONLY a single JSON object.

Examples:
Input: level=10%, temp=30C -> {"pump_power": 100, "valve_open": false, "heater_on": true, "reason": "level critical low, pump max; temp low, heater on"}
Input: level=48%, temp=36C -> {"pump_power": 0, "valve_open": true, "heater_on": false, "reason": "level near float switch, drain; temp within band"}
Input: level=42%, temp=33C -> {"pump_power": 30, "valve_open": false, "heater_on": true, "reason": "level slightly low, gentle fill; temp low, heater on"}
```
