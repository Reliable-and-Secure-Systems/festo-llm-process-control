#!/usr/bin/env python3
"""
festo_llama3.2_exp05 — dataset generator
Combines exp04 winning pattern (4-field schema, anti-hallucination correction pairs,
30%% maintenance zone, history in prompt) with course techniques
(few-shot examples + CoT in system prompt).

Targets: level 45% +- 5%, temp 35C +- 3°C (level priority).
Output: 12,500 examples (12,000 normal + 500 anti-hallucination).
"""

import json
import random
from collections import Counter
from pathlib import Path

random.seed(3407)

TARGET_LEVEL = 45
TARGET_TEMP = 35
FLOAT_SWITCH_CUTOFF = 47
HISTORY_STEPS = 6

FILL_RATE = 3.27
DRAIN_RATE = 1.58
HEAT_RATE = 0.074
COOL_RATE = 0.0173
THERMAL_INERTIA_STEPS = 3
INERTIA_FACTOR = 0.3

TEMP_HYSTERESIS_LOW = 34.0
TEMP_HYSTERESIS_HIGH = 36.0

OUT_FILE = Path(__file__).parent / "dataset.jsonl"


def compute_pump(level):
    error = TARGET_LEVEL - level
    if level >= FLOAT_SWITCH_CUTOFF:
        # float switch stops pump; drain valve still opens if above target+5
        return 0, error < -5
    if error > 25:
        return 100, False
    if error > 15:
        return 80, False
    if error > 10:
        return 60, False
    if error > 5:
        return 40, False
    if error > 2:
        return 20, False
    if error > 0:
        return 10, False
    if error >= -5:
        return 0, False
    return 0, True


def compute_heater(temp, prev_heater):
    if temp < TEMP_HYSTERESIS_LOW:
        return True
    if temp > TEMP_HYSTERESIS_HIGH:
        return False
    return prev_heater


def short_reason(level, temp, pump, heater, valve):
    le = TARGET_LEVEL - level
    if level >= FLOAT_SWITCH_CUTOFF:
        l_part = "float switch active, pump off"
    elif le > 25:
        l_part = "level critical low, pump max"
    elif le > 10:
        l_part = "level low, filling"
    elif le > 0:
        l_part = "level slightly low"
    elif le >= -5:
        l_part = "level on target"
    else:
        l_part = "level high, draining"

    if temp < TEMP_HYSTERESIS_LOW:
        t_part = "temp low, heater on"
    elif temp > TEMP_HYSTERESIS_HIGH:
        t_part = "temp high, heater off"
    else:
        t_part = f"temp in band, heater {'on' if heater else 'off'}"
    return f"{l_part}; {t_part}"


def simulate_step(level, temp, pump, valve, heater, inertia):
    dl = 0.0
    if pump > 0 and level < FLOAT_SWITCH_CUTOFF:
        dl += FILL_RATE * pump / 100.0
    if valve:
        dl -= DRAIN_RATE
    dl += random.uniform(-0.1, 0.1)
    level = round(max(0, min(100, level + dl)), 1)
    if level >= FLOAT_SWITCH_CUTOFF:
        level = FLOAT_SWITCH_CUTOFF

    if heater:
        dt = HEAT_RATE
        inertia = THERMAL_INERTIA_STEPS
    elif inertia > 0:
        dt = HEAT_RATE * INERTIA_FACTOR
        inertia -= 1
    else:
        dt = 0.0
    dt -= COOL_RATE
    dt += random.uniform(-0.005, 0.005)
    temp = round(max(15, min(95, temp + dt)), 2)
    return level, temp, inertia


def make_history(target_level, target_temp, prev_heater):
    level = round(max(0, min(100, target_level + random.uniform(-10, 10))), 1)
    temp = round(max(15, min(95, target_temp + random.uniform(-5, 5))), 2)
    inertia = random.randint(0, THERMAL_INERTIA_STEPS)
    history = []
    for _ in range(HISTORY_STEPS):
        pump, valve = compute_pump(level)
        heater = compute_heater(temp, prev_heater)
        history.append({
            "level": level, "temp": temp,
            "pump": pump, "valve": valve, "heater": heater,
        })
        level, temp, inertia = simulate_step(level, temp, pump, valve, heater, inertia)
        prev_heater = heater
    return history


def fmt_history(hist):
    parts = []
    for i, h in enumerate(reversed(hist)):
        v = "true" if h["valve"] else "false"
        ht = "true" if h["heater"] else "false"
        parts.append(
            f"t-{i+1}: level={h['level']:.1f}%, temp={h['temp']:.2f}C, "
            f"pump={h['pump']}, valve={v}, heater={ht}"
        )
    return " | ".join(parts)


SYSTEM_PROMPT = (
    "You are a Festo water tank controller. PRIMARY: keep level at 45% (range 40-50%). "
    "SECONDARY: keep temperature at 35C (range 32-38C). "
    "Think step by step about level error first, then temperature, then output ONLY a "
    "single JSON object with these exact keys: pump_power (integer 0-100), "
    "upper_valve_open (boolean true/false), heater_on (boolean true/false), "
    "reason (short string). NEVER use strings for boolean fields. NEVER add extra keys. "
    "Examples:\n"
    "Input: level=10%, temp=30C -> {\"pump_power\": 100, \"upper_valve_open\": false, "
    "\"heater_on\": true, \"reason\": \"level critical low, pump max; temp low, heater on\"}\n"
    "Input: level=44%, temp=35C -> {\"pump_power\": 10, \"upper_valve_open\": false, "
    "\"heater_on\": true, \"reason\": \"level slightly low; temp in band, heater on\"}\n"
    "Input: level=60%, temp=40C -> {\"pump_power\": 0, \"upper_valve_open\": true, "
    "\"heater_on\": false, \"reason\": \"level high, draining; temp high, heater off\"}"
)


def build_user_msg(level, temp, history):
    return (
        f"Upper level: {level}%, Temperature: {temp}C, "
        f"Target level: {TARGET_LEVEL}%, Target temp: {TARGET_TEMP}C\n"
        f"History: {fmt_history(history)}"
    )


# ---------- distribution ----------
raw = []

# Maintenance zone (30%) — model must learn stability
for _ in range(3600):
    level = round(random.uniform(40, 50), 1)
    temp = round(random.uniform(32, 38), 2)
    raw.append((level, temp, "maintenance"))

# Level-focused (25%)
for _ in range(600):
    raw.append((round(random.uniform(0, 20), 1), round(random.uniform(20, 50), 2), "level_critical_low"))
for _ in range(600):
    raw.append((round(random.uniform(20, 35), 1), round(random.uniform(20, 50), 2), "level_below"))
for _ in range(900):
    raw.append((round(random.uniform(35, 45), 1), round(random.uniform(25, 45), 2), "level_approach"))
for _ in range(450):
    raw.append((round(random.uniform(45, 55), 1), round(random.uniform(20, 50), 2), "level_above"))
for _ in range(450):
    raw.append((round(random.uniform(55, 100), 1), round(random.uniform(20, 60), 2), "level_high_drain"))

# Temperature trajectories (25%)
for _ in range(750):
    raw.append((round(random.uniform(0, 100), 1), round(random.uniform(15, 28), 2), "temp_cold"))
for _ in range(600):
    raw.append((round(random.uniform(0, 100), 1), round(random.uniform(28, TEMP_HYSTERESIS_LOW), 2), "temp_warming"))
for _ in range(600):
    raw.append((round(random.uniform(0, 100), 1), round(random.uniform(TEMP_HYSTERESIS_HIGH, 50), 2), "temp_overshoot"))
for _ in range(525):
    if random.random() > 0.5:
        t = round(random.uniform(50, 80), 2)
    else:
        t = round(random.uniform(15, 22), 2)
    raw.append((round(random.uniform(0, 100), 1), t, "temp_extreme"))

# Cross-over and edge cases (20%)
for _ in range(600):
    raw.append((round(random.uniform(0, 25), 1), round(random.uniform(15, 28), 2), "critical_both"))
for _ in range(600):
    level = round(random.uniform(40, 50), 1)
    t = round(random.uniform(15, 28), 2) if random.random() > 0.5 else round(random.uniform(45, 60), 2)
    raw.append((level, t, "level_ok_temp_bad"))
for _ in range(600):
    level = round(random.uniform(0, 25), 1) if random.random() > 0.5 else round(random.uniform(60, 100), 1)
    raw.append((level, round(random.uniform(33, 37), 2), "temp_ok_level_bad"))
for _ in range(600):
    raw.append((round(random.uniform(46, 50), 1), round(random.uniform(20, 50), 2), "float_switch_edge"))

random.shuffle(raw)

print(f"Total raw: {len(raw)}")
for cat, count in sorted(Counter(c for _, _, c in raw).items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count} ({count/len(raw)*100:.1f}%)")

# ---------- main records ----------
records = []
for level, temp, _ in raw:
    prev_heater = random.choice([True, False])
    hist = make_history(level, temp, prev_heater)
    pump, valve = compute_pump(level)
    heater = compute_heater(temp, hist[-1]["heater"])
    response = {
        "pump_power": int(pump),
        "upper_valve_open": bool(valve),
        "heater_on": bool(heater),
        "reason": short_reason(level, temp, pump, heater, valve),
    }
    records.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_msg(level, temp, hist)},
            {"role": "assistant", "content": json.dumps(response)},
        ]
    })

# ---------- 500 anti-hallucination correction pairs ----------
ah_pool = [(l, t) for l in [10.0, 25.0, 40.0, 45.0, 47.0, 60.0, 80.0]
                 for t in [25.0, 32.0, 34.5, 35.0, 35.5, 38.0, 45.0]]
random.shuffle(ah_pool)
ah_scenarios = (ah_pool * (500 // len(ah_pool) + 1))[:500]

for level, temp in ah_scenarios:
    prev_heater = random.choice([True, False])
    hist = make_history(level, temp, prev_heater)
    pump, valve = compute_pump(level)
    heater = compute_heater(temp, hist[-1]["heater"])
    user_msg = build_user_msg(level, temp, hist)

    corrupted = json.dumps({
        "pump_power": str(pump) if random.random() > 0.7 else pump,
        "upper_valve_open": ("open" if valve else "closed") if random.random() > 0.5 else valve,
        "heater_on": ("true" if heater else "false") if random.random() > 0.5 else heater,
        "reason": short_reason(level, temp, pump, heater, valve),
        "expected_next_level": round(level + random.uniform(-2, 2), 2),
    })
    correct = {
        "pump_power": int(pump),
        "upper_valve_open": bool(valve),
        "heater_on": bool(heater),
        "reason": short_reason(level, temp, pump, heater, valve),
    }
    records.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT
             + " If a previous response had wrong types or extra keys, fix it."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": corrupted},
            {"role": "user", "content":
             "That response had format errors (string booleans, extra keys, or wrong types). "
             "Output ONLY the correct JSON with the 4 required keys and proper types."},
            {"role": "assistant", "content": json.dumps(correct)},
        ]
    })

random.shuffle(records)

with OUT_FILE.open("w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"\nSaved {len(records)} records to {OUT_FILE}")
print(f"  Normal: {len(raw)}")
print(f"  Anti-hallucination: 500")
