#!/usr/bin/env python3
"""
Autonomous LLM Direct Control Agent for Hydraulic Two-Tank System
exp07 (exp05 base) - Predictive Look-Ahead + All Prompting Variants Test
- LLM makes ALL decisions freely with zero external logic
- Predictive look-ahead: 3-step horizon simulation added to ALL variants
- 3 hardware safety limits (cannot be overridden):
  1. Temp > 38C  -> heater OFF
  2. Level > 70% -> pump OFF, valve OPEN (overflow)
  3. No simultaneous pump + valve (mechanical conflict)
- Target level: 45%, Target temp: 35C (reference only)
"""

# =====================
# IMPORTS
# =====================
import os
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/scratch/rayarvid/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/scratch/rayarvid/hf_cache/hub")

# Load HF token from /scratch/rayarvid/Experiments/.env (HUGGING_FACE_KEY=...)
_env = "/scratch/rayarvid/Experiments/.env"
if os.path.exists(_env):
    for _ln in open(_env):
        if _ln.startswith("HUGGING_FACE_KEY"):
            _tok = _ln.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["HF_TOKEN"] = _tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = _tok
            break

import json
import re
import time
import atexit
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# =====================
# PROMPT VARIANT (selects which prompting recipe to run)
# =====================
# Choose one of: v1_plain, v2_system, v3_cot, v4_fewshot
# Set via env var, e.g. PROMPT_VARIANT=v2_system python -u llm_supervisor_exp07.py
PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "v4_fewshot")
assert PROMPT_VARIANT in ("v1_plain", "v2_system", "v3_cot", "v4_fewshot"), \
    f"unknown PROMPT_VARIANT: {PROMPT_VARIANT}"

# =====================
# FILE PATHS
# =====================
SENSOR_LOG_PATH    = "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/json_data.txt"
LLM_CONTROL_PATH   = "/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/llm_control.json"
# Summary + unified log are split per variant so multiple runs don't clobber each other.
SUMMARY_STATS_PATH = f"/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/llm_summary_stats_{PROMPT_VARIANT}.json"
UNIFIED_LOG_PATH   = f"/scratch/rayarvid/Experiments/Experiment_2_equal_cycles/festo_live/llm_unified_log_{PROMPT_VARIANT}.jsonl"

# =====================
# LOGGING
# =====================
logger = logging.getLogger("LLM_UNIFIED_LOGGER")
logger.setLevel(logging.INFO)
os.makedirs(os.path.dirname(UNIFIED_LOG_PATH), exist_ok=True)
handler = logging.FileHandler(UNIFIED_LOG_PATH)
handler.setFormatter(logging.Formatter('%(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

# =====================
# CONFIG
# =====================
CHECK_INTERVAL  = 5
MAX_DECISIONS   = 250   # exit naturally after this many decisions
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B-Instruct"

# Auto-select latest checkpoint from exp07 adapter dir
_adapter_dir = "/scratch/rayarvid/Experiments/exp07/festo_llama3.2_finetuned_exp07"
_checkpoints = sorted(
    [p for p in os.listdir(_adapter_dir) if p.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1])
) if os.path.isdir(_adapter_dir) else []
ADAPTER_PATH = os.path.join(_adapter_dir, _checkpoints[-1]) if _checkpoints else _adapter_dir

TARGET_LEVEL    = 45.0
TARGET_TEMP     = 35.0
LEVEL_TOLERANCE = 5.0   # ±5% band
TEMP_TOLERANCE  = 3.0   # ±3C band
HISTORY_SIZE    = 6

# True hardware limits (matching exp07 training context)
MAX_SAFE_TEMP   = 38.0
MAX_SAFE_LEVEL  = 70.0

# =====================
# PREDICTIVE LOOK-AHEAD CONSTANTS & HELPERS (NEW in exp07)
# =====================
PREDICTION_HORIZON  = 3
FILL_RATE           = 3.27
DRAIN_RATE          = 1.58
HEAT_RATE           = 0.074
COOL_RATE           = 0.0173
THERMAL_INERTIA     = 3
INERTIA_FACTOR      = 0.3
FLOAT_CUTOFF        = 47.0
TEMP_HYSTERESIS_LOW = 34.0
TEMP_HYSTERESIS_HIGH= 36.0


def _sim_step(level, temp, pump, valve, heater, inertia):
    dl = 0.0
    if pump > 0 and level < FLOAT_CUTOFF:
        dl += FILL_RATE * pump / 100.0
    if valve:
        dl -= DRAIN_RATE
    level = round(max(0, min(100, level + dl)), 1)
    if level >= FLOAT_CUTOFF:
        level = FLOAT_CUTOFF
    if heater:
        dt = HEAT_RATE
        inertia = THERMAL_INERTIA
    elif inertia > 0:
        dt = HEAT_RATE * INERTIA_FACTOR
        inertia -= 1
    else:
        dt = 0.0
    dt -= COOL_RATE
    temp = round(max(15, min(95, temp + dt)), 2)
    return level, temp, inertia


def _nominal_action(level, temp, prev_heater):
    """Rule-based nominal action for computing the horizon prediction."""
    err = TARGET_LEVEL - level
    if level >= FLOAT_CUTOFF:
        pump, valve = 0, err < -5
    elif err > 25: pump, valve = 100, False
    elif err > 15: pump, valve = 80,  False
    elif err > 10: pump, valve = 60,  False
    elif err > 5:  pump, valve = 40,  False
    elif err > 2:  pump, valve = 20,  False
    elif err > 0:  pump, valve = 10,  False
    elif err >= -5: pump, valve = 0,  False
    else:           pump, valve = 0,  True
    heater = (temp < TEMP_HYSTERESIS_LOW) or (prev_heater and temp <= TEMP_HYSTERESIS_HIGH)
    return pump, valve, heater


def compute_horizon(level, temp, recent_history):
    """Simulate PREDICTION_HORIZON steps ahead using nominal action."""
    prev_heater = recent_history[-1]["heater_on"] if recent_history else False
    pump, valve, heater = _nominal_action(level, temp, prev_heater)
    inertia = 0
    future = []
    for _ in range(PREDICTION_HORIZON):
        level, temp, inertia = _sim_step(level, temp, pump, valve, heater, inertia)
        future.append((round(level, 1), round(temp, 2)))
    return future


def fmt_horizon(future):
    return " | ".join(f"t+{i+1}: level={lv:.1f}%, temp={tp:.2f}C"
                      for i, (lv, tp) in enumerate(future))


# =====================
# LLM SETUP
# =====================
print(f"Loading fine-tuned Llama 3.2 exp07 model...")
print(f"Adapter: {ADAPTER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"},
    low_cpu_mem_usage=True,
)
print(f"Base model loaded. Loading PEFT adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=False,
)
model.eval()
print("exp07 Model loaded - Predictive Look-Ahead capability test")

# =====================
# METRICS
# =====================
class MetricsCollector:
    def __init__(self):
        self.start_time = datetime.now()
        self.session_decisions = 0
        self.safety_overrides = 0
        self.level_hits = 0
        self.temp_hits = 0
        self.overall_hits = 0
        self.time_to_target = None
        self.current_streak = 0
        self.longest_streak = 0
        self.response_times = []
        self.level_errors = []
        self.temp_errors = []
        # Hallucination tracking
        # Two categories:
        #   - control: corrupts the action itself; supervisor must fall back
        #   - text:    schema cosmetics; action still extractable
        self.hallucinations = 0
        self.control_hallucinations = 0
        self.text_hallucinations = 0
        self.hallucination_types = {
            "parse_error":   0,
            "missing_field": 0,
            "wrong_type":    0,
            "out_of_range":  0,
            "wrong_key":     0,
            "extra_content": 0,
        }
        self._decision_seen_types = set()
        self._decision_overall_flagged = False
        self._decision_control_flagged = False
        self._decision_text_flagged = False

    CONTROL_TYPES = frozenset({"parse_error", "missing_field", "wrong_type", "out_of_range"})
    TEXT_TYPES    = frozenset({"wrong_key", "extra_content"})

    def begin_decision(self):
        self._decision_seen_types = set()
        self._decision_overall_flagged = False
        self._decision_control_flagged = False
        self._decision_text_flagged = False

    def record_decision(self, data, latency, safety_triggered):
        self.session_decisions += 1
        if safety_triggered:
            self.safety_overrides += 1
        self.response_times.append(latency)

        level_error = abs(data['upper_level'] - TARGET_LEVEL)
        temp_error  = abs(data['temp']        - TARGET_TEMP)
        self.level_errors.append(level_error)
        self.temp_errors.append(temp_error)

        level_hit = level_error <= LEVEL_TOLERANCE
        temp_hit  = temp_error  <= TEMP_TOLERANCE
        if level_hit: self.level_hits += 1
        if temp_hit:  self.temp_hits  += 1
        if level_hit and temp_hit:
            self.overall_hits += 1
            if self.time_to_target is None:
                self.time_to_target = self.session_decisions
            self.current_streak += 1
            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak
        else:
            self.current_streak = 0

        return level_error, temp_error, level_hit, temp_hit

    def get_current_stats(self):
        total = self.session_decisions
        if total == 0:
            return {"timestamp": datetime.now().isoformat(), "decisions": 0}
        return {
            "timestamp": datetime.now().isoformat(),
            "decisions": total,
            "accuracy": {
                "level_mae": sum(self.level_errors) / total,
                "temp_mae":  sum(self.temp_errors)  / total,
                "level_control_accuracy": (self.level_hits / total) * 100,
                "temp_control_accuracy":  (self.temp_hits  / total) * 100,
                "overall_accuracy":       (self.overall_hits / total) * 100,
                "time_to_target_cycles":  self.time_to_target,
                "band_stability_longest_run": self.longest_streak,
            },
            "latency": {"avg_ms": (sum(self.response_times) / total) * 1000},
            "safety_override_rate":         (self.safety_overrides / total) * 100,
            "hallucination_rate":           (self.hallucinations / total) * 100,
            "control_hallucination_rate":   (self.control_hallucinations / total) * 100,
            "text_hallucination_rate":      (self.text_hallucinations / total) * 100,
            "hallucination_types":          dict(self.hallucination_types),
        }

    def record_hallucination(self, h_type):
        # Type counter: bumped at most once per decision per type
        if h_type in self.hallucination_types and h_type not in self._decision_seen_types:
            self.hallucination_types[h_type] += 1
            self._decision_seen_types.add(h_type)
        # Overall: at most once per decision
        if not self._decision_overall_flagged:
            self.hallucinations += 1
            self._decision_overall_flagged = True
        # Per-category: at most once per decision per category
        if h_type in self.CONTROL_TYPES and not self._decision_control_flagged:
            self.control_hallucinations += 1
            self._decision_control_flagged = True
        if h_type in self.TEXT_TYPES and not self._decision_text_flagged:
            self.text_hallucinations += 1
            self._decision_text_flagged = True

    def finalize_session(self):
        with open(SUMMARY_STATS_PATH, "w") as f:
            json.dump(self.get_current_stats(), f, indent=2)

# =====================
# SENSOR READ (live single-object json_data.txt with retry)
# =====================
def read_sensor():
    for _ in range(8):
        try:
            with open(SENSOR_LOG_PATH, "r") as f:
                d = json.load(f)
            return {
                "lower_level": float(d["RightTankWaterlevel"]),
                "upper_level": float(d["LeftTankWaterlevel"]),
                "temp":        float(d["Temperature"]),
            }
        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
            time.sleep(0.25)
    raise RuntimeError("read_sensor: 8 retries failed")

# =====================
# PROMPT VARIANTS — exp05 variants preserved, predictive look-ahead ADDED on top
# =====================

# v1_plain: NO system message. Task + schema + targets + predictive look-ahead in user message.
PROMPT_V1_PLAIN_USER_PREFIX = (
    "Decide pump_power (integer 0-100), upper_valve_open (boolean), "
    "heater_on (boolean), reason (short string) and output ONLY a single "
    "JSON object with exactly those keys. Targets: level 45% (40-50%), "
    "temp 35C (32-38C). "
    "A 3-step prediction of future states is provided — use it to verify "
    "your action will converge to target before deciding.\n"
)

# v2_system: role + schema + targets + predictive look-ahead cue. No CoT, no examples.
PROMPT_V2_SYSTEM = (
    "You are a Festo water tank controller with predictive look-ahead. "
    "Output ONLY a single JSON object "
    "with exactly these keys: pump_power (integer 0-100), upper_valve_open "
    "(boolean true/false), heater_on (boolean true/false), reason (short string). "
    "NEVER use strings for boolean fields. NEVER add extra keys. "
    "Targets: level 45% (40-50%), temp 35C (32-38C). "
    "A 3-step prediction of future states is provided in the user message — "
    "use it to confirm your action converges to target over the horizon."
)

# v3_cot: v2 + Chain-of-Thought cue extended to include horizon check.
PROMPT_V3_COT = PROMPT_V2_SYSTEM + (
    " Think step by step: (1) assess level error, (2) assess temp error, "
    "(3) check the predicted future states to confirm convergence, "
    "then emit the JSON."
)

# v4_fewshot: v3 + 3 few-shot examples updated with horizon predictions.
PROMPT_V4_FEWSHOT = PROMPT_V3_COT + (
    " Examples:\n"
    'Input: level=10%, temp=30C, predicted t+1=13.3%, t+2=16.6%, t+3=19.9% -> {"pump_power": 100, "upper_valve_open": false, "heater_on": true, "reason": "level critical low, pump max; temp low, heater on; horizon converging"}\n'
    'Input: level=44%, temp=35C, predicted t+1=44.1%, t+2=44.3%, t+3=44.5% -> {"pump_power": 10, "upper_valve_open": false, "heater_on": true, "reason": "level slightly low; temp in band, heater on; horizon stable"}\n'
    'Input: level=60%, temp=40C, predicted t+1=58.4%, t+2=56.8%, t+3=55.3% -> {"pump_power": 0, "upper_valve_open": true, "heater_on": false, "reason": "level high, draining; temp high, heater off; horizon draining toward target"}'
)

# Resolve the active variant.
SYSTEM_PROMPT = {
    "v1_plain":   None,
    "v2_system":  PROMPT_V2_SYSTEM,
    "v3_cot":     PROMPT_V3_COT,
    "v4_fewshot": PROMPT_V4_FEWSHOT,
}[PROMPT_VARIANT]
USER_PREFIX = PROMPT_V1_PLAIN_USER_PREFIX if PROMPT_VARIANT == "v1_plain" else ""


def build_prompt(data, recent_history):
    # History string (same as exp05)
    history_str = ""
    if recent_history:
        lines = []
        for i, h in enumerate(reversed(recent_history[-HISTORY_SIZE:])):
            valve_str  = "true" if h["upper_valve_open"] else "false"
            heater_str = "true" if h["heater_on"]        else "false"
            lines.append(
                f"t-{i+1}: level={h['upper_level']:.1f}%, temp={h['temp']:.2f}C, "
                f"pump={h['pump_power']}, valve={valve_str}, heater={heater_str}"
            )
        history_str = "\nHistory: " + " | ".join(lines)

    # Predictive look-ahead (NEW in exp07) — added to ALL variants
    future = compute_horizon(data['upper_level'], data['temp'], recent_history)
    horizon_str = "\nPredicted future (if recommended action applied): " + fmt_horizon(future)

    return (
        f"{USER_PREFIX}"
        f"Upper level: {data['upper_level']:.1f}%, Temperature: {data['temp']:.2f}C, "
        f"Target level: 45%, Target temp: 35C"
        f"{history_str}"
        f"{horizon_str}"
    )

# =====================
# LLM CALL (system + user, greedy decoding to match offline eval)
# =====================
def ask_llm(prompt, metrics):
    start = time.time()
    raw = ""
    metrics.begin_decision()
    try:
        if SYSTEM_PROMPT is None:
            # v1_plain: user-only message (no system role)
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
        # Prime with '{' to force the model to complete a JSON object
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        chat_text_primed = chat_text + '{"'
        inputs = tokenizer(chat_text_primed, return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        attention_mask = (inputs.input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=250,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs.input_ids.shape[1]
        raw = '{"'  + tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        latency = time.time() - start
        print(f"   -> LLM raw: {raw[:200]}")

        json_match = re.search(r'\{[^{}]*"pump_power"[^{}]*\}', raw, re.DOTALL) or \
                     re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            metrics.record_hallucination("parse_error")
            raise ValueError("No JSON found")

        json_str = json_match.group().replace("'", '"')
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            metrics.record_hallucination("parse_error")
            raise

        # Trailing prose after the JSON closing brace = text-generation hallucination
        trailing = raw[json_match.end():].strip()
        if len(trailing) > 5:
            metrics.record_hallucination("extra_content")

        # Schema / type / range checks. Per-decision dedup is handled inside
        # MetricsCollector, so each call site can record without guarding.
        expected_keys = {"pump_power", "upper_valve_open", "heater_on", "reason"}
        if set(parsed.keys()) - expected_keys:
            metrics.record_hallucination("wrong_key")

        for required in ("pump_power", "heater_on", "upper_valve_open"):
            if required not in parsed:
                metrics.record_hallucination("missing_field")

        pump_raw = parsed.get("pump_power", 0)
        if isinstance(pump_raw, bool) or not isinstance(pump_raw, (int, float)):
            metrics.record_hallucination("wrong_type")
            pump = 0
        else:
            pump = int(pump_raw)
            if pump < 0 or pump > 100:
                metrics.record_hallucination("out_of_range")
            pump = max(0, min(100, pump))

        heater_raw = parsed.get("heater_on", False)
        if not isinstance(heater_raw, bool):
            metrics.record_hallucination("wrong_type")
            heater = str(heater_raw).strip().lower() in ("true", "on", "1")
        else:
            heater = heater_raw

        valve_raw = parsed.get("upper_valve_open", parsed.get("valve_open"))
        if valve_raw is None:
            metrics.record_hallucination("missing_field")
            valve = False
        elif not isinstance(valve_raw, bool):
            metrics.record_hallucination("wrong_type")
            valve = str(valve_raw).strip().lower() in ("true", "on", "1")
        else:
            valve = valve_raw

        reason = str(parsed.get("reason", "none"))

        return {
            "pump_power": pump,
            "heater_on":  heater,
            "valve_open": valve,
            "reason":     reason,
            "latency":    latency,
            "raw_response": raw,
        }
    except Exception as e:
        latency = time.time() - start
        if "parse_error" not in str(e).lower():
            try: metrics.record_hallucination("parse_error")
            except Exception: pass
        print(f"LLM ERROR: {e}")
        return {
            "pump_power": 0,
            "heater_on":  False,
            "valve_open": True,
            "reason":     f"parse_error: {str(e)[:50]}",
            "latency":    latency,
            "raw_response": str(e),
        }


# =====================
# HARDWARE SAFETY ONLY
# =====================
def hardware_safety(data, decision):
    pump   = decision["pump_power"]
    heater = decision["heater_on"]
    valve  = decision["valve_open"]
    triggered = False

    if data["temp"] > MAX_SAFE_TEMP and heater:
        heater = False
        triggered = True
        print(f"    HARDWARE SAFETY: Temp {data['temp']:.1f}C > {MAX_SAFE_TEMP}C - heater OFF")

    if data["upper_level"] > MAX_SAFE_LEVEL:
        if pump > 0:
            pump = 0
            triggered = True
            print(f"    HARDWARE SAFETY: Level {data['upper_level']:.1f}% > {MAX_SAFE_LEVEL}% - pump OFF")
        if not valve:
            valve = True
            triggered = True
            print(f"    HARDWARE SAFETY: Level {data['upper_level']:.1f}% > {MAX_SAFE_LEVEL}% - valve OPEN")

    if pump > 0 and valve:
        pump = 0
        triggered = True
        print(f"    HARDWARE SAFETY: Pump+valve simultaneous - pump OFF")

    return {
        "pump_power": pump,
        "heater_on":  heater,
        "valve_open": valve,
        "reason":     decision["reason"],
    }, triggered

# =====================
# UNIFIED LOG
# =====================
def unified_log(data, decision, metrics, level_error, temp_error, level_hit, temp_hit, safety):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "state": data,
        "decision": {
            "pump_power": decision["pump_power"],
            "heater_on":  decision["heater_on"],
            "valve_open": decision["valve_open"],
            "reason":     decision["reason"],
        },
        "metrics": {
            "level_error": level_error,
            "temp_error":  temp_error,
            "level_hit":   level_hit,
            "temp_hit":    temp_hit,
            "latency_ms":  decision.get("latency", 0) * 1000,
            "hardware_safety_triggered": safety,
        },
        "running_summary": metrics.get_current_stats(),
    }
    logger.info(json.dumps(entry))

# =====================
# MAIN LOOP
# =====================
def main():
    if os.path.exists(SUMMARY_STATS_PATH):
        os.remove(SUMMARY_STATS_PATH)

    metrics = MetricsCollector()
    atexit.register(metrics.finalize_session)
    recent_history = []

    print("=" * 70)
    print("LLM CONTROL AGENT exp07 - PREDICTIVE LOOK-AHEAD TEST")
    print("=" * 70)
    print(f"Prompt variant: {PROMPT_VARIANT}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} steps")
    print(f"Target: Upper Tank Level = {TARGET_LEVEL}% +/- {LEVEL_TOLERANCE}%")
    print(f"Target: Temperature      = {TARGET_TEMP}C +/- {TEMP_TOLERANCE}C")
    print(f"Hardware limits: temp > {MAX_SAFE_TEMP}C, level > {MAX_SAFE_LEVEL}%, no pump+valve")
    print(f"History window: {HISTORY_SIZE} steps")
    print("=" * 70)

    while metrics.session_decisions < MAX_DECISIONS:
        data   = read_sensor()
        prompt = build_prompt(data, recent_history)
        decision = ask_llm(prompt, metrics)
        print(f"   -> LLM decision: Pump={decision['pump_power']}%, Heater={decision['heater_on']}, Valve={decision['valve_open']}")
        print(f"   -> Reason: {decision['reason'][:80]}")

        safe_decision, safety_triggered = hardware_safety(data, decision)

        level_error, temp_error, level_hit, temp_hit = metrics.record_decision(
            data, decision.get("latency", 0), safety_triggered)

        unified_log(data, safe_decision, metrics, level_error, temp_error,
                    level_hit, temp_hit, safety_triggered)

        recent_history.append({
            "upper_level":      data["upper_level"],
            "temp":             data["temp"],
            "pump_power":       safe_decision["pump_power"],
            "upper_valve_open": safe_decision["valve_open"],
            "heater_on":        safe_decision["heater_on"],
        })
        recent_history = recent_history[-HISTORY_SIZE:]

        control_output = {
            "pump_on":     safe_decision["pump_power"] > 0,
            "pump_power":  safe_decision["pump_power"],
            "heater_on":   safe_decision["heater_on"],
            "valve_open":  safe_decision["valve_open"],
            "timestamp":   datetime.now().isoformat(),
        }
        temp_path = LLM_CONTROL_PATH + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(control_output, f, indent=2)
        os.replace(temp_path, LLM_CONTROL_PATH)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Decision #{metrics.session_decisions}")
        print(f"  Upper={data['upper_level']:.1f}%  Lower={data['lower_level']:.1f}%  Temp={data['temp']:.2f}C")
        print(f"  Level error: {level_error:.2f}% ({'HIT' if level_hit else 'MISS'})")
        print(f"  Temp error:  {temp_error:.2f}C ({'HIT' if temp_hit else 'MISS'})")
        print(f"  Control: Pump={safe_decision['pump_power']}%  Heater={safe_decision['heater_on']}  Valve={safe_decision['valve_open']}")
        if safety_triggered:
            print(f"  HARDWARE SAFETY APPLIED")
        print("=" * 70)

        with open(SUMMARY_STATS_PATH, "w") as f:
            json.dump(metrics.get_current_stats(), f, indent=2)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
