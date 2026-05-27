#!/usr/bin/env python3
"""
rule_controller.py — runtime rule-based controller for the Festo plant.

Lifted out of generate_dataset.py so it can serve THREE roles:
  1. Generate training labels (called by generate_dataset.py).
  2. Generate evaluation gold (called by eval_offline.py).
  3. Act as the runtime safety fallback in the live supervisor when the
     LLM hallucinates or returns a low-trust output (architecture A + C).

Pure-Python, zero deps. Deterministic. Same logic that produced the
labels the LLM is being trained to imitate — so falling back to it is
guaranteed to be at-or-better-than what the LLM would do on cases it
already learned, and safe on cases it hasn't.
"""

from dataclasses import dataclass

# ---- plant constants (must match generate_dataset.py) ----
TARGET_LEVEL = 45
TARGET_TEMP = 35
FLOAT_SWITCH_CUTOFF = 47

FILL_RATE = 3.27          # % per step at pump=100
DRAIN_RATE = 1.58         # % per step when valve open
HEAT_RATE = 0.074         # °C per step when heater on
COOL_RATE = 0.0173        # °C per step ambient loss

TEMP_HYSTERESIS_LOW = 34.0
TEMP_HYSTERESIS_HIGH = 36.0


# ---- output schema (matches the assistant message in dataset_experiment_a.jsonl) ----
@dataclass
class ControlAction:
    pump_power: int            # 0..100
    upper_valve_open: bool
    heater_on: bool
    reason: str

    def to_dict(self):
        return {
            "pump_power": int(self.pump_power),
            "upper_valve_open": bool(self.upper_valve_open),
            "heater_on": bool(self.heater_on),
            "reason": self.reason,
        }


# ---- core decision logic ----
def compute_pump(level: float):
    """Return (pump_power 0..100, upper_valve_open bool) for the given level.

    Float switch (level >= 47%) forces pump off, but the drain valve still
    opens whenever level is above target + 5%, so the plant can recover from
    overfill instead of being stuck.
    """
    error = TARGET_LEVEL - level
    if level >= FLOAT_SWITCH_CUTOFF:
        # pump physically cannot run; drain if we are above the target band
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
    return 0, True            # level above target+5 → drain


def compute_heater(temp: float, prev_heater: bool) -> bool:
    """Hysteresis: ON below 34°C, OFF above 36°C, hold in band."""
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


# ---- single entry point for the live supervisor ----
def decide(level: float, temp: float, prev_heater: bool = False) -> ControlAction:
    """
    Compute one control action from the current sensor reading.
    `prev_heater` is the heater state on the previous cycle (for hysteresis).

    Use from the live supervisor when LLM output is hallucinated or low-trust:

        from rule_controller import decide
        action = decide(level=current_level, temp=current_temp, prev_heater=last_heater)
        cpp_command = action.to_dict()
    """
    pump, valve = compute_pump(level)
    heater = compute_heater(temp, prev_heater)
    reason = short_reason(level, temp, pump, heater, valve)
    return ControlAction(
        pump_power=int(pump),
        upper_valve_open=bool(valve),
        heater_on=bool(heater),
        reason=reason,
    )


# ---- forward dynamics model — used by architecture A (predictive trust) ----
def predict_next_state(level: float, temp: float,
                        pump: int, valve_open: bool, heater_on: bool,
                        prev_heater_on_steps: int = 0):
    """
    Deterministic forward model — what the plant SHOULD do given the action.

    Returns: (predicted_level, predicted_temp, updated_inertia_counter).

    The supervisor calls this with the LLM's chosen action. On the next cycle
    it compares the predicted level/temp against the actual sensor reading
    to compute a TRUST SCORE for the LLM's understanding of the dynamics.

    Note: this is a noise-free version — used as the "expected" outcome.
    Compare against actual with a tolerance (e.g. ±0.5% level, ±0.1°C temp).
    """
    # level dynamics
    dl = 0.0
    if pump > 0 and level < FLOAT_SWITCH_CUTOFF:
        dl += FILL_RATE * pump / 100.0
    if valve_open:
        dl -= DRAIN_RATE
    next_level = max(0.0, min(100.0, level + dl))
    if next_level >= FLOAT_SWITCH_CUTOFF:
        next_level = float(FLOAT_SWITCH_CUTOFF)

    # temperature dynamics with thermal inertia
    THERMAL_INERTIA_STEPS = 3
    INERTIA_FACTOR = 0.3
    if heater_on:
        dt = HEAT_RATE
        next_inertia = THERMAL_INERTIA_STEPS
    elif prev_heater_on_steps > 0:
        dt = HEAT_RATE * INERTIA_FACTOR
        next_inertia = prev_heater_on_steps - 1
    else:
        dt = 0.0
        next_inertia = 0
    dt -= COOL_RATE
    next_temp = max(15.0, min(95.0, temp + dt))

    return round(next_level, 2), round(next_temp, 3), next_inertia


def trust_score(predicted_level, predicted_temp, actual_level, actual_temp,
                level_tol=1.0, temp_tol=0.2):
    """
    Architecture A — compare last cycle's prediction against this cycle's
    actual sensor reading. Returns a float in [0, 1].

    1.0 = both within tolerance (LLM's predicted action is consistent with
          plant dynamics).
    0.0 = both off by >= 5x tolerance (LLM does not understand dynamics).

    Linear interpolation between. The supervisor compares this to a
    threshold (e.g. 0.6) to decide LLM-vs-rule routing.
    """
    def err_to_score(err, tol):
        if err <= tol:
            return 1.0
        if err >= 5 * tol:
            return 0.0
        return 1.0 - (err - tol) / (4 * tol)

    s_level = err_to_score(abs(predicted_level - actual_level), level_tol)
    s_temp = err_to_score(abs(predicted_temp - actual_temp), temp_tol)
    return min(s_level, s_temp)         # both must be reasonable


# ---- self-test ----
if __name__ == "__main__":
    cases = [
        (10.0, 25.0, False),
        (44.0, 35.0, True),
        (60.0, 40.0, False),
        (47.5, 35.0, False),
    ]
    print("=== rule_controller smoke test ===")
    for lv, t, ph in cases:
        a = decide(lv, t, prev_heater=ph)
        print(f"  level={lv:5.1f}%  temp={t:5.2f}°C  prev_heater={ph}")
        print(f"    -> {a.to_dict()}")

    print("\n=== forward model + trust ===")
    pred_l, pred_t, _ = predict_next_state(
        level=44.0, temp=35.0, pump=10, valve_open=False, heater_on=True
    )
    print(f"  predicted next: level={pred_l}, temp={pred_t}")
    print(f"  trust if actual=(44.3, 35.07): "
          f"{trust_score(pred_l, pred_t, 44.3, 35.07):.3f}")
    print(f"  trust if actual=(50.0, 40.0):  "
          f"{trust_score(pred_l, pred_t, 50.0, 40.0):.3f}")
