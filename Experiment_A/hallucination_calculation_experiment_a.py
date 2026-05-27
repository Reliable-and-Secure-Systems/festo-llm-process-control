#!/usr/bin/env python3
import re, json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/rayarvid/sim_src/job_1344.log"
total = parse_err = extra_content = wrong_keys = missing = wrong_type = oor = 0
samples = []
expected = {"pump_power", "upper_valve_open", "heater_on", "reason"}

with open(path) as f:
    for line in f:
        if "LLM raw:" not in line:
            continue
        total += 1
        raw = line.split("LLM raw:", 1)[1].rstrip("\n")
        # mimic the supervisor's regex
        m = (re.search(r'\{[^{}]*"pump_power"[^{}]*\}', raw, re.DOTALL)
             or re.search(r'\{.*?\}', raw, re.DOTALL))
        if not m:
            parse_err += 1
            continue
        js = m.group().replace("'", '"')
        try:
            p = json.loads(js)
        except Exception:
            parse_err += 1
            continue
        trailing = raw[m.end():].strip()
        if len(trailing) > 5:
            extra_content += 1
            if len(samples) < 3:
                samples.append(trailing[:140])
        if set(p.keys()) - expected:
            wrong_keys += 1
        for k in ("pump_power", "heater_on", "upper_valve_open"):
            if k not in p and not (k == "upper_valve_open" and "valve_open" in p):
                missing += 1
                break
        pp = p.get("pump_power")
        if isinstance(pp, bool) or not isinstance(pp, (int, float)):
            wrong_type += 1
        elif pp < 0 or pp > 100:
            oor += 1

print(f"total decisions in log:  {total}")
print(f"parse_error:             {parse_err}")
print(f"extra_content (trailing):{extra_content}   ({100*extra_content/total:.1f}%)")
print(f"wrong_key:               {wrong_keys}")
print(f"missing_field:           {missing}")
print(f"wrong_type:              {wrong_type}")
print(f"out_of_range:            {oor}")
old = parse_err + wrong_keys + missing + wrong_type + oor
new = old + extra_content
print()
print(f"old-definition hallucination rate: {100*old/total:.2f}%  ({old}/{total})")
print(f"new-definition (incl. trailing):   {100*new/total:.2f}%  ({new}/{total})")
print()
print("sample trailing content:")
for s in samples:
    print("  -", s)
