"""Rule-based fault flags. These are CAUTIOUS indications, not confirmed diagnoses."""
TH = {"low_flow": 0.6, "high_flow": 1.3, "tank_low_pct": 15, "min_response": 0.3}

def mk(zone, typ, sev, desc, cause, action):
    return {"zone_id": zone, "type": typ, "severity": sev, "description": desc, "possible_cause": cause, "suggested_inspection": action}

def check_flow(zid, expected, measured, pump_on):
    if expected <= 0: return []
    r = measured / expected
    if pump_on and r < 0.05:
        return [mk(zid, "no_flow", "high", "Valve open and pump on, but almost no flow measured.", "Valve stuck closed, pump fault, or flow-sensor fault.", "Check pump, valve wiring and flow sensor.")]
    if r < TH["low_flow"]:
        return [mk(zid, "low_flow", "medium", f"Flow is {r:.0%} of expected. Possible irrigation delivery abnormality.", "Partial clogging or low supply pressure (not confirmed).", "Check flow, valve and drip-line conditions.")]
    if r > TH["high_flow"]:
        return [mk(zid, "high_flow", "high", f"Flow is {r:.0%} of expected. Possible irrigation delivery abnormality.", "Possible leak or broken line (not confirmed).", "Walk the drip line and check fittings.")]
    return []

def check_tank(pct):
    return [mk(0, "tank_low", "medium", f"Tank level low ({pct:.0f}%).", "Heavy use or insufficient refill.", "Refill tank.")] if pct < TH["tank_low_pct"] else []

def check_sensor(zid, moisture):
    return [mk(zid, "sensor_range", "medium", f"Moisture reading {moisture} is outside 0-100%.", "Sensor or wiring fault.", "Inspect/replace moisture sensor.")] if not 0 <= moisture <= 100 else []

def check_response(zid, expected_gain, observed_gain):
    if expected_gain > 3 and observed_gain < TH["min_response"] * expected_gain:
        return [mk(zid, "no_moisture_response", "medium", f"Moisture rose {observed_gain:.1f} pts vs {expected_gain:.1f} expected.", "Leak, clogged emitters, or a stuck moisture sensor (cannot tell which).", "Inspect emitters and test the moisture sensor.")]
    return []
