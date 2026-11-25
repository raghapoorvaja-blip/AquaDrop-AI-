import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ["AQUADROP_MODEL_DIR"] = tempfile.mkdtemp(); os.environ["AQUADROP_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import pandas as pd, pytest
from app import allocation, db, faults, ml
from app.simulation import Sim, DEFAULT_ZONES

def zones(): return [dict(z) for z in Sim().zones]
def needs(zs): return {z["id"]: {"irrigate": True, "probability": 0.9} for z in zs}

def test_fallback_when_no_model():
    p = ml.predict(zones()[0], {"temperature": 30, "humidity": 50})
    assert "NOT an ML" in p["source"] and p["irrigate"] is True

def test_train_save_load_predict():
    m = ml.train(); assert m["metrics"]["accuracy"] > 0.8 and "synthetic" in m["source"]
    assert ml.status()["rows"] == 1500
    p = ml.predict({"moisture": 15, "threshold": 30}, {"temperature": 30, "humidity": 50})
    assert p["source"].startswith("ml") and p["irrigate"] is True
    assert ml.predict({"moisture": 75, "threshold": 30}, {"temperature": 25, "humidity": 70})["irrigate"] is False

def test_train_rejects_bad_input():
    df = pd.DataFrame({"a": range(50), "y": range(50)})
    with pytest.raises(ValueError): ml.train(df, {"moisture": "a"}, "y")        # 50 classes, not binary
    with pytest.raises(ValueError): ml.train(df, {"moisture": "zzz"}, "y")      # missing column

def test_train_uploaded_with_missing_values_and_text_target():
    df = ml.synthetic_data(200); df["irrigate"] = df["irrigate"].map({0: "OFF", 1: "ON"}); df.loc[:10, "humidity"] = None
    assert ml.train(df, {f: f for f in ml.FEATURES}, "irrigate", "test")["metrics"]["test_rows"] == 40

@pytest.mark.parametrize("avail", [0, 100, 500, 1500, 50000])
def test_allocation_never_exceeds_available(avail):
    zs = zones(); a = allocation.allocate(zs, needs(zs), avail)
    assert a["total_allocated_l"] <= avail + 0.2
    assert abs(a["total_allocated_l"] + a["total_unmet_l"] - a["total_demand_l"]) < 0.5

def test_priority_order_and_unmet():
    zs = zones(); a = allocation.allocate(zs, needs(zs), 300)
    r = {x["zone_id"]: x for x in a["zones"]}
    assert r[3]["rank"] == 1 and r[3]["allocated_l"] == 300 and r[1]["allocated_l"] == 0 and r[1]["unmet_l"] > 0

def test_no_demand_when_not_needed():
    zs = zones(); a = allocation.allocate(zs, {z["id"]: {"irrigate": False} for z in zs}, 1000)
    assert a["total_demand_l"] == 0 and a["total_allocated_l"] == 0

def run(s, minutes=60):
    for _ in range(minutes): s.tick(1)

def test_irrigation_simulation():
    s = Sim(); zs = s.zones; a = allocation.allocate(zs, needs(zs), s.tank)
    before = {z["id"]: z["moisture"] for z in zs}; tank0 = s.tank
    assert s.start_allocation(a) and s.pump and all(z["valve"] for z in zs if z["allocated_l"])
    s.tick(1); assert s.zone(1)["moisture"] < before[1] + 5          # no instant jump
    run(s, 120)
    assert not s.pump and not any(z["valve"] for z in zs)
    assert s.zone(3)["moisture"] > before[3] + 10 and s.zone(1)["moisture"] > before[1] + 5
    used = sum(z["total_used_l"] for z in zs); assert abs((tank0 - s.tank) - used) < 0.01
    assert used <= a["total_allocated_l"] + 0.5 and len(s.sessions) >= 2

def test_stops_when_tank_empty():
    s = Sim(tank_l=10); s.start_zone(1, 500); run(s, 10)
    assert s.tank >= 0 and not s.zone(1)["valve"]

def test_clog_leak_valve_faults_raise_alerts():
    for kind, val, typ in [("clog", 0.3, "low_flow"), ("leak", 0.6, "high_flow"), ("valve_fault", True, "no_flow")]:
        s = Sim(); s.zone(1)[kind] = val; s.start_zone(1, 200); run(s, 5)
        assert typ in {a["type"] for a in s.alerts}, kind

def test_healthy_run_has_no_flow_alerts():
    s = Sim(); s.start_zone(1, 150); run(s, 60)
    assert not [a for a in s.alerts if a["type"] in ("low_flow", "high_flow", "no_flow", "no_moisture_response")]

def test_low_tank_alert_and_reset():
    s = Sim(tank_l=100); s.tick(1); assert any(a["type"] == "tank_low" for a in s.alerts)
    s.reset(); assert s.tank == 100 and s.alerts == [] and s.t == 0

def test_sensor_fault_rule():
    assert faults.check_sensor(1, 140) and not faults.check_sensor(1, 50)

def test_db_roundtrip():
    db.log("x", {"a": 1}); assert db.history("x")[0]["a"] == 1
