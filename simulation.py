"""Irrigation simulation. ALL readings here are SIMULATED. Real hardware would replace this module."""
import copy, math, time
from . import faults
from .allocation import litres_per_pct

DEFAULT_ZONES = [
    dict(id=1, name="Zone 1", crop="Tomato", soil="Loam", area=100.0, moisture=28.0, threshold=35.0, target=60.0, root_depth_mm=60.0, sensitivity=0.9),
    dict(id=2, name="Zone 2", crop="Wheat", soil="Clay", area=150.0, moisture=45.0, threshold=30.0, target=55.0, root_depth_mm=60.0, sensitivity=0.5),
    dict(id=3, name="Zone 3", crop="Chilli", soil="Sandy", area=80.0, moisture=22.0, threshold=35.0, target=60.0, root_depth_mm=60.0, sensitivity=0.8),
]
RUNTIME = {"allocated_l": 0.0, "delivered_l": 0.0, "valve": False, "status": "idle", "pending_l": 0.0, "clog": 1.0, "leak": 0.0,
           "valve_fault": False, "total_used_l": 0.0, "session": None, "last_flow_lpm": 0.0}

class Sim:
    def __init__(self, tank_l=1200.0, capacity_l=2000.0):
        self.capacity, self.tank0 = capacity_l, tank_l
        self.cfg = {"flow_lpm": 20.0, "evap_pct_per_min": 0.02, "infiltration_min": 5.0}
        self.sink = None   # optional callback(kind, data): lets the app record events (sessions, sensor readings, alerts) in SQLite
        self.reset()

    def _emit(self, kind, data):
        if self.sink: self.sink(kind, data)

    def snapshot(self):
        """Everything needed to restore the simulation after a restart (JSON-serialisable)."""
        return {"tank": self.tank, "capacity": self.capacity, "pump": self.pump, "mode": self.mode, "t": self.t, "cfg": self.cfg,
                "zones": self.zones, "alerts": self.alerts[-200:], "sessions": self.sessions[-200:], "history": self.history, "aid": self._aid}

    def restore(self, s):
        self.capacity, self.tank = s.get("capacity", self.capacity), s.get("tank", self.tank)
        self.pump, self.mode, self.t = s.get("pump", False), s.get("mode", "manual"), s.get("t", 0.0)
        self.cfg.update(s.get("cfg", {}))
        if s.get("zones"): self.zones = [{**copy.deepcopy(RUNTIME), **z} for z in s["zones"]]
        self.alerts, self.sessions, self.history = s.get("alerts", []), s.get("sessions", []), s.get("history", [])
        self._aid = max([s.get("aid", 0)] + [a["id"] for a in self.alerts])
        for z in self.zones:   # re-link a zone's last session to its entry in the sessions list (they were the same object before saving)
            cur = z.get("session")
            if cur and not z["valve"]:
                z["session"] = next((x for x in self.sessions if x["zone_id"] == cur["zone_id"] and x["start_min"] == cur["start_min"]), cur)

    def reset(self):
        self.zones = [{**copy.deepcopy(z), **copy.deepcopy(RUNTIME)} for z in DEFAULT_ZONES]
        self.tank, self.pump, self.mode, self.t = self.tank0, False, "manual", 0.0
        self.alerts, self.sessions, self.history, self._aid = [], [], [], 0
        self._log()

    def zone(self, zid):
        return next(z for z in self.zones if z["id"] == zid)

    def active(self):
        return any(z["valve"] for z in self.zones)

    def add_alerts(self, items):
        for a in items:
            if any(x["status"] == "active" and x["type"] == a["type"] and x["zone_id"] == a["zone_id"] for x in self.alerts): continue
            self._aid += 1
            self.alerts.append({**a, "id": self._aid, "timestamp": time.strftime("%H:%M:%S"), "sim_minute": round(self.t, 1), "status": "active"})
            self._emit("alert", self.alerts[-1])

    def start_zone(self, zid, litres):
        z = self.zone(zid)
        litres = min(float(litres), self.tank)
        if litres <= 0 or z["valve"]: return False
        z.update(allocated_l=litres, delivered_l=0.0, valve=True, status="irrigating")
        z["session"] = {"zone_id": zid, "start_min": round(self.t, 1), "moisture_before": z["moisture"], "allocated_l": round(litres, 1)}
        self.pump = True
        return True

    def start_allocation(self, alloc):
        return [r["zone_id"] for r in alloc["zones"] if r["allocated_l"] > 0 and self.start_zone(r["zone_id"], r["allocated_l"])]

    def stop_zone(self, z, why="manual stop"):
        if not z["valve"]: return
        z.update(valve=False, status="soaking")
        z["session"].update(end_min=round(self.t, 1), delivered_l=round(z["delivered_l"], 1), reason=why)
        self.sessions.append(z["session"])
        self._emit("session", z["session"])
        if not self.active(): self.pump = False

    def stop_all(self):
        for z in self.zones: self.stop_zone(z)

    def tick(self, minutes=1.0):
        self.t += minutes
        for z in self.zones:
            z["moisture"] = max(0.0, z["moisture"] - self.cfg["evap_pct_per_min"] * minutes)
            z["last_flow_lpm"] = 0.0
            if z["valve"]:
                want = min(self.cfg["flow_lpm"] * minutes, z["allocated_l"] - z["delivered_l"])
                actual = 0.0 if z["valve_fault"] else want * z["clog"]
                billed = min(actual * (1 + z["leak"]), self.tank)   # leak = extra water that never reaches the soil
                self.tank -= billed; z["delivered_l"] += billed; z["total_used_l"] += billed
                z["pending_l"] += billed / (1 + z["leak"])
                z["last_flow_lpm"] = billed / minutes
                self.add_alerts(faults.check_flow(z["id"], want / minutes, z["last_flow_lpm"], self.pump))
                if z["delivered_l"] >= z["allocated_l"] - 0.01: self.stop_zone(z, "allocation used")
                elif z["moisture"] >= z["target"]: self.stop_zone(z, "target reached")
                elif self.tank <= 0.01: self.stop_zone(z, "tank empty")
                elif z["valve_fault"] and z["session"] and self.t - z["session"]["start_min"] >= 3: self.stop_zone(z, "no flow - safety stop")
            moved = z["pending_l"] * (1 - math.exp(-minutes / self.cfg["infiltration_min"]))  # gradual soaking, not instant
            z["pending_l"] -= moved
            z["moisture"] = min(100.0, z["moisture"] + moved / litres_per_pct(z))
            s = z["session"]
            if s and not z["valve"] and z["pending_l"] < 1.0 and "checked" not in s:
                s["checked"] = True; s["moisture_after"] = round(z["moisture"], 1)
                exp = s["delivered_l"] / (1 + z["leak"]) / litres_per_pct(z)   # gain expected if no water was lost
                self.add_alerts(faults.check_response(z["id"], exp, z["moisture"] - s["moisture_before"]))
                z["status"] = "idle"
                self._emit("session_update", {"zone_id": z["id"], "start_min": s["start_min"], "moisture_after": s["moisture_after"]})
            self.add_alerts(faults.check_sensor(z["id"], z["moisture"]))
        self.add_alerts(faults.check_tank(100 * self.tank / self.capacity))
        self._log()
        self._emit("sensor", {"sim_minute": round(self.t, 1), "tank_l": round(self.tank, 1),
                              "zones": {z["id"]: {"moisture": round(z["moisture"], 2), "flow_lpm": round(z["last_flow_lpm"], 2), "used_l": round(z["total_used_l"], 1)} for z in self.zones}})

    def _log(self):
        self.history.append({"t": round(self.t, 1), "tank": round(self.tank, 1), "used": round(sum(z["total_used_l"] for z in self.zones), 1)})
        self.history = self.history[-300:]
