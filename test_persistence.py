"""Persistence test. Run from the project root:  python -m pytest tests -q
Simulates an application restart by re-importing app.main against the same SQLite file."""
import importlib, os, sys, tempfile, unittest

class PersistenceTest(unittest.TestCase):
    def boot(self):
        """Fresh import of app.db / app.main = a restarted backend pointed at the same database file."""
        os.environ["AQUADROP_DB"] = self.db_path
        pkg = importlib.import_module("app")
        for name in ("db", "main"):
            sys.modules.pop(f"app.{name}", None)
            if hasattr(pkg, name): delattr(pkg, name)   # `from . import db` would otherwise reuse the cached module
        return importlib.import_module("app.main")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "test.db")

    def tearDown(self):
        os.environ.pop("AQUADROP_DB", None)
        self.tmp.cleanup()

    def test_zone_and_tank_survive_restart(self):
        m = self.boot()
        m.edit_zone(1, {"name": "Greenhouse", "crop": "Brinjal", "soil": "Sandy", "moisture": 41.5, "threshold": 33})
        m.settings({"tank_l": 777, "flow_lpm": 25, "speed": 2})
        m.mode({"mode": "auto"})

        m2 = self.boot()   # "restart"
        z = m2.sim.zone(1)
        self.assertEqual((z["name"], z["crop"], z["soil"]), ("Greenhouse", "Brinjal", "Sandy"))
        self.assertEqual(z["moisture"], 41.5)
        self.assertEqual(z["threshold"], 33.0)
        self.assertEqual(m2.sim.tank, 777.0)
        self.assertEqual(m2.sim.capacity, 2000.0)
        self.assertEqual(m2.sim.cfg["flow_lpm"], 25.0)
        self.assertEqual(m2.S["speed"], 2.0)
        self.assertEqual(m2.sim.mode, "auto")
        self.assertEqual(m2.state()["tank_l"], 777.0)   # dashboard endpoint reflects saved data

    def test_sessions_alerts_and_history_survive_restart(self):
        m = self.boot()
        m.settings({"tank_l": 300})                       # 15% of capacity -> tank alert
        m.start({"zone_id": 1, "litres": 40})
        for _ in range(5): m.tick({"minutes": 1})         # 20 L/min -> session finishes after 2 min
        used = m.sim.zone(1)["total_used_l"]
        self.assertGreater(used, 0)
        alert_id = m.sim.alerts[0]["id"]
        m.resolve(alert_id)

        events = m.db.history("sensor", 100)
        self.assertEqual(len(events), 5)                  # one sensor reading per tick
        self.assertEqual(len(m.db.history("session")), 1)
        self.assertEqual(len(m.db.history("alert_resolved")), 1)

        m2 = self.boot()   # "restart"
        self.assertEqual(m2.sim.zone(1)["total_used_l"], used)
        self.assertEqual(len(m2.sim.sessions), 1)
        self.assertEqual(m2.sim.sessions[0]["delivered_l"], 40.0)
        self.assertEqual(m2.sim.t, 5.0)
        a = next(a for a in m2.sim.alerts if a["id"] == alert_id)
        self.assertEqual(a["status"], "resolved")
        self.assertEqual(len(m2.sim.history), len(m.sim.history))
        m2.tick({"minutes": 1})                           # keeps running; new alert ids don't collide
        ids = [a["id"] for a in m2.sim.alerts]
        self.assertEqual(len(ids), len(set(ids)))

    def test_reset_is_persisted(self):
        m = self.boot()
        m.edit_zone(2, {"name": "Changed"}); m.reset()
        m2 = self.boot()
        self.assertEqual(m2.sim.zone(2)["name"], "Zone 2")

    def test_fresh_database_uses_defaults(self):
        m = self.boot()
        self.assertEqual(m.sim.tank, 1200.0)
        self.assertEqual(m.sim.zone(1)["name"], "Zone 1")

if __name__ == "__main__": unittest.main()
