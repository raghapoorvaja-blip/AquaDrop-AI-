"""FastAPI app. Run:  uvicorn app.main:app --reload   (from the backend folder)"""
import io, os
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from . import allocation, db, ml, weather
from .ml import ROOT
from .simulation import Sim

try:
    from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

app = FastAPI(title="AQUADROP AI (prototype)")
sim = Sim()
S = {"alloc": None, "preds": {}, "weights": {}, "dataset": None, "speed": 1.0}
UPLOADS = os.path.join(ROOT, "datasets", "uploads")
FRONT = os.path.join(ROOT, "frontend")

# ---- persistence: latest state lives in SQLite (kv table); history/events go to the events table ----
def persist():
    db.set_kv("sim_state", sim.snapshot())
    db.set_kv("app_state", S)

def restore():
    saved = db.get_kv("sim_state")
    if saved: sim.restore(saved)
    S.update({k: v for k, v in (db.get_kv("app_state") or {}).items() if k in S})

restore()
sim.sink = db.log   # from now on sessions, sensor readings and alerts are also written to the events table
persist()

def run_predictions():
    w = weather.current()
    S["preds"] = {z["id"]: ml.predict(z, w) for z in sim.zones}
    for zid, p in S["preds"].items(): db.log("prediction", {"zone_id": zid, **p})
    return S["preds"]

def run_allocation():
    w = weather.current()
    factor = 0.5 if w.get("rain_mm", 0) >= 2 else 1.0   # simple weather rule: expected rain halves demand
    S["alloc"] = allocation.allocate(sim.zones, run_predictions(), sim.tank, S["weights"], factor)
    db.log("allocation", S["alloc"]); persist()
    return S["alloc"]

@app.get("/api/state")
def state():
    return {"zones": sim.zones, "tank_l": round(sim.tank, 1), "capacity_l": sim.capacity, "pump": sim.pump, "mode": sim.mode,
            "sim_minute": round(sim.t, 1), "alerts": sim.alerts[-50:], "sessions": sim.sessions[-20:], "history": sim.history,
            "model": ml.status(), "alloc": S["alloc"], "preds": S["preds"], "weather": weather.current(), "cfg": sim.cfg,
            "weights": S["weights"], "dataset": S["dataset"], "speed": S["speed"], "label": "Zone values below are SIMULATED"}

@app.put("/api/zones/{zid}")
def edit_zone(zid: int, body: dict = Body(...)):
    try: z = sim.zone(zid)
    except StopIteration: raise HTTPException(404, "No such zone")
    for k in ("name", "crop", "soil"):
        if k in body: z[k] = str(body[k])
    for k in ("area", "threshold", "target", "root_depth_mm", "sensitivity", "moisture"):
        if k in body:
            try: z[k] = float(body[k])
            except ValueError: raise HTTPException(400, f"{k} must be a number")
    if z["area"] <= 0 or not 0 <= z["threshold"] < z["target"] <= 100: raise HTTPException(400, "Need area>0 and 0 <= threshold < target <= 100")
    persist(); return z

@app.post("/api/allocate")
def allocate_ep(body: dict = Body(default={})):
    if body.get("weights"): S["weights"] = {k: float(v) for k, v in body["weights"].items()}
    return run_allocation()

@app.post("/api/predict")
def predict_ep(): return run_predictions()

@app.post("/api/train")
def train_ep(body: dict = Body(default={})):
    try:
        if body.get("use_uploaded") and S["dataset"]:
            df = pd.read_csv(S["dataset"]["path"])
            meta = ml.train(df, body["feature_map"], body["target"], source=f"uploaded file: {S['dataset']['name']}")
        else: meta = ml.train()
    except (ValueError, KeyError) as e: raise HTTPException(400, f"Training failed: {e}")
    db.log("model", meta); return meta

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400, "Please upload a .csv file")
    try: df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as e: raise HTTPException(400, f"Could not read CSV: {e}")
    if df.empty: raise HTTPException(400, "CSV is empty")
    os.makedirs(UPLOADS, exist_ok=True)
    path = os.path.join(UPLOADS, os.path.basename(file.filename)); df.to_csv(path, index=False)
    S["dataset"] = {"name": file.filename, "path": path, "rows": len(df), "columns": list(df.columns), "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
                    "preview": df.head(8).fillna("").astype(str).to_dict("records"), "origin": "USER-UPLOADED data"}
    db.log("dataset", {k: v for k, v in S["dataset"].items() if k != "preview"}); persist(); return S["dataset"]

@app.post("/api/irrigation/start")
def start(body: dict = Body(default={})):
    if "zone_id" in body:
        if not sim.start_zone(int(body["zone_id"]), float(body.get("litres", 100))): raise HTTPException(400, "Cannot start: no water, or zone already irrigating")
        persist(); return {"started": [int(body["zone_id"])]}
    started = sim.start_allocation(S["alloc"] or run_allocation())
    if not started: raise HTTPException(400, "Nothing to irrigate (no allocation).")
    persist(); return {"started": started}

@app.post("/api/irrigation/stop")
def stop(): sim.stop_all(); persist(); return {"stopped": True}

@app.post("/api/mode")
def mode(body: dict = Body(...)):
    if body.get("mode") not in ("auto", "manual"): raise HTTPException(400, "mode must be auto or manual")
    sim.mode = body["mode"]; persist(); return {"mode": sim.mode}

@app.post("/api/tick")
def tick(body: dict = Body(default={})):
    sim.tick(float(body.get("minutes", S["speed"])))
    if sim.mode == "auto" and not sim.active() and any(z["moisture"] < z["threshold"] and z["status"] == "idle" for z in sim.zones):
        sim.start_allocation(run_allocation())   # auto mode: predict -> prioritise -> allocate -> irrigate
    persist(); return {"ok": True}

@app.post("/api/fault")
def inject(body: dict = Body(...)):
    """Simulation only: inject a fault so the detectors can be demonstrated."""
    z = sim.zone(int(body["zone_id"])); z[body["kind"]] = body["value"]; persist(); return z

@app.post("/api/settings")
def settings(body: dict = Body(...)):
    for k in sim.cfg:
        if k in body: sim.cfg[k] = max(0.01, float(body[k]))
    if "speed" in body: S["speed"] = max(0.1, float(body["speed"]))
    if "tank_l" in body: sim.tank = min(sim.capacity, max(0.0, float(body["tank_l"])))
    persist(); return sim.cfg

@app.post("/api/weather")
def manual_weather(body: dict = Body(...)): weather.set_manual(body); return weather.current()

@app.post("/api/alerts/{aid}/resolve")
def resolve(aid: int):
    for a in sim.alerts:
        if a["id"] == aid and a["status"] != "resolved":
            a["status"] = "resolved"; db.log("alert_resolved", {"id": aid, "type": a["type"], "zone_id": a["zone_id"]})
    persist(); return {"ok": True}

@app.get("/api/history/{kind}")
def hist(kind: str): return db.history(kind)

@app.post("/api/reset")
def reset(): sim.reset(); S.update(alloc=None, preds={}); persist(); return {"reset": True}

@app.get("/")
def index(): return FileResponse(os.path.join(FRONT, "index.html"))
app.mount("/static", StaticFiles(directory=FRONT), name="static")
