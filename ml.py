"""ML pipeline: train / evaluate / save / load / predict. Falls back to a labelled rule when no model exists."""
import json, os
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.environ.get("AQUADROP_MODEL_DIR", os.path.join(ROOT, "models"))
FEATURES = ["moisture", "temperature", "humidity"]

def synthetic_data(n=1500, seed=42):
    """SYNTHETIC training data (not real). Label = hand-written rule + 5% noise, so metrics only show the pipeline works."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({"moisture": r.uniform(10, 80, n), "temperature": r.uniform(15, 42, n), "humidity": r.uniform(20, 95, n)})
    y = ((df.moisture < 32) | ((df.moisture < 42) & (df.temperature > 33) & (df.humidity < 45))).astype(int)
    flip = r.random(n) < 0.05
    df["irrigate"] = np.where(flip, 1 - y, y)
    return df

def _binary(s):
    s = s.dropna()
    if pd.api.types.is_numeric_dtype(s):
        vals = sorted(s.unique())
        if len(vals) != 2: raise ValueError(f"Target must have exactly 2 values (found {len(vals)}).")
        return (s == vals[1]).astype(int)
    low = s.astype(str).str.strip().str.lower()
    vals = sorted(low.unique())
    if len(vals) != 2: raise ValueError(f"Target must have exactly 2 values (found {len(vals)}).")
    pos = next((v for v in vals if v in ("on", "yes", "true", "1")), vals[1])
    return (low == pos).astype(int)

def train(df=None, feature_map=None, target=None, source="synthetic (generated, NOT real data)"):
    if df is None:
        df, feature_map, target = synthetic_data(), {f: f for f in FEATURES}, "irrigate"
    if not feature_map or target not in df.columns: raise ValueError("Choose a target column and at least one feature column.")
    bad = [c for c in feature_map.values() if c not in df.columns]
    if bad: raise ValueError(f"Columns not found: {bad}")
    X = pd.DataFrame({k: pd.to_numeric(df[c], errors="coerce") for k, c in feature_map.items()})
    y = _binary(df[target]); X = X.loc[y.index]
    if len(y) < 30: raise ValueError("Need at least 30 labelled rows to train.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # split BEFORE fitting (no leakage)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", RandomForestClassifier(200, random_state=42, class_weight="balanced"))])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    meta = {"model": "RandomForestClassifier", "features": list(X.columns), "target": target, "source": source, "rows": int(len(y)),
            "metrics": {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
                        "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0),
                        "confusion_matrix": confusion_matrix(yte, pred).tolist(), "test_rows": int(len(yte))},
            "note": "Metrics are on a held-out 20% test split of THIS dataset only; not field validation."}
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, os.path.join(MODEL_DIR, "model.joblib"))
    json.dump(meta, open(os.path.join(MODEL_DIR, "meta.json"), "w"), indent=1)
    return meta

def status():
    ok = os.path.exists(os.path.join(MODEL_DIR, "meta.json")) and os.path.exists(os.path.join(MODEL_DIR, "model.joblib"))
    return json.load(open(os.path.join(MODEL_DIR, "meta.json"))) if ok else None

def predict(zone, weather):
    meta = status()
    if meta is None:
        return {"irrigate": zone["moisture"] < zone["threshold"], "probability": None, "source": "rule_based_fallback (NOT an ML prediction)"}
    pipe = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    vals = {"moisture": zone["moisture"], "temperature": weather["temperature"], "humidity": weather["humidity"]}
    p = float(pipe.predict_proba(pd.DataFrame([{f: vals[f] for f in meta["features"]}]))[0][1])
    return {"irrigate": p >= 0.5, "probability": round(p, 3), "source": f"ml ({meta['source']})"}
