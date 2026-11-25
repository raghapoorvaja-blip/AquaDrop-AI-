"""Priority-based water allocation. Plain algorithm (NOT AI), separate from the ML model."""
WETTED_FRACTION = 0.5  # share of zone area wetted by drip lines (configurable assumption)

def litres_per_pct(z):
    """Litres needed to raise soil moisture by 1 percentage point (simulation assumption)."""
    return z["area"] * z["root_depth_mm"] * WETTED_FRACTION / 100.0

def allocate(zones, preds, available_l, weights=None, demand_factor=1.0):
    w = {"deficit": 0.5, "sensitivity": 0.3, "ml": 0.2, **(weights or {})}
    rows = []
    for z in zones:
        p = preds.get(z["id"], {})
        need = bool(p.get("irrigate"))
        deficit = max(0.0, z["target"] - z["moisture"])
        demand = deficit * litres_per_pct(z) * demand_factor if need else 0.0
        prob = p.get("probability")
        score = w["deficit"] * min(deficit / 40, 1) + w["sensitivity"] * z["sensitivity"] + w["ml"] * (prob if prob is not None else 1.0)
        rows.append({"zone_id": z["id"], "name": z["name"], "needs_irrigation": need, "demand_l": round(demand, 1),
                     "priority_score": round(score, 3) if need else 0.0, "allocated_l": 0.0, "unmet_l": 0.0, "reason": "", "rank": None})
    remaining = max(0.0, float(available_l))
    ranked = sorted([r for r in rows if r["demand_l"] > 0], key=lambda r: -r["priority_score"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
        give = min(r["demand_l"], remaining)
        r["allocated_l"] = round(give, 1); remaining -= give
        r["unmet_l"] = round(r["demand_l"] - give, 1)
        r["reason"] = (f"Rank {i} (score {r['priority_score']}): soil deficit, crop sensitivity and prediction. " +
                       ("Fully supplied." if r["unmet_l"] == 0 else f"Water ran short: {r['unmet_l']} L unmet."))
    for r in rows:
        if r["demand_l"] == 0: r["reason"] = "No irrigation needed (prediction says moisture is adequate)."
    return {"zones": rows, "available_l": round(available_l, 1), "total_demand_l": round(sum(r["demand_l"] for r in rows), 1),
            "total_allocated_l": round(sum(r["allocated_l"] for r in rows), 1), "total_unmet_l": round(sum(r["unmet_l"] for r in rows), 1),
            "weights": w, "source": "Algorithmically calculated (rule-based, not AI)"}
