"""Weather: OpenWeather if key + internet exist, otherwise labelled demo/manual values. Key stays server-side."""
import json, os, urllib.request
DEMO = {"temperature": 32.0, "humidity": 45.0, "rain_mm": 0.0, "conditions": "clear", "source": "DEMO fallback (not real weather)"}
_manual = {}

def set_manual(d):
    _manual.update({k: float(d[k]) for k in ("temperature", "humidity", "rain_mm") if k in d}); _manual["source"] = "manual input"

def current():
    key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if key:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={os.environ.get('WEATHER_CITY','Tirunelveli,IN')}&units=metric&appid={key}"
            j = json.load(urllib.request.urlopen(url, timeout=4))
            return {"temperature": j["main"]["temp"], "humidity": j["main"]["humidity"], "rain_mm": j.get("rain", {}).get("1h", 0.0),
                    "conditions": j["weather"][0]["description"], "source": "OpenWeather API"}
        except Exception as e:
            return {**DEMO, **_manual, "source": f"DEMO fallback (API failed: {type(e).__name__})"} if _manual else {**DEMO, "source": f"DEMO fallback (API failed: {type(e).__name__})"}
    return {**DEMO, **_manual}
