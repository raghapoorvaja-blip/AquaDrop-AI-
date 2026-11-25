"""SQLite storage (standard library). One 'events' table holds history of every kind, tagged by 'kind'.
The 'kv' table holds the latest saved state (zones, tank, settings...) so it can be restored on restart."""
import json, os, sqlite3, time
from contextlib import closing
from .ml import ROOT
DB_PATH = os.environ.get("AQUADROP_DB", os.path.join(ROOT, "aquadrop.db"))
KEEP = {"sensor": 5000}   # kinds that grow every tick: keep only the newest N rows

def _c():
    c = sqlite3.connect(DB_PATH)
    c.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, kind TEXT, ts TEXT, data TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT)")
    return c

def log(kind, data):
    with closing(_c()) as c, c:
        c.execute("INSERT INTO events(kind,ts,data) VALUES(?,?,?)", (kind, time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(data)))
        if kind in KEEP:
            c.execute("DELETE FROM events WHERE kind=? AND id NOT IN (SELECT id FROM events WHERE kind=? ORDER BY id DESC LIMIT ?)", (kind, kind, KEEP[kind]))

def history(kind, limit=50):
    with closing(_c()) as c: rows = c.execute("SELECT ts,data FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)).fetchall()
    return [{"ts": t, **json.loads(d)} for t, d in rows]

def set_kv(k, v):
    with closing(_c()) as c, c: c.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v)))

def get_kv(k, default=None):
    with closing(_c()) as c: r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default
