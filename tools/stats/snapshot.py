#!/usr/bin/env python3
"""Saves every FINISHED day of GoatCounter hits into stats-days.json so the stats page opens fast.
The page keys each day by the device's own day boundaries; this file uses Riyadh (UTC+3) boundaries,
so it matches phones in Riyadh and other devices simply fall back to live requests.
Run from the repo root:  python3 tools/stats/snapshot.py"""
import json, re, time, urllib.request, datetime, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
html = open(os.path.join(ROOT, "stats.html"), encoding="utf-8").read()
TOKEN = re.search(r'TOKEN = "([^"]+)"', html).group(1); LAUNCH = re.search(r'LAUNCH = "([^"]+)"', html).group(1)
API = "https://qatami.goatcounter.com/api/v0/stats/hits?"
OUT = os.path.join(ROOT, "stats-days.json")
last = 0
def api(q, tries=5):
    global last
    for i in range(tries):
        wait = last + 0.27 - time.time()
        if wait > 0: time.sleep(wait)
        last = time.time()
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(API + q, headers={"Authorization": "Bearer " + TOKEN}), timeout=60))
        except Exception as e:
            time.sleep(1.2 * (i + 1))
    raise SystemExit("api failed: " + q[:120])
def page_hits(q):
    """Same paging as the page itself (stats.html pageHits): excluded ids go in ONE comma-separated parameter."""
    out, seen = [], set()
    for _ in range(30):
        d = api("limit=100&" + q + ("&exclude_paths=" + ",".join(str(h["path_id"]) for h in out) if out else ""))
        new = [h for h in (d.get("hits") or []) if h["path_id"] not in seen]
        for h in new: seen.add(h["path_id"])
        out += new
        if not d.get("more") or not new: break
    return out
def slim(h):
    return {k: h[k] for k in ("path_id", "path", "title", "event", "count", "stats") if k in h}
now = datetime.datetime.now(datetime.timezone.utc)
old = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
res = dict(old)
d = datetime.date.fromisoformat(LAUNCH)
Z = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
while True:
    s = datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone(datetime.timedelta(hours=3)))
    e = s + datetime.timedelta(hours=23, minutes=59, seconds=59)
    cs, ce = Z(s.astimezone(datetime.timezone.utc)), Z(e.astimezone(datetime.timezone.utc))
    if e.astimezone(datetime.timezone.utc) > now - datetime.timedelta(hours=2): break
    key = cs + ce
    if key not in res:
        q = "start=%s&end=%s" % (cs, ce)
        hits = page_hits(q)
        tot = json.load(urllib.request.urlopen(urllib.request.Request(API.replace("stats/hits?", "stats/total?") + q, headers={"Authorization": "Bearer " + TOKEN}), timeout=60))["total"]
        got = sum(h.get("count", 0) for h in hits)
        if got != tot or len({h["path_id"] for h in hits}) != len(hits):
            raise SystemExit("%s: paths add up to %d but GoatCounter's total is %d, not saved" % (d, got, tot))
        res[key] = [slim(h) for h in hits]
        print(d, len(res[key]), flush=True)
    d += datetime.timedelta(days=1)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("days", len(res), "bytes", os.path.getsize(OUT))
