#!/usr/bin/env python3
"""Make new reels (short clips) automatically.

Picks the moments people rewind to or mark (GoatCounter events rw/<key>@<sec>, mk/<key>@<sec>),
falling back to tools/reels/seeds.json. For each moment it transcribes the recitation around it,
finds which ayat are being read, cuts a clip that starts at the start of an ayah and ends at the
end of an ayah, verifies both ends by transcribing the clip again, and adds it to reels.json.

Run from the repo root:  python3 tools/reels/auto.py [--max 3] [--days 45] [--min 3] [--dry]
Needs: ffmpeg, pip install faster-whisper numpy, and quran-json (npm) — see ensure_quran().
Prints a JSON summary on the last line.
"""
import json, re, os, sys, subprocess, difflib, urllib.request, datetime, argparse, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SR = 16000
TOKEN = "1chh9xicxa1jazrv1ipo9sv6ls5t4hf9oo7dho5mnawad6yp0"
AR = lambda n: str(n).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))

ap = argparse.ArgumentParser()
ap.add_argument("--max", type=int, default=3)
ap.add_argument("--days", type=int, default=45)
ap.add_argument("--min", type=float, default=2, help="minimum score for a stats moment (rewind=1, mark=2)")
ap.add_argument("--marks-only", action="store_true", help="use only mk/ events (every mark becomes a reel candidate, no popularity ranking)")
ap.add_argument("--dry", action="store_true")
A = ap.parse_args()

def log(*a): print(*a, file=sys.stderr, flush=True)

# ---------- data ----------
def ensure_quran():
    for p in ["/tmp/qj/node_modules/quran-json/dist/quran.json", "/tmp/fnt/node_modules/quran-json/dist/quran.json"]:
        if os.path.exists(p): return json.load(open(p))
    subprocess.run(["npm", "i", "--silent", "--prefix", "/tmp/qj", "quran-json"], check=True)
    return json.load(open("/tmp/qj/node_modules/quran-json/dist/quran.json"))

Q = ensure_quran()
html = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
QD = json.loads(re.search(r'<script type="application/json" id="qdata">(.*?)</script>', html, re.S).group(1))
BY = {c["id"] + ":" + t["id"]: (c, t) for c in QD for t in c["tracks"]}
RJ = os.path.join(ROOT, "reels.json")
REELS = json.load(open(RJ, encoding="utf-8"))
SEEDS_P = os.path.join(ROOT, "tools", "reels", "seeds.json")
SEEDS = json.load(open(SEEDS_P)) if os.path.exists(SEEDS_P) else []
REJ_P = os.path.join(ROOT, "tools", "reels", "rejected.json")
REJ = json.load(open(REJ_P)) if os.path.exists(REJ_P) else []

def norm(s):
    s = re.sub(r'[ؐ-ًؚ-ٰٟۖ-ۭـ]', '', s)
    for a, b in [('ٱ','ا'),('أ','ا'),('إ','ا'),('آ','ا'),('ى','ي'),('ة','ه'),('ؤ','و'),('ئ','ي')]: s = s.replace(a, b)
    return re.sub(r'[^ء-ي ]', '', s).strip()

SURAH_BY_NAME = {norm(c["name"]): c["id"] for c in Q}

# ---------- candidates ----------
def stats_moments():
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=A.days)).strftime("%Y-%m-%dT00:00:00Z")
    ex, out = [], []
    while True:
        u = f"https://qatami.goatcounter.com/api/v0/stats/hits?limit=100&start={since}" + ("&exclude_paths=" + ",".join(map(str, ex)) if ex else "")
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"Authorization": "Bearer " + TOKEN}), timeout=60))
        for h in d["hits"]:
            ex.append(h["path_id"])
            m = re.match(r"^(rw|mk)/([^@]+)@(\d+)$", h["path"])
            if m and A.marks_only and m.group(1) != "mk": continue
            if m: out.append((m.group(2), int(m.group(3)), h["count"] * (2 if m.group(1) == "mk" else 1)))
        if not d.get("more"): break
    # merge neighbouring 10 s buckets of the same recording
    agg = {}
    for k, b, sc in out:
        agg.setdefault(k, {}).setdefault(b, 0); agg[k][b] += sc
    res = []
    for k, bs in agg.items():
        for b in sorted(bs):
            sc = sum(v for bb, v in bs.items() if abs(bb - b) <= 20)
            res.append({"key": k, "t": b, "score": sc, "from": "stats"})
    res.sort(key=lambda x: -x["score"])
    keep = []
    for r in res:
        if r["score"] < A.min: continue
        if any(r["key"] == x["key"] and abs(r["t"] - x["t"]) < 60 for x in keep): continue
        keep.append(r)
    return keep

def used(key, t):
    for r in REELS:
        if r.get("key") == key and r["off"] - 40 <= t <= r["off"] + r["dur"] + 20: return True
    return any(x["key"] == key and abs(x["t"] - t) < 40 for x in REJ)

# ---------- audio helpers ----------
def pcm(src, a, dur):
    x = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0,a):.2f}", "-t", f"{dur:.2f}", "-i", src, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, timeout=900).stdout
    return np.frombuffer(x, dtype=np.float32)

_M = None
def model():
    global _M
    if _M is None:
        from faster_whisper import WhisperModel
        _M = WhisperModel("small", device="cpu", compute_type="int8")
    return _M

def words(aud, off=0.0):
    segs, _ = model().transcribe(aud, language="ar", word_timestamps=True, vad_filter=False, beam_size=3, condition_on_previous_text=False)
    return [(w.word.strip(), round(w.start + off, 2), round(w.end + off, 2)) for s in segs for w in s.words if norm(w.word)]  # skip stray tokens like '2.' that whisper sometimes emits (they broke the first-word check)

def env_db(src, a, b):
    x = pcm(src, a, b - a); n = int(0.02 * SR); m = len(x) // n
    if m < 5: return None, None
    d = 20 * np.log10(np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(1) + 1e-12))
    return np.convolve(d, np.ones(5) / 5, "same"), a + np.arange(m) * 0.02

def snap(src, t, lo, hi, side):
    d, ts = env_db(src, t - lo - 1, t + hi + 1)
    if d is None: return t
    msk = (ts >= t - lo) & (ts <= t + hi); i = int(np.argmin(np.where(msk, d, 1e9)))
    j0 = i
    while j0 > 0 and d[j0 - 1] < d[i] + 3: j0 -= 1
    j1 = i
    while j1 < len(d) - 1 and d[j1 + 1] < d[i] + 3: j1 += 1
    return float(max(ts[j0], ts[j1] - 0.35)) if side == "start" else float(min(ts[j1], ts[j0] + 0.45))


def end_after(src, e):
    """End of the last word: whisper's end time is often early when the reciter stretches the last syllable, so go to the first real
    dip in the sound after it (up to 3 s) and finish 0.35 s into it."""
    d, ts = env_db(src, e - 0.4, e + 3.3)
    if d is not None:
        for t, x in zip(ts, d):
            if t >= e - 0.2 and x < -18: return float(min(t + 0.35, e + 3.0))
    return snap(src, e, 0.25, 1.6, "end")

def lead_trim(path, dur):
    """If the clip starts in the middle of sound (the tail of the previous ayah), return how many seconds to cut so it starts in a pause."""
    x = pcm(path, 0, min(dur, 6)); n = int(0.05 * SR); m = len(x) // n
    if m < 20: return 0.0
    d = 20 * np.log10(np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(1)) + 1e-9); med = float(np.median(d)); low = d < med - 7
    if low[:4].all(): return 0.0          # already starts in a pause
    i = 0
    while i < min(m, 70):                  # first pause (>= 0.2 s) in the first 3.5 s
        if low[i]:
            j = i
            while j < m and low[j]: j += 1
            if j - i >= 4: return round((i + j) / 2 * 0.05, 2)
            i = j
        else: i += 1
    return 0.0

# ---------- ayah matching ----------
def match(W, sur):
    V = Q[sur - 1]["verses"]; nw = [norm(w[0]) for w in W]; res = []
    for v in V:
        t = norm(v["text"]); n = len(t.split()); best = []
        if n == 0: continue
        for L in range(max(1, n - 2), n + 3):
            for i in range(0, len(W) - L + 1):
                r = difflib.SequenceMatcher(None, " ".join(nw[i:i + L]), t).ratio()
                if r >= 0.62: best.append((r, i, L))
        best.sort(reverse=True); taken = []
        for r, i, L in best:
            if all(i + L <= a or i >= a + b for _, a, b in taken): taken.append((r, i, L))
        for r, i, L in taken: res.append(dict(v=v["id"], r=r, s=W[i][1], e=W[i + L - 1][2]))
    return sorted(res, key=lambda x: x["s"])

def guess_surahs(track, t, W):
    c = []
    for it in (track.get("idx") or []):
        st, name = it[0], it[1]
        if st <= t + 30:
            nm = norm(re.sub(r"^(تتمة|خواتيم|فواتح|بداية|ما تيسر من)\s+", "", name).replace("سورة", "").strip())
            if nm in SURAH_BY_NAME: c.append(SURAH_BY_NAME[nm])
    c = c[-2:][::-1]
    # fallback / extra: the surah sharing most word-trigrams with the transcript
    tri = lambda ws: {" ".join(ws[i:i + 3]) for i in range(len(ws) - 2)}
    tw = tri([norm(w[0]) for w in W]); best = sorted(((len(tw & tri(norm(" ".join(v["text"] for v in ch["verses"])).split())), ch["id"]) for ch in Q), reverse=True)[:2]
    for _, s in best:
        if s not in c: c.append(s)
    return c

def chain(ms, t):
    """Pick consecutive ayat around moment t: start at the ayah that holds t (or starts just after), 90–130 s long (never under 1:30).
    A single ayah the transcript missed in the middle is allowed (it gets a placeholder; refine.py times its words later);
    the first and last ayah must have been heard."""
    first = {}
    for m in ms:
        if m["v"] not in first: first[m["v"]] = m
    seq = sorted(first.values(), key=lambda m: m["v"])
    starts = [m for m in seq if m["s"] <= t + 4] or seq[:1]
    if not starts: return None
    for st in sorted(starts, key=lambda m: -m["s"])[:4]:
        cur = [st]
        while True:
            v = cur[-1]["v"]; nx = first.get(v + 1)
            if nx is None:
                n2 = first.get(v + 2)
                if n2 and 3 <= n2["s"] - cur[-1]["e"] <= 24 and n2["e"] - cur[0]["s"] <= 130:
                    cur.append(dict(v=v + 1, r=0, s=cur[-1]["e"] + 0.3, e=n2["s"] - 0.3, ph=True)); nx = n2
                else: break
            if nx["s"] < cur[-1]["e"] - 1 or nx["s"] - cur[-1]["e"] > 24: break
            if nx["e"] - cur[0]["s"] > 130: break
            cur.append(nx)
            if cur[-1]["e"] - cur[0]["s"] >= 92: break
        while cur and cur[-1].get("ph"): cur.pop()
        dur = cur[-1]["e"] - cur[0]["s"]
        if 90 <= dur <= 130 and cur[0]["s"] <= t + 4: return cur
    return None

# ---------- labels ----------
def label(c, tr):
    y, ti = c.get("year"), tr.get("title", "")
    if tr.get("cat") == "ramadan" or c["id"].startswith("r"):
        p = "تراويح " if "تراويح" in ti else "قيام " if ("قيام" in ti or tr["id"].endswith("t")) else ""
        n = tr.get("night")
        return f"{p}ليلة {AR(n)} · رمضان {AR(y)}هـ" if n and y else c["title"]
    for pr in ["فجر", "ظهر", "عصر", "مغرب", "عشاء"]:
        if pr in ti or (pr == "عشاء" and "عشائية" in ti): return f"صلاة {'ال' + pr if pr != 'عشاء' else 'العشاء'} · {AR(y)}هـ" if y else pr
    return c["title"]

def new_id():
    n = 1
    ids = {r["k"] for r in REELS}
    while f"r{n:03d}" in ids: n += 1
    return f"r{n:03d}"

# ---------- build one reel ----------
def build(key, t):
    if key not in BY: return None, "unknown key"
    c, tr = BY[key]; src = tr.get("url")
    if not src or "youtube" in src: return None, "no audio url"
    a = max(0, t - 60); W = words(pcm(src, a, 290), a)
    if len(W) < 20: return None, "no speech"
    for sur in guess_surahs(tr, t, W):
        ms = match(W, sur)
        ch = chain(ms, t)
        if not ch: continue
        s0 = snap(src, ch[0]["s"], 1.6, 0.25, "start"); e0 = end_after(src, ch[-1]["e"])
        dur = round(e0 - s0, 2); k = new_id(); out = os.path.join(ROOT, "audio", "reels", k + ".mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
                        "-af", f"afade=t=in:d=0.25,afade=t=out:st={max(0,dur-0.3):.2f}:d=0.3,loudnorm=I=-16:TP=-1.5:LRA=11", "-b:a", "112k", out], check=True, timeout=900)
        lt = lead_trim(out, dur)
        if lt > 0.15:
            tmp = out + ".tmp.mp3"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{lt:.2f}", "-i", out, "-t", f"{dur-lt:.2f}", "-af", "afade=t=in:d=0.2", "-b:a", "112k", tmp], check=True)
            os.replace(tmp, out); s0 += lt; dur = round(dur - lt, 2); log(f"  trimmed {lt}s of the previous ayah's tail")
        # verify: first and last words of the clip must be the first/last words of the ayat
        CW = words(pcm(out, 0, dur + 1))
        V = Q[sur - 1]["verses"]; f3 = " ".join(norm(V[ch[0]["v"] - 1]["text"]).split()[:3]); l3 = " ".join(norm(V[ch[-1]["v"] - 1]["text"]).split()[-3:])
        cw = [norm(w[0]) for w in CW]
        okf = min(difflib.SequenceMatcher(None, " ".join(cw[:3]), f3).ratio(), difflib.SequenceMatcher(None, cw[0] if cw else "", f3.split()[0]).ratio() + 0.2); okl = difflib.SequenceMatcher(None, " ".join(cw[-3:]), l3).ratio()
        log(f"  {key}@{t} سورة {sur} {ch[0]['v']}-{ch[-1]['v']} {dur}s  start~{okf:.2f} end~{okl:.2f}")
        if okf < 0.7 or okl < 0.55:
            os.remove(out); continue
        ay = []
        for j, m in enumerate(ch):
            st = round(max(0, m["s"] - s0), 2); en = (ch[j + 1]["s"] - s0) if j + 1 < len(ch) else dur
            txt = V[m["v"] - 1]["text"]; qn = txt.split()
            ws = [w[1] for w in CW if len(w[0]) > 1 and st - 0.3 <= w[1] < en - 0.2] or [st]
            ay.append({"n": m["v"], "t": st, "text": txt, "w": [round(max(st, ws[min(len(ws) - 1, int(i * len(ws) / len(qn)))]), 2) for i in range(len(qn))]})
        rng = AR(ch[0]["v"]) if len(ch) == 1 else f"{AR(ch[0]['v'])}–{AR(ch[-1]['v'])}"
        r = {"k": k, "surah": Q[sur - 1]["name"], "range": rng, "src": label(c, tr), "dur": dur, "video": False, "ay": ay,
             "key": key, "off": round(s0, 2), "aud": f"audio/reels/{k}.mp3", "add": datetime.date.today().isoformat()}
        vu = tr.get("v")
        if vu:
            vo = os.path.join(ROOT, "audio", "reels", k + ".mp4")
            p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.2f}", "-i", vu, "-t", f"{dur:.2f}", "-an",
                                "-vf", "crop=trunc(ih*9/16/2)*2:ih:(iw-ih*9/16)/2:0,scale=720:1280:flags=lanczos,unsharp=5:5:0.6:5:5:0.0",
                                "-c:v", "libx264", "-crf", "22", "-maxrate", "1500k", "-bufsize", "3000k", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", vo], timeout=1800)
            if p.returncode == 0 and os.path.exists(vo) and os.path.getsize(vo) > 100000:
                r["video"] = True; r["file"] = f"audio/reels/{k}.mp4"
        return r, "ok"
    return None, "no clean ayah boundaries"

# ---------- main ----------
cands = [x for x in stats_moments() if not used(x["key"], x["t"])]
if len(cands) < A.max:
    cands += [dict(s, score=0, **{"from": "seed"}) for s in SEEDS if not used(s["key"], s["t"])]
log(f"candidates: {len(cands)}")
made, skipped = [], []
for cd in cands:
    if len(made) >= A.max: break
    if A.dry: made.append(cd); continue
    try: r, why = build(cd["key"], cd["t"])
    except Exception as e: r, why = None, f"error: {e}"[:200]
    if r:
        REELS.insert(0, r); made.append({"k": r["k"], "surah": r["surah"], "range": r["range"], "src": r["src"], "video": r["video"], "from": cd["from"]})
        json.dump(REELS, open(RJ, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    else:
        skipped.append({"key": cd["key"], "t": cd["t"], "why": why})
        if not why.startswith("error"): REJ.append({"key": cd["key"], "t": cd["t"]})
    if cd.get("from") == "seed" and not A.dry and (r or not why.startswith("error")):
        SEEDS = [s for s in SEEDS if not (s["key"] == cd["key"] and s["t"] == cd["t"])]
if not A.dry:
    json.dump(SEEDS, open(SEEDS_P, "w"), ensure_ascii=False, indent=0)
    json.dump(REJ[-300:], open(REJ_P, "w"), ensure_ascii=False, indent=0)
if made and not A.dry:
    ids = [m["k"] for m in made]
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "reels", "refine.py")] + ids, capture_output=True, text=True, cwd=ROOT)
    log("refine:", (p.stdout or p.stderr)[-300:])
    try: REELS = json.load(open(RJ, encoding="utf-8"))
    except Exception: pass
print(json.dumps({"made": made, "skipped": skipped, "total": len(REELS)}, ensure_ascii=False))
