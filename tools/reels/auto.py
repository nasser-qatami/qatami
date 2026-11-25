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

_MM = None
def words_m(path, a, d):
    """Second, stronger opinion (whisper medium) on a short piece of a finished clip: used to check its first and last words.
    The small model misses things at the edges: it heard no extra word where the next ayah had already begun (Naziat 35)."""
    global _MM
    if _MM is None:
        from faster_whisper import WhisperModel
        _MM = WhisperModel("medium", device="cpu", compute_type="int8")
    segs, _ = _MM.transcribe(pcm(path, a, d), language="ar", word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    return [(w.word.strip(), round(w.start + a, 2), round(w.end + a, 2)) for s in segs for w in s.words if norm(w.word)]

def edge_check(out, dur, q_first, q_last, muq, src=None, s0=0.0):
    """Check a cut clip with the medium model. Returns (ok, why, new_dur). If words of the NEXT ayah are heard after the last ayah's
    last word, new_dur ends the clip right before them."""
    sim = lambda a, b: difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
    A = words_m(out, 0, min(dur, 20))
    if not A: return False, "medium heard nothing at the start", dur
    if muq:
        if A[0][1] > 4.5: return False, f"first word late ({A[0][1]}s)", dur
    elif sim(A[0][0], q_first) < 0.6 or A[0][1] > 0.7:
        return False, f"starts with '{A[0][0]}' at {A[0][1]}s, ayah starts '{q_first}'", dur
    B = words_m(out, max(0, dur - 20), min(dur, 20) + 1)
    j = max([i for i, w in enumerate(B) if sim(w[0], q_last) >= 0.6], default=None)
    if j is None: return False, f"last ayah's last word '{q_last}' not heard at the end", dur
    if j < len(B) - 1:
        # word times are only good to ~0.1 s: end at the quietest 20 ms of the SOURCE between the two words
        lo, hi = min(B[j][2], B[j + 1][1]) - 0.3, B[j + 1][1] + 0.15
        nd = round(hi - 0.15, 2)
        if src is not None:
            d, ts = env_db(src, s0 + lo, s0 + hi)
            if d is not None: nd = round(float(ts[int(np.argmin(d))]) - s0 + 0.02, 2)
        if abs(nd - dur) < 0.06: return True, "ok (ends at the pause before the next ayah)", dur
        if nd < B[j][2] - 0.3: return False, "next ayah overlaps the last word", dur
        return True, f"next ayah heard ('{B[j + 1][0]}'): end moved to {nd}s", nd
    return True, "ok", dur

def words(aud, off=0.0):
    segs, _ = model().transcribe(aud, language="ar", word_timestamps=True, vad_filter=False, beam_size=3, condition_on_previous_text=False)
    return [(w.word.strip(), round(w.start + off, 2), round(w.end + off, 2)) for s in segs for w in s.words if norm(w.word)]  # skip stray tokens like '2.' that whisper sometimes emits (they broke the first-word check)

def env_db(src, a, b):
    # read 0.5 s earlier and drop it: the first ~0.1 s after a seek decodes as near-silence (-70 dB), a fake "pause"
    pre = min(0.5, max(0.0, a)); x = pcm(src, a - pre, b - a + pre)[int(pre * SR):]; n = int(0.02 * SR); m = len(x) // n
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

def runs_below(d, ts, thr, minlen, lo, hi):
    """(start, end) of stretches where the level stays below thr for at least minlen frames (20 ms each), inside [lo, hi]."""
    out, i, n = [], 0, len(d)
    while i < n:
        if d[i] < thr and lo <= ts[i] <= hi:
            j = i
            while j < n and d[j] < thr and ts[j] <= hi: j += 1
            if j - i >= minlen: out.append((float(ts[i]), float(ts[j - 1])))
            i = j
        else: i += 1
    return out

def end_cut(src, e, lim=None):
    """Where to end the clip after the last ayah. whisper's end of the last word is often EARLY (stretched last syllable) and the
    matcher can even drop a garbled last word, so: look from e-0.2 up to lim (start of the next ayah, if known) or e+3 s for the
    first SUSTAINED pause (>= 0.16 s below -18 dB, not a one-frame dip inside a word) and end 0.35 s into it.
    Returns None if there is no pause at all (the clip would end in the middle of sound)."""
    hi = max(lim if lim else e + 3.0, e + 0.6)
    d, ts = env_db(src, e - 0.4, hi + 0.5)
    if d is None: return None
    r = runs_below(d, ts, -18, 8, e - 0.2, hi)
    if r: return float(min(r[0][0] + 0.35, r[0][1]))
    m = (ts >= e - 0.2) & (ts <= hi); i = int(np.argmin(np.where(m, d, 1e9)))
    return float(ts[i]) if d[i] < -14 else None

def level_at(src, t):
    d, ts = env_db(src, t - 0.3, t + 0.3)
    return None if d is None else float(d[np.argmin(abs(ts - t))])

# surahs that open with separate letters (الحروف المقطعة): whisper usually does not hear them, so the start of ayah 1 must be
# taken from the pause BEFORE it, never from the first transcribed word.
MUQ = {2, 3, 7, 10, 11, 12, 13, 14, 15, 19, 20, 26, 27, 28, 29, 30, 31, 32, 36, 38, 40, 41, 42, 43, 44, 45, 46, 50, 68}

def start_before(src, s, back=7.0):
    """Start of a surah: the end of the last real pause (>= 0.25 s, 12 dB under the local median) before s."""
    d, ts = env_db(src, s - back, s + 0.2)
    if d is None: return s - 1
    r = runs_below(d, ts, float(np.median(d)) - 12, 12, s - back, s)
    return float(max(r[-1][0], r[-1][1] - 0.3)) if r else s - 1

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
    return [x for x in c if x != 1]          # the owner does not want Al-Fatiha clips

def chain(ms, t, sur=None):
    """Pick consecutive ayat around moment t: start at the ayah that holds t (or starts just after), 90–130 s long (never under 1:30).
    Each next ayah is the occurrence that follows the previous one in time (the transcript can match an ayah in several places:
    a repeated phrase, or a weak match far away), not simply the earliest one.
    A single ayah the transcript missed in the middle is allowed (placeholder; refine.py times its words later);
    the first and last ayah must have been heard."""
    occ = {}
    for m in ms: occ.setdefault(m["v"], []).append(m)
    def after(v, e):
        c = [m for m in occ.get(v, []) if e - 1 <= m["s"] <= e + 24]
        return min(c, key=lambda m: (m["s"], -m["r"])) if c else None
    starts = [m for m in ms if m["r"] >= 0.75 and m["s"] <= t + 4]
    for st in sorted(starts, key=lambda m: -m["s"])[:6]:
        cur = [st]
        while True:
            v = cur[-1]["v"]; nx = after(v + 1, cur[-1]["e"])
            if nx is None:
                n2 = after(v + 2, cur[-1]["e"])
                if n2 and 3 <= n2["s"] - cur[-1]["e"] <= 24 and n2["e"] - cur[0]["s"] <= 130:
                    cur.append(dict(v=v + 1, r=0, s=cur[-1]["e"] + 0.3, e=n2["s"] - 0.3, ph=True)); nx = n2
                else: break
            if nx["e"] - cur[0]["s"] > 130: break
            cur.append(nx)
            if cur[-1]["e"] - cur[0]["s"] >= 92: break
        while cur and cur[-1].get("ph"): cur.pop()
        # owner's rule: if less than 40 s of the surah remain after the clip, carry on to the END of the surah (up to 170 s)
        if sur and cur and cur[-1]["e"] - cur[0]["s"] >= 90:
            N = len(Q[sur - 1]["verses"]); ext = list(cur)
            while ext[-1]["v"] < N:
                nx = after(ext[-1]["v"] + 1, ext[-1]["e"])
                if nx is None or nx["e"] - ext[0]["s"] > 170: break
                ext.append(nx)
            if ext[-1]["v"] == N and ext[-1]["e"] - cur[-1]["e"] < 40: cur = ext
        dur = cur[-1]["e"] - cur[0]["s"]
        if 90 <= dur <= 170 and cur[0]["s"] <= t + 4: return cur
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
    # a moment inside Al-Fatiha (start of a prayer): take the surah read right after it instead (no Fatiha clips)
    fat = [m for m in match(W, 1) if m["v"] >= 5 and m["r"] >= 0.8]
    if fat and min(m["s"] for m in fat) - 6 <= t <= max(m["e"] for m in fat) + 4:
        t = max(m["e"] for m in fat) + 12; log(f"  moment is in Al-Fatiha -> using the surah after it (t={t:.0f})")
    for sur in guess_surahs(tr, t, W):
        ms = match(W, sur)
        ch = chain(ms, t, sur)
        if not ch: continue
        if ch[0]["v"] == 1 and sur in MUQ: s0 = start_before(src, ch[0]["s"])
        else: s0 = snap(src, ch[0]["s"], 1.6, 0.25, "start")
        nxt = [m["s"] for m in ms if m["v"] == ch[-1]["v"] + 1 and m["s"] > ch[-1]["s"]]
        e0 = end_cut(src, ch[-1]["e"], (min(nxt) + 0.8) if nxt else None)
        if e0 is None: log(f"  {key}@{t}: no pause after ayah {ch[-1]['v']} (would end mid-sound)"); continue
        dur = round(e0 - s0, 2); k = new_id(); out = os.path.join(ROOT, "audio", "reels", k + ".mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
                        "-af", f"afade=t=in:d=0.25,afade=t=out:st={max(0,dur-0.3):.2f}:d=0.3,loudnorm=I=-16:TP=-1.5:LRA=11", "-b:a", "112k", out], check=True, timeout=900)
        lt = 0.0 if (ch[0]["v"] == 1 and sur in MUQ) else lead_trim(out, dur)
        if lt > 0.15:
            tmp = out + ".tmp.mp3"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{lt:.2f}", "-i", out, "-t", f"{dur-lt:.2f}", "-af", "afade=t=in:d=0.2", "-b:a", "112k", tmp], check=True)
            os.replace(tmp, out); s0 += lt; dur = round(dur - lt, 2); log(f"  trimmed {lt}s of the previous ayah's tail")
        # verify: first and last words of the clip must be the first/last words of the ayat
        CW = words(pcm(out, 0, dur + 1))
        # whisper places word starts up to ~1 s early, so the cut can still hold the tail of the previous ayah before the first word:
        # if the first heard word starts late, move the start to the last pause before it (not for separate-letter openings).
        qf = norm(V0[ch[0]["v"] - 1]["text"].split()[0]) if (V0 := Q[sur - 1]["verses"]) else ""
        fi = next((i for i, w in enumerate(CW[:4]) if difflib.SequenceMatcher(None, norm(w[0]), qf).ratio() >= 0.6), None)
        ws = CW[fi][1] if fi is not None else (CW[0][1] if CW else 0)
        glued = fi is not None and difflib.SequenceMatcher(None, norm(CW[fi][0]), qf).ratio() < 0.85   # e.g. "سيقال" = tail + "قال"
        if glued and fi + 1 < len(CW): ws = CW[fi + 1][1]
        log(f"  start words: {[(w[0], w[1]) for w in CW[:3]]} first-match={fi} glued={glued} ws={ws}")
        # also when a junk "word" (the previous ayah's tail) comes before the ayah's first word, or is glued to it
        if CW and (ws > 0.9 or (fi or 0) > 0 or glued) and not (ch[0]["v"] == 1 and sur in MUQ):
            d, ts = env_db(src, s0, s0 + ws + 0.3)
            r = runs_below(d, ts, float(np.percentile(d, 75)) - 8, 6, s0 + 0.2, s0 + ws + 0.3) if d is not None else []   # 8 dB under the loud part
            if r:   # the deepest pause there is the one between the ayat (gaps between words are shallower)
                lv = lambda q: float(np.mean(d[(ts >= q[0]) & (ts <= q[1])]))
                best = min(r, key=lv); ns = max(best[0], best[1] - 0.3); dur = round(dur - (ns - s0), 2); s0 = ns
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
                                "-af", f"afade=t=in:d=0.25,afade=t=out:st={max(0,dur-0.3):.2f}:d=0.3,loudnorm=I=-16:TP=-1.5:LRA=11", "-b:a", "112k", out], check=True, timeout=900)
                CW = words(pcm(out, 0, dur + 1)); log(f"  moved start to the pause before the first word ({s0:.2f})")
        lv = level_at(src, s0 + dur)
        if lv is not None and lv > -14: log(f"  {key}@{t}: cut is in loud sound ({lv:.0f} dB)"); os.remove(out); continue
        V = Q[sur - 1]["verses"]
        ok, why, nd = edge_check(out, dur, V[ch[0]["v"] - 1]["text"].split()[0], V[ch[-1]["v"] - 1]["text"].split()[-1], ch[0]["v"] == 1 and sur in MUQ, src, s0)
        log(f"  medium check: {why}")
        if not ok: os.remove(out); continue
        if nd < dur:
            dur = nd
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.2f}", "-t", f"{dur:.2f}", "-i", src, "-vn", "-ac", "2", "-ar", "44100",
                            "-af", f"afade=t=in:d=0.25,afade=t=out:st={max(0,dur-0.15):.2f}:d=0.15,loudnorm=I=-16:TP=-1.5:LRA=11", "-b:a", "112k", out], check=True, timeout=900)
            CW = words(pcm(out, 0, dur + 1))
        V = Q[sur - 1]["verses"]; f3 = " ".join(norm(V[ch[0]["v"] - 1]["text"]).split()[:3]); l3 = " ".join(norm(V[ch[-1]["v"] - 1]["text"]).split()[-3:])
        cw = [norm(w[0]) for w in CW]
        okf = min(difflib.SequenceMatcher(None, " ".join(cw[:3]), f3).ratio(), difflib.SequenceMatcher(None, cw[0] if cw else "", f3.split()[0]).ratio() + 0.2); okl = min(difflib.SequenceMatcher(None, " ".join(cw[-3:]), l3).ratio(), difflib.SequenceMatcher(None, cw[-1] if cw else "", l3.split()[-1]).ratio() + 0.25)
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
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "reels", "tafsir_fill.py")], capture_output=True, text=True, cwd=ROOT)
    log("tafsir:", (p.stdout or p.stderr)[-300:])
    try: REELS = json.load(open(RJ, encoding="utf-8"))
    except Exception: pass
print(json.dumps({"made": made, "skipped": skipped, "total": len(REELS)}, ensure_ascii=False))
