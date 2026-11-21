#!/usr/bin/env python3
"""Check and refine reels that are already in reels.json.

For every clip (or the ids given): transcribe the clip itself, align the Quran words to the recited words
(sequence alignment, not by position), then
  * set exact start times for every word and for every ayah,
  * check that the first spoken word is the first word of the first ayah and the last is the last word of the last ayah,
  * trim silence at the head/tail (keeps ~0.35 s head, ~0.8 s tail) and a leftover tail of the previous ayah,
  * report anything that still looks wrong.
Run from the repo root:  python3 tools/reels/refine.py [r001 r002 ...] [--report]
Prints one JSON line at the end: {"ok": [...], "fixed": [...], "bad": [...]}.
"""
import json, os, re, sys, subprocess, difflib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SR = 16000
ids = [a for a in sys.argv[1:] if not a.startswith("--")]
REPORT = "--report" in sys.argv

def norm(s):
    s = re.sub(r'[ؐ-ًؚ-ٰٟۖ-ۭـ]', '', s)
    for a, b in [('ٱ','ا'),('أ','ا'),('إ','ا'),('آ','ا'),('ى','ي'),('ة','ه'),('ؤ','و'),('ئ','ي')]: s = s.replace(a, b)
    return re.sub(r'[^ء-ي ]', '', s).strip()

_M = None
def model():
    global _M
    if _M is None:
        from faster_whisper import WhisperModel
        _M = WhisperModel("small", device="cpu", compute_type="int8")
    return _M

def pcm(path, a=0, dur=None):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{a:.2f}", "-i", path] + (["-t", f"{dur:.2f}"] if dur else []) + ["-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, timeout=600).stdout, dtype=np.float32)

def words(path):
    segs, _ = model().transcribe(pcm(path), language="ar", word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in s.words if w.word.strip()]

def align(Qw, Ww):
    """Qw: normalised quran words; Ww: [(text,start,end)] -> start time per quran word (None if unknown) and a 'matched' flag."""
    wn = [norm(w[0]) for w in Ww]
    sm = difflib.SequenceMatcher(None, Qw, wn, autojunk=False)
    st = [None] * len(Qw); mt = [False] * len(Qw)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1): st[i1 + k] = Ww[j1 + k][1]; mt[i1 + k] = True
        elif tag == "replace":
            n, m = i2 - i1, j2 - j1
            for k in range(n):
                jj = j1 + min(m - 1, int(k * m / n)); st[i1 + k] = Ww[jj][1]
                mt[i1 + k] = difflib.SequenceMatcher(None, Qw[i1 + k], wn[jj]).ratio() >= 0.6
    # fill the gaps by interpolation between known neighbours
    known = [i for i, v in enumerate(st) if v is not None]
    if not known: return st, mt
    for i in range(len(st)):
        if st[i] is None:
            lo = max([k for k in known if k < i], default=None); hi = min([k for k in known if k > i], default=None)
            if lo is None: st[i] = st[hi] - (hi - i) * 0.35
            elif hi is None: st[i] = st[lo] + (i - lo) * 0.35
            else: st[i] = st[lo] + (st[hi] - st[lo]) * (i - lo) / (hi - lo)
    return st, mt

def recut(path, a, dur, video=None):
    tmp = path + ".tmp.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-i", path, "-t", f"{dur:.2f}", "-af",
                    f"afade=t=in:d=0.2,afade=t=out:st={max(0, dur - 0.5):.2f}:d=0.5", "-b:a", "112k", tmp], check=True)
    os.replace(tmp, path)
    if video and os.path.exists(video):
        tv = video + ".tmp.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.2f}", "-i", video, "-t", f"{dur:.2f}", "-c:v", "libx264", "-crf", "22", "-maxrate", "1500k",
                        "-bufsize", "3000k", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", tv], check=True)
        os.replace(tv, video)

RJ = os.path.join(ROOT, "reels.json")
R = json.load(open(RJ, encoding="utf-8"))
out = {"ok": [], "fixed": [], "bad": []}
for r in R:
    if ids and r["k"] not in ids: continue
    path = os.path.join(ROOT, r["aud"]); vid = os.path.join(ROOT, r["file"]) if r.get("video") and r.get("file") else None
    notes = []
    for rnd in range(2):
        W = words(path)
        if len(W) < 5: notes.append("no speech"); break
        Qw, owner = [], []
        for j, a in enumerate(r["ay"]):
            for w in a["text"].split(): Qw.append(norm(w)); owner.append(j)
        st, mt = align(Qw, W)
        jq = "".join(Qw); jw = "".join(norm(w[0]) for w in W)
        first_ok = difflib.SequenceMatcher(None, jq[:12], jw[:12]).ratio() >= 0.75
        last_ok = difflib.SequenceMatcher(None, jq[-12:], jw[-12:]).ratio() >= 0.75
        lead = W[0][1]; tail = r["dur"] - W[-1][2]      # first SPOKEN word, never an interpolated one
        # trim head / tail silence (only when clearly long)
        a0, d1 = 0.0, r["dur"]
        if lead > 1.2 and first_ok: a0 = lead - 0.5
        if tail > 1.8 and last_ok: d1 = W[-1][2] + 0.9
        if (a0 > 0 or d1 < r["dur"] - 0.05) and rnd == 0:
            recut(path, a0, d1 - a0, vid); r["off"] = round(r["off"] + a0, 2); r["dur"] = round(d1 - a0, 2); notes.append(f"trimmed head {a0:.1f}s tail {r['dur'] + a0 - d1 + (d1 - a0 - r['dur']):.1f}")
            continue
        break
    else:
        pass
    if len(W) >= 5:
        k = 0
        for j, a in enumerate(r["ay"]):
            n = len(a["text"].split()); a["w"] = [round(max(0, st[k + i]), 2) for i in range(n)]
            a["t"] = 0.0 if j == 0 else round(max(0, a["w"][0] - 0.1), 2); k += n
        # words that never got a spoken match are a warning
        miss = sum(1 for m in mt if not m)
        if miss > 0.25 * len(mt): notes.append(f"{miss}/{len(mt)} words unmatched")
        if not first_ok: notes.append("first word not the ayah's first word")
        if not last_ok: notes.append("last word not the ayah's last word")
    (out["bad"] if any(("not the" in n or "unmatched" in n or n == "no speech") for n in notes) else out["fixed"] if notes else out["ok"]).append({"k": r["k"], "notes": notes} if notes else r["k"])
    if REPORT: print(r["k"], notes, file=sys.stderr)
json.dump(R, open(RJ, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(json.dumps(out, ensure_ascii=False))
