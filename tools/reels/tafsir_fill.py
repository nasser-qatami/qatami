#!/usr/bin/env python3
"""Fill the field "tf" (التفسير الميسر) for every ayah in reels.json that has none. Text is COPIED from a published source, never written.

Source 1 (newer revised edition): https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/ar-tafsir-muyassar/<surah>/<ayah>.json  (field "text")
Source 2 (only when source 1 starts with «تسمية السورة», i.e. a surah introduction): https://api.quran.com/api/v4/tafsirs/16/by_ayah/<s>:<a>
If source 1 cannot be fetched after retries, the ayah is LEFT EMPTY and reported (never silently replaced by source 2).
Allowed edits only: «ﷺ» -> «صلى الله عليه وسلم» (the site font cannot draw it), Arabic presentation-form letters -> normal letters (NFKC,
only when such letters occur), and removing HTML tags that the sources carry (e.g. <span class="blue">…</span> in Taha 97;
the page shows the text as plain text). Wording is never changed.
Run from the repo root:  python3 tools/reels/tafsir_fill.py   then   python3 tools/reels/tafsir_verify.py
"""
import json, re, subprocess, unicodedata, time, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def n2(x):
    x = unicodedata.normalize('NFKC', x)
    return re.sub(r'[^ء-ي ]', '', re.sub('[ً-ٰٟـۖ-ۭ]', '', x)).replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ٱ','ا')

def curl(u, tries=8):
    for i in range(tries):
        r = subprocess.run(["curl", "-s", "-f", "-m", "25", u], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip(): return r.stdout
        time.sleep(2)
    return ""

def clean(t):
    t = t.replace("ﷺ", "صلى الله عليه وسلم")
    if re.search("[ﭐ-﷿ﹰ-﻿]", t): t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"</?[a-zA-Z][^>]*>", "", t)          # markup only
    return t.strip()

ch = json.loads(curl("https://api.quran.com/api/v4/chapters?language=ar"))["chapters"]
num = {n2(c["name_arabic"]): c["id"] for c in ch}
RJ = os.path.join(ROOT, "reels.json")
D = json.load(open(RJ, encoding="utf-8")); miss = []
for r in D:
    s = num.get(n2(r["surah"]))
    for a in r["ay"]:
        if a.get("tf"):
            if "<" in a["tf"]: a["tf"] = clean(a["tf"])
            continue
        t = None
        j = curl(f"https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/ar-tafsir-muyassar/{s}/{a['n']}.json")
        if not j: miss.append((r["k"], r["surah"], a["n"], "source 1 could not be fetched")); continue
        t = json.loads(j)["text"]
        if t.strip().startswith("تسمية السورة"):
            j = curl(f"https://api.quran.com/api/v4/tafsirs/16/by_ayah/{s}:{a['n']}")
            t = json.loads(j)["tafsir"]["text"] if j else None
            if not t or t.strip().startswith("تسمية السورة"): miss.append((r["k"], r["surah"], a["n"], "surah intro; source 2 failed")); continue
        a["tf"] = clean(t); print(r["k"], r["surah"], s, a["n"], a["tf"][:60])
json.dump(D, open(RJ, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("left without tafsir:", miss)
sys.exit(1 if miss else 0)
