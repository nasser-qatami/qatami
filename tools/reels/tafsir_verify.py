#!/usr/bin/env python3
"""Verify the tafsir text (field "tf") in reels.json against two independent copies of التفسير الميسر.
Never edits anything. Prints every ayah whose text is not found (after removing tashkeel) in BOTH sources, and exits 1 if any.
Sources: tafsir.app (page meta) is not used for bulk; we use the spa5k/tafsir_api JSON (jsdelivr) and api.quran.com (resource 16).
Run from the repo root:  python3 tools/reels/tafsir_verify.py"""
import json, re, subprocess, sys, os, unicodedata, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def n2(x):
    x = unicodedata.normalize('NFKC', x)
    return re.sub(r'[^ء-ي ]', '', re.sub('[ً-ٰٟـۖ-ۭ]', '', x)).replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ٱ','ا')
def curl(u):
    # retry: a failed fetch used to show up as a false "NOT FOUND"
    for i in range(5):
        r = subprocess.run(["curl", "-s", "-f", "-m", "25", u], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip(): return r.stdout
        time.sleep(2)
    return ""
D = json.load(open(os.path.join(ROOT, "reels.json"), encoding="utf-8"))
ch = json.loads(curl("https://api.quran.com/api/v4/chapters?language=ar"))["chapters"]
num = {n2(c["name_arabic"]): c["id"] for c in ch}
bad = 0
for r in D:
    s = num.get(n2(r["surah"]))
    for a in r["ay"]:
        if not a.get("tf"): continue
        try: b = json.loads(curl(f"https://cdn.jsdelivr.net/gh/spa5k/tafsir_api@main/tafsir/ar-tafsir-muyassar/{s}/{a['n']}.json"))["text"]
        except Exception: b = ""
        try: q = json.loads(curl(f"https://api.quran.com/api/v4/tafsirs/16/by_ayah/{s}:{a['n']}"))["tafsir"]["text"]
        except Exception: q = ""
        if not b and not q: bad += 1; print("FETCH FAILED (both sources):", r["k"], r["surah"], a["n"]); continue
        t = n2(a["tf"].replace("صلى الله عليه وسلم", "")); ok_b = t in n2(b.replace("ﷺ", "").replace("صلى الله عليه وسلم", "")); ok_q = t in n2(q.replace("ﷺ", "").replace("صلى الله عليه وسلم", ""))
        if not (ok_b or ok_q): bad += 1; print("NOT FOUND:", r["k"], r["surah"], a["n"])
        elif not (ok_b and ok_q): print("note: differs in one source:", r["k"], r["surah"], a["n"], "spa5k" if ok_b else "quran.com")
print("checked; unmatched:", bad); sys.exit(1 if bad else 0)
