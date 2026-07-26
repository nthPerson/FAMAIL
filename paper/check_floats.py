"""Report every float's rendered position and verify it appears AFTER its first
in-text reference, in true two-column reading order (page, then column, then y).

KNOWN LIMITATION (measured 2026-07-26): `pdftotext -bbox` does not always emit
every word that plain `pdftotext` does — on page 8 it reports 2 "Table" tokens
where the text extraction finds 3, silently dropping one in-text reference. So a
"FLOAT BEFORE REF" verdict is a PROMPT TO GO LOOK, not proof: confirm against
`pdftotext -f N -l N main.pdf -` before acting on it. Under-detection can only
make this tool too pessimistic, never too permissive, which is the safe
direction — but do not treat a clean run as proof that every reference exists.

Captions render as "Table 3:" / "Figure 3:"; in-text references render as
"Table 3" / "Fig. 3" with no colon. We locate both from pdftotext -bbox.
"""
import re
import subprocess
import sys

PDF = "/home/robert/FAMAIL/paper/main.pdf"
COL_SPLIT = 300.0  # page midpoint in pdftotext -bbox coordinates


def words(page):
    out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", PDF, "-"],
                         capture_output=True, text=True).stdout
    return [(float(m.group(1)), float(m.group(2)), m.group(3))
            for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)"[^>]*>([^<]*)</word>', out)]


def key(page, x, y):
    """Reading order: page, then column (left before right), then vertical."""
    return (page, 0 if x < COL_SPLIT else 1, y)


captions, refs = {}, {}
for page in range(1, 15):
    ws = words(page)
    for i, (x, y, w) in enumerate(ws):
        w = w.lstrip("([{\u201c\u2018~")          # rendered tokens carry leading punctuation
        if w not in ("Table", "Figure", "Fig."):
            continue
        nxt = ws[i + 1][2] if i + 1 < len(ws) else ""
        kind = "Table" if w == "Table" else "Figure"
        m = re.match(r"^(\d+)([:.,;)\]]*)$", nxt)
        if not m:
            continue
        name = f"{kind} {m.group(1)}"
        if m.group(2).startswith(":"):               # a caption ("Table 3:")
            captions.setdefault(name, key(page, x, y))
        else:                                        # an in-text reference
            k = key(page, x, y)
            if name not in refs or k < refs[name]:
                refs[name] = k

print(f"{'float':10} {'caption at':22} {'first ref at':22} verdict")
bad = 0
for name in sorted(captions, key=lambda n: captions[n]):
    c = captions[name]
    r = refs.get(name)
    fmt = lambda k: f"p{k[0]} col{'LR'[k[1]]} y={k[2]:.0f}" if k else "NONE"
    if r is None:
        verdict, bad = "!! NEVER REFERENCED", bad + 1
    elif r < c:
        verdict = "ok (ref first)"
    else:
        verdict, bad = "!! FLOAT BEFORE REF", bad + 1
    print(f"{name:10} {fmt(c):22} {fmt(r):22} {verdict}")

print()
print("reading-order sequence of floats:",
      " -> ".join(sorted(captions, key=lambda n: captions[n])))
sys.exit(1 if bad else 0)
