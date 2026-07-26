#!/usr/bin/env bash
# Convention lint for the FAMAIL manuscript (see README.md). Exit 1 on any hit.
# A line may carry `% lint-allow: <reason>` to suppress a hit (e.g. ablation context).
set -u
cd "$(dirname "$0")"
fail=0
check() { # $1 = grep -E pattern, $2 = description
  local hits
  hits=$(grep -RInE "$1" --include='*.tex' . | grep -v 'lint-allow' || true)
  if [ -n "$hits" ]; then
    echo "LINT FAIL — $2:"
    echo "$hits"
    fail=1
  fi
}
check '54 ?(\\%|%|percent)'                        'ungrounded "54%" figure (banned until grounded)'
check 'causal (effect|impact|estimate)|causally'   'causality-claim language (F_demo is associational)'
check '0\.0144|0\.0128|0\.0139'                    'trim-only headline number outside ablation context'
check '0\.0222|0\.0328|0\.0310|0\.8132|0\.0205|0\.0278|0\.0311|87\.4|84\.9' 'old-alpha (0.2,0.7,0.1) supply-lift number outside a labeled prior-era context'
check '[Cc]laude|[Aa]nthropic|[Cc]owork|[Cc]opilot|ChatGPT' 'tool/product name'
check '(beats|outperforms) (Shenzhen|the first city)' 'SF must reproduce, not beat'

# Render-geometry gate (2026-07-16, from reviews/2026-07-15-render-qa.md): main.log is
# ISO-8859-encoded, so grep NEEDS -a (plain grep treats it as binary and silently matches
# nothing — that is how 55 Overfull boxes passed the gates). Threshold 5pt (tightened from 8pt after the cut campaign, 2026-07-21); sub-5pt
# boxes are largely absorbed by microtype expansion on Overleaf and get swept in the 8-page
# compression pass — tighten to 5pt after that. Requires a fresh `latexmk` (main.log present).
if [ -f main.log ]; then
  overfull=$(grep -a 'Overfull \\hbox' main.log | grep -oE '\(([0-9.]+)pt too wide' | grep -oE '[0-9.]+' | awk '$1 > 5' || true)
  if [ -n "$overfull" ]; then
    echo "LINT FAIL — Overfull hbox(es) > 5pt in main.log (tightened from 8pt post-cut, 2026-07-21; render and inspect):"
    grep -a 'Overfull \\hbox' main.log | awk '{ if (match($0, /\(([0-9.]+)pt/, m) && m[1]+0 > 5) print "  " $0 }'
    fail=1
  fi
else
  echo "LINT WARN — main.log not found; run latexmk first so the Overfull gate can check geometry."
fi

# Citation-checklist coverage guard (2026-07-16): every cited key must have a block in
# CITATION_PRIORITY_CHECKLIST.md — the maintenance rule says any session changing \cite
# usage or refs.bib updates the checklist in the same session. Fails on cited-but-unlisted.
if [ -f CITATION_PRIORITY_CHECKLIST.md ]; then
  missing=$(comm -23 \
    <(grep -rho 'cite{[^}]*}' --include='*.tex' . | sed 's/.*cite{//;s/}//' | tr ',' '\n' | sed 's/ //g' | sort -u) \
    <(grep -o '\*\*[a-z0-9]*\*\*' CITATION_PRIORITY_CHECKLIST.md | sed 's/\*//g' | sort -u))
  if [ -n "$missing" ]; then
    echo "LINT FAIL — cited keys missing from CITATION_PRIORITY_CHECKLIST.md (update it in this session):"
    echo "$missing"
    fail=1
  fi
fi

# Body-spill gate (2026-07-26, added for the Zhang R2 restructure). KDD's 8pp body
# limit is HARD and page 8 currently has exactly ONE spare line-slot, so any prose or
# float added to §4 can silently push §6's tail onto page 9. Before this gate the only
# check was reading the PDF by eye. The bibliography heading is the marker: it must
# begin on page 9, i.e. the body must fit in 8 pages. Requires a fresh main.pdf.
if [ -f main.pdf ] && command -v pdftotext >/dev/null 2>&1; then
  refpage=""
  for p in $(seq 1 12); do
    if pdftotext -f "$p" -l "$p" main.pdf - 2>/dev/null | grep -qE '^REFERENCES$'; then
      refpage=$p; break
    fi
  done
  if [ -z "$refpage" ]; then
    echo "LINT WARN — could not locate the REFERENCES heading; body-spill gate did not run."
  elif [ "$refpage" -lt 9 ]; then
    echo "LINT WARN — REFERENCES starts on page $refpage (expected 9); body is shorter than budgeted."
  elif [ "$refpage" -gt 9 ]; then
    echo "LINT FAIL — body spilled past 8 pages: REFERENCES starts on page $refpage, expected 9."
    fail=1
  else
    # 2026-07-26: "REFERENCES is on page 9" is NOT sufficient. The heading can sit
    # partway DOWN page 9 with body text above it, which is still a body overrun and
    # passed the earlier version of this gate silently. Count the rendered lines that
    # precede the heading in its own column.
    spill=$(python3 - <<'PYEOF'
import re, subprocess
out = subprocess.run(['pdftotext','-f','9','-l','9','-bbox','main.pdf','-'],
                     capture_output=True, text=True).stdout
ws = [(float(m.group(1)), float(m.group(2)), m.group(3))
      for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)"[^>]*>([^<]*)</word>', out)]
h = [(x, y) for x, y, w in ws if w.upper().startswith('REFERENCE')]
if not h:
    print(0)
else:
    hx, hy = h[0]
    col = 0 if hx < 300 else 1
    ys = {round(y) for x, y, w in ws if (0 if x < 300 else 1) == col and 90 < y < hy}
    print(len(ys))
PYEOF
)
    if [ "${spill:-0}" -gt 0 ]; then
      echo "LINT WARN — body overruns 8 pages: $spill rendered line(s) of body text sit above the REFERENCES heading on page 9."
    fi
  fi
else
  echo "LINT WARN — main.pdf or pdftotext unavailable; body-spill gate did not run."
fi
exit $fail
