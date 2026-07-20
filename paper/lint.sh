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
# nothing — that is how 55 Overfull boxes passed the gates). Threshold 8pt for now; sub-8pt
# boxes are largely absorbed by microtype expansion on Overleaf and get swept in the 8-page
# compression pass — tighten to 5pt after that. Requires a fresh `latexmk` (main.log present).
if [ -f main.log ]; then
  overfull=$(grep -a 'Overfull \\hbox' main.log | grep -oE '\(([0-9.]+)pt too wide' | grep -oE '[0-9.]+' | awk '$1 > 8' || true)
  if [ -n "$overfull" ]; then
    echo "LINT FAIL — Overfull hbox(es) > 8pt in main.log (render and inspect; see reviews/2026-07-15-render-qa.md):"
    grep -a 'Overfull \\hbox' main.log | awk '{ if (match($0, /\(([0-9.]+)pt/, m) && m[1]+0 > 8) print "  " $0 }'
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
exit $fail
