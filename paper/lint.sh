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
check 'causal (effect|impact|estimate)|causally'   'causality-claim language (F_causal is associational)'
check '0\.0144|0\.0128|0\.0139'                    'trim-only headline number outside ablation context'
check '[Cc]laude|[Aa]nthropic|[Cc]owork|[Cc]opilot|ChatGPT' 'tool/product name'
check '(beats|outperforms) (Shenzhen|the first city)' 'SF must reproduce, not beat'
exit $fail
