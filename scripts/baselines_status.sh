#!/usr/bin/env bash
# Is the overnight baseline run still going, and how far has it got?
#
#   bash scripts/baselines_status.sh
cd "$(dirname "$0")/.."
STATUS="results/mle_fits/.overnight_status"

if pgrep -f run_baselines_overnight >/dev/null; then
  echo "RUNNING  (started $(ps -o lstart= -p "$(pgrep -f run_baselines_overnight | head -1)" 2>/dev/null | xargs))"
else
  if [[ -f "$STATUS" ]] && grep -q "DONE" "$STATUS"; then echo "FINISHED"
  elif [[ -f "$STATUS" ]] && grep -q "FAILED" "$STATUS"; then echo "FAILED — see the log"
  else echo "NOT RUNNING (and no completion marker: it may have been killed)"; fi
fi

echo
echo "--- progress ---"
[[ -f "$STATUS" ]] && cat "$STATUS" || echo "(no status file yet)"

echo
echo "--- last lines of the log ---"
tail -n 5 /tmp/grin_baselines.log 2>/dev/null || echo "(no log at /tmp/grin_baselines.log)"
