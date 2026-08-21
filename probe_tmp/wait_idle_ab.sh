#!/bin/bash
# Wait for the box to go idle, then run the A/B. Measuring under a load
# that is still falling is how the v45/v46 halves ended up incomparable.
set -u
CW=/home/physicar/physicar_ws/coweek
LIMIT=${LIMIT:-1.90}
for i in $(seq 1 40); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  echo "[$(date +%H:%M:%S)] loadavg1=$L (limit $LIMIT) try $i/40"
  if awk -v l="$L" -v k="$LIMIT" 'BEGIN{exit !(l<k)}'; then
    echo "idle reached at $L"
    break
  fi
  sleep 15
done
echo "=== starting A/B at load $(cut -d' ' -f1-3 /proc/loadavg) ==="
exec bash "$CW/ab_practice.sh" "$@"
