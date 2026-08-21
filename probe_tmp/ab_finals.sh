#!/bin/bash
# 본선 조건(콘 6개 전부 랜덤) A/B. 유휴를 기다렸다가 시작한다.
#   usage: ab_finals.sh <runs> <ref-tag>
# --pin '' 를 셸 인용 없이 확실히 빈 인자로 넘기려고 별도 스크립트로 뺐다.
CW=/home/physicar/physicar_ws/coweek
RUNS=${1:-15}
REF=${2:-racer-v60}
LIMIT=1.90
for i in $(seq 1 40); do
  L=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$L" -v k="$LIMIT" 'BEGIN{exit !(l<k)}'; then
    echo "idle reached at $L"
    break
  fi
  echo "[$(date +%H:%M:%S)] loadavg1=$L try $i/40"
  sleep 15
done
echo "=== 본선 조건 A/B 시작 (load $(cut -d' ' -f1-3 /proc/loadavg)) ==="
exec bash "$CW/ab_practice.sh" "$RUNS" "$REF" --random-cones --pin ''
