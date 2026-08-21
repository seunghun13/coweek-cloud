#!/bin/bash
# 본선 조건(콘 6개 전부 랜덤) 단독 배치. A/B 가 아니라 실패 귀속용 데이터 수집.
#   usage: run_finals.sh <outfile> <runs>
# 결과와 트레이스를 outfile 옆에 같이 남긴다 — /tmp/ab_cur.txt 는 다음 배치가
# 덮어쓰므로, 나중에 분석하려면 배치마다 따로 보관해야 한다 (한 번 잃어봤다).
set -u
OUT=${1:-/tmp/finals.txt}
RUNS=${2:-15}
CW=/home/physicar/physicar_ws/coweek
cd /home/physicar/physicar_ws
python3 "$CW/kill_orphans.py" | tail -1
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  L=$(cut -d' ' -f1 /proc/loadavg)
  if awk -v l="$L" 'BEGIN{exit !(l<1.90)}'; then break; fi
  echo "load $L — 대기"
  sleep 15
done
echo "=== 본선 조건 $RUNS 런 시작 (load $(cut -d' ' -f1-3 /proc/loadavg)) ==="
python3 -u "$CW/referee.py" --runs "$RUNS" --random-cones --pin '' > "$OUT" 2>&1
cp /tmp/ref_traces.jsonl "$OUT.traces.jsonl" 2>/dev/null || true
python3 "$CW/kill_orphans.py" | tail -1
echo "=== 완료 ==="
grep -E '^SUMMARY' "$OUT"
