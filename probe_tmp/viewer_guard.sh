#!/bin/bash
# 웹소켓 뷰어(gz-launch) 폭주 감시자.
#
# 오늘만 세 번 재발했다: 코어 하나를 100~110% 물고 loadavg 를 1.6 -> 3.9 로 올린다.
# 배치 도중에 이게 터지면 A/B 두 반쪽의 부하 조건이 달라져 결과가 통째로 무효다
# (CLAUDE.md 의 v45/v46 사고와 같은 형태). sim_api 가 자식으로 계속 되살리므로
# 한 번 죽이는 것으로는 부족하고, 배치가 도는 동안 지켜봐야 한다.
#
# 죽여도 안전하다 — 관전 화면만 꺼지고 시뮬·채점에는 영향이 없다.
# usage: viewer_guard.sh [cpu_limit] [interval]
LIMIT=${1:-50}
IVL=${2:-20}
echo "[guard] 시작 — gz-launch CPU > ${LIMIT}% 면 종료, ${IVL}초 간격"
while true; do
  for P in $(pgrep -f websocket.gzlaunch); do
    C=$(ps -p "$P" -o %cpu= 2>/dev/null | tr -d ' ')
    [ -z "$C" ] && continue
    if awk -v c="$C" -v l="$LIMIT" 'BEGIN{exit !(c+0 > l+0)}'; then
      echo "[guard] $(date +%H:%M:%S) pid=$P cpu=${C}% -> kill"
      kill "$P" 2>/dev/null
    fi
  done
  sleep "$IVL"
done
