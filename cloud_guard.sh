#!/bin/bash
# 클라우드 전용 가드 — 채점에 **필요 없는** 것을 계속 죽인다.
#
#   nohup bash coweek/cloud_guard.sh 30 > ~/physicar_ws/cloud_guard.log 2>&1 &
#
# 로컬의 `probe_tmp/viewer_guard.sh` 는 CPU 임계값(50 %)을 쓴다 — 로컬에서는
# 관전 화면을 실제로 보기 때문이다. **클라우드는 채점 전용이라 3D 뷰어가
# 아예 필요 없다.** 그래서 여기서는 임계값 없이 보이는 즉시 죽인다.
#
# 왜 상주가 필요한가 (실측): `sim_api` 가 뷰어를 자식으로 계속 되살리므로 한 번
# 죽이는 것으로는 부족하다. 2026-08-22 에 **2시간 31분 동안 CPU 54.6 %** 를
# 물고 loadavg 를 6.5 까지 올린 채로 발견됐다 (4 vCPU 의 절반 이상).
# 로컬에서도 하루에 세 번 재발한 이력이 있다.
#
# 죽여도 안전하다: 관전 화면(웹 3D)만 꺼지고 `gz sim`·`/sim/api`·ROS 브리지·
# 채점에는 영향이 없다. 실제로 죽인 뒤 기준선 2런이 정상 종료됐다.
#
# ⚠️ 여기서 **죽이지 않는 것들** (전부 채점에 필요하다):
#   gz sim              시뮬레이터 본체
#   sim_api.py          /sim/api (심판이 쓰는 그 API)
#   webserver_node      웹 서버 — /sim 프록시 경로가 이걸 거친다
#   parameter_bridge    ROS <-> gz (카메라·라이다·cmd_vel)
#   ekf_node            /odom (드라이버는 안 쓰지만 스택이 기대한다)
IVL=${1:-30}
echo "[cloud-guard] 시작 — gz-launch 뷰어를 ${IVL}초마다 무조건 종료"
while true; do
  for P in $(pgrep -f websocket.gzlaunch 2>/dev/null); do
    if kill "$P" 2>/dev/null; then
      echo "[cloud-guard] $(date -u +%H:%M:%S) killed gz-launch pid=$P"
    fi
  done
  sleep "$IVL"
done
