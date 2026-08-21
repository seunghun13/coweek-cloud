#!/bin/bash
# 클라우드 채점 표준 절차 — 클라우드에서 실행한다.
#
#   bash coweek/cloud_eval.sh --model rl/models/s7/final.zip --runs 15 --baseline --finals
#   bash coweek/cloud_eval.sh --baseline --runs 15 --finals        # 기준선만
#
# 결과: ~/physicar_ws/cloud_eval.log  (프록시 8099 로 읽는다)
#       https://sim.physicar.ai/proxy/8099/cloud_eval.log
#
# ⚠️ **뷰어(gz-launch) 를 먼저 죽인다.** 실측: 2 시간 31 분 동안 CPU 54.6 % 를
#    물고 loadavg 를 6.5 까지 올린다 (4 vCPU 머신에서 절반이 넘는다). 죽이면
#    RTF 0.863 -> 0.953 까지 올라온 이력이 있다. 죽여도 안전하다 — sim_api 의
#    자식이라 잠시 뒤 새 PID 로 되살아나고 그건 얌전하다(~2 %).
#
# ⚠️ **여기서 나온 `결승선 놓침` 수치는 판정에 쓰지 않는다.** 클라우드 RTF
#    0.90 은 랩타임·페널티를 거의 안 바꾸지만 놓침만 3 % -> 23 % 로 터뜨린다
#    (실측). 놓침·속도 판정은 **로컬**에서 한다 (역할 분담, STATUS.md).
set -u
cd ~/physicar_ws

echo "=== 사전 정리 ==="
pkill -f 'gz-launch' 2>/dev/null && echo "뷰어(gz-launch) 종료" || echo "뷰어 없음"
sleep 3
python3 coweek/kill_orphans.py || true

echo "=== 부하 ==="
cat /proc/loadavg
ps -eo pcpu,args --sort=-pcpu | head -4

echo "=== 채점 시작: $* ==="
date -u
python3 -u coweek/rl/eval_policy.py "$@"
rc=$?
echo "=== 채점 종료 rc=$rc ==="
date -u
cat /proc/loadavg
echo CLOUD_EVAL_DONE
exit $rc
