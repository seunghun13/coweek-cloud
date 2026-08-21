#!/bin/bash
# AMET 2026 — team 코윅윅 (#28) race entrypoint.
# Executed by officials as: source /home/physicar/physicar_ws/run.sh
# Load the platform ROS environment (RMW/CycloneDDS pinning included).
# Sim path first; on the real kit the official login shell provides the env,
# and the fallbacks below cover a bare shell.
if [ -f /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append ]; then
  . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
elif [ -f /opt/physicar/src/physicar-ros/deploy/real/bashrc-append ]; then
  . /opt/physicar/src/physicar-ros/deploy/real/bashrc-append
fi
command -v ros2 >/dev/null 2>&1 || source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

cd /home/physicar/physicar_ws

# ── RL 잔차 디스패치 ──────────────────────────────────────────────
# 심판이 띄우는 프로세스에는 우리 환경변수가 안 닿으므로 파일로 고른다.
# ACTIVE_MODEL 이 없으면(기본) 아래는 정확히 종전과 같은 driver.py 다.
RL_ACTIVE=/home/physicar/physicar_ws/coweek/rl/ACTIVE_MODEL
ENTRY="coweek/driver.py"
ARGS=""
if [ -s "$RL_ACTIVE" ]; then
  RL_M=$(head -1 "$RL_ACTIVE" | tr -d '\r\n')
  if [ "$RL_M" = "none" ]; then
    ENTRY="coweek/rl/run_policy.py"; ARGS=""
    echo "[run.sh] RL runner, zero residual (verification)" >&2
  elif [ -f "$RL_M" ]; then
    ENTRY="coweek/rl/run_policy.py"; ARGS="--model $RL_M"
    echo "[run.sh] RL residual active: $RL_M" >&2
  else
    echo "[run.sh] !! ACTIVE_MODEL points at a missing file — falling back to driver.py" >&2
  fi
fi

# Restart loop: if the driver dies mid-run the watchdog stops the car, green
# stays latched in the world, and a fresh (stateless) process resumes driving.
while true; do
  python3 -u $ENTRY $ARGS
  echo "[run.sh] driver exited — restarting in 0.5s" >&2
  sleep 0.5
done
