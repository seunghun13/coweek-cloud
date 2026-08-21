#!/bin/bash
# run.sh 와 같은 역할이되 잔차 정책을 얹어 돌린다.
# COWEEK_RL_MODEL 이 비어 있으면 잔차 0 = racer-v64 그대로다.
if [ -f /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append ]; then
  . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
fi
command -v ros2 >/dev/null 2>&1 || source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
cd /home/physicar/physicar_ws
while true; do
  python3 -u coweek/rl/run_policy.py 
  echo "[run_rl.sh] runner exited — restarting in 0.5s" >&2
  sleep 0.5
done
