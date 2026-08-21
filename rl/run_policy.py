#!/usr/bin/env python3
"""driver.py 를 그대로 돌리되 학습된 속도 잔차를 얹는다.

`driver.py` 는 **한 줄도 바뀌지 않는다.** `/speed` 퍼블리셔만 가로채서
base + residual 을 대신 낸다.

    정책 없이 실행  ->  잔차 0  ->  **정확히 racer-v64**
    --model 지정    ->  잔차 적용

## 쓰임새
1. 심판 채점: run.sh 대신 이걸 띄우고 referee.py 를 돌린다
2. 실차 배포: 잔차가 이상하면 --model 만 빼면 즉시 v64 로 복귀

## 안전 장치 (전부 코드로 강제)
- 최종 명령은 [0, V_HARD=1.40] 하드 클립
- 프레임이 낡으면(STALE) 잔차를 0 으로 — 낡은 관측으로 가속하지 않는다
- 정책 로드 실패 시 조용히 잔차 0 으로 계속 달린다 (완주가 최우선)
- 관측 구성은 coweek_env 의 obs_row 를 **그대로 재사용**한다.
  학습과 배포가 다른 관측을 쓰면 조용히 성능만 무너진다.

usage:
    python3 rl/run_policy.py                 # = racer-v64
    python3 rl/run_policy.py --model rl/runs/s1/final.zip
"""
import argparse
import os
import sys
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np
import rclpy
from std_msgs.msg import Float64

import driver as drv
from coweek_env import (CAP_NAMES, DEFAULT_LT, HIST, OBS_ROW_DIM, RES_MAX,
                        V_HARD, obs_row)

STALE_NO_RESIDUAL = 0.35     # 초. driver.STALE_SLOW 와 같은 문턱


class ResidualRunner:
    def __init__(self, model_path=None, res_max=RES_MAX, log_every=0):
        if not rclpy.ok():
            rclpy.init()
        self.node = drv.CoweekDriver()
        self.real_speed = self.node.pub_speed

        class _Cap:
            base = 0.0
            n = 0

            def publish(self, msg):
                self.base = float(msg.data)
                self.n += 1

        self.cap = _Cap()
        self.node.pub_speed = self.cap

        self.res_max = float(res_max)
        self.model = None
        if model_path:
            try:
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path, device='cpu')
                print('[rl] policy loaded: %s' % model_path, flush=True)
            except Exception as e:
                # 완주가 최우선이다. 정책이 없으면 v64 로 달린다.
                print('[rl] !! policy load failed (%s) — 잔차 0 으로 진행' % e,
                      flush=True)
        else:
            print('[rl] no model — 잔차 0 (= racer-v64)', flush=True)

        self.hist = []
        self.log_every = log_every
        self.k = 0
        self.res_sum = 0.0
        # 직전 행동. 학습 env 가 관측에 넣으므로 **여기도 똑같이** 넣어야 한다
        # (coweek_env.obs_row 독스트링). 학습과 배포가 다른 관측을 쓰면
        # 예외 없이 조용히 성능만 무너진다.
        self.prev_a = 0.0

    def _obs(self, lt):
        row = obs_row(lt, self.prev_a)
        if not self.hist:
            self.hist = [row] * HIST
        else:
            self.hist.append(row)
            self.hist = self.hist[-HIST:]
        return np.concatenate(self.hist).astype(np.float32)

    def spin(self):
        node = self.node
        while rclpy.ok():
            n0 = self.cap.n
            t0 = time.monotonic()
            while self.cap.n == n0 and time.monotonic() - t0 < 1.0:
                rclpy.spin_once(node, timeout_sec=0.005)
            lt = getattr(node, 'last_tick', None) or DEFAULT_LT

            res = 0.0
            act = 0.0                     # 이번 틱에 실제로 적용한 정규화 행동
            if self.model is not None and lt.get('green'):
                stale = (lt['frame_t'] is not None
                         and time.monotonic() - lt['frame_t'] > STALE_NO_RESIDUAL)
                if not stale:
                    try:
                        a, _ = self.model.predict(self._obs(lt), deterministic=True)
                        act = float(np.clip(a[0], -1.0, 1.0))
                        res = act * self.res_max
                    except Exception:
                        act, res = 0.0, 0.0
            # 잔차를 0 으로 강제한 틱(초록 전·STALE·예외)은 prev_a 도 0 이다 —
            # 실제로 낸 행동이 0 이므로 관측이 사실과 어긋나면 안 된다
            self.prev_a = act
            cmd = float(np.clip(self.cap.base + res, 0.0, V_HARD))
            self.real_speed.publish(Float64(data=cmd))

            self.k += 1
            self.res_sum += res
            if self.log_every and self.k % self.log_every == 0:
                print('[rl] tick %d base %.2f res %+.3f cmd %.2f (avg res %+.3f)'
                      % (self.k, self.cap.base, res, cmd,
                         self.res_sum / self.k), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('COWEEK_RL_MODEL', ''))
    ap.add_argument('--res-max', type=float, default=RES_MAX)
    ap.add_argument('--log-every', type=int, default=0)
    a = ap.parse_args()
    r = ResidualRunner(a.model or None, a.res_max, a.log_every)
    try:
        r.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            r.real_speed.publish(Float64(data=0.0))
            r.node.pub_steer.publish(Float64(data=0.0))
        except Exception:
            pass


if __name__ == '__main__':
    main()
