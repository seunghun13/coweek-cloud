#!/usr/bin/env python3
"""driver.py 를 그대로 돌리되 학습된 속도 잔차를 얹는다.

`driver.py` 는 계측(last_tick·teleport_n) 말고는 바뀌지 않는다. `/speed`
퍼블리셔만 가로채서 base + residual 을 대신 낸다.

    정책 없이 실행  ->  잔차 0  ->  **정확히 racer-v64**
    --model 지정    ->  잔차 적용

## 쓰임새
1. 심판 채점: run.sh 대신 이걸 띄우고 referee.py 를 돌린다
2. 실차 배포: 잔차가 이상하면 --model 만 빼면 즉시 v64 로 복귀

## 안전 장치 (전부 코드로 강제)
- 최종 명령은 [0, V_HARD=1.40] 하드 클립
- 프레임이 낡으면(STALE — `frame_t is None` 포함) 잔차 0 **그리고 히스토리
  리셋**. 낡은 관측으로 가속하지 않고, 낡은 row 가 나중에 새 row 와 섞이지도
  않는다 (전수검수 P1-10 — 종전엔 None 이 stale 로 안 잡혔고, stale 동안
  히스토리가 얼어 있다가 pre-stale row 가 새 row 와 섞였다)
- **텔레포트 감지 시 히스토리 리셋** (P0-4): driver v34c 가 장면 점프로
  텔레포트를 감지하면(`teleport_n` 증가) 자기 래치 5개를 지운다 — 러너도
  같은 신호에서 hist·prev_a 를 지운다. 학습 env 가 같은 규칙을 쓴다.
- 정책 로드 실패 시 조용히 잔차 0 으로 계속 달린다 (**런타임은** 완주가
  최우선). 단 그 사실을 상태 파일에 남기므로 **평가는** fail-closed 다 —
  eval_policy.py 가 로드 실패·predict 예외를 보면 그 배치를 무효화한다.
- 관측 구성은 coweek_env 의 obs_row 를 **그대로 재사용**한다.

usage:
    python3 rl/run_policy.py                 # = racer-v64
    python3 rl/run_policy.py --model rl/runs/s1/final.zip
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np
import rclpy
from std_msgs.msg import Float64

import driver as drv
from coweek_env import (CAP_NAMES, CONTRACT, DEFAULT_LT, HIST, OBS_ROW_DIM,
                        RES_MAX, STALE_NO_RESIDUAL, V_HARD, obs_row)
# STALE_NO_RESIDUAL 은 **학습 env 에서 가져온다** — 두 곳에 따로 적으면
# 조용히 어긋나고, 그 순간 학습과 배포가 다른 규칙으로 달린다.
# ⚠️ **append 로그**여야 한다. 심판은 런마다 run.sh 를 새로 spawn 하고 run.sh
# 는 죽은 러너를 재시작하므로, 고정 경로를 덮어쓰면 3번째 런에서 모델 로드가
# 실패해도 4번째 런의 러너가 깨끗한 상태로 덮어써 평가가 그 배치를 유효로 본다.
# 한 줄씩 붙이면 배치 전체의 모든 프로세스가 남아 fail-closed 가 실제로 닫힌다.
STATUS_PATH = '/tmp/rl_runner_status.jsonl'
STATUS_EVERY = 100           # 틱마다 상태 한 줄 추가


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
        self.model_path = model_path or ''
        self.loaded = False
        self.predict_errors = 0
        if model_path:
            try:
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path, device='cpu')
                self.loaded = True
                print('[rl] policy loaded: %s' % model_path, flush=True)
            except Exception as e:
                # 완주가 최우선이다. 정책이 없으면 v64 로 달린다.
                # (평가는 상태 파일로 이 상황을 잡아 무효 처리한다 — P0-5)
                print('[rl] !! policy load failed (%s) — 잔차 0 으로 진행' % e,
                      flush=True)
        else:
            print('[rl] no model — 잔차 0 (= racer-v64)', flush=True)

        self.hist = []
        self.log_every = log_every
        self.k = 0
        self.res_abs_sum = 0.0
        # 직전 행동. 학습 env 가 관측에 넣으므로 **여기도 똑같이** 넣어야 한다
        # (coweek_env.obs_row 독스트링). 학습과 배포가 다른 관측을 쓰면
        # 예외 없이 조용히 성능만 무너진다.
        self.prev_a = 0.0
        self.tp_seen = None          # driver teleport_n 미러 (첫 틱에 흡수)
        self._write_status('start')

    def _write_status(self, event='tick'):
        """평가 fail-closed 용 상태 로그 (P0-5). 실패해도 주행은 계속한다.

        `os.open(O_APPEND)` 한 줄 쓰기라 여러 러너 프로세스가 겹쳐도
        레코드가 섞이지 않는다."""
        try:
            rec = json.dumps({'pid': os.getpid(), 'event': event,
                              'model': self.model_path,
                              'loaded': bool(self.loaded),
                              'ticks': self.k,
                              'res_abs_mean': (self.res_abs_sum / self.k
                                               if self.k else 0.0),
                              'res_max': self.res_max,
                              'contract': CONTRACT,
                              'predict_errors': self.predict_errors,
                              't': time.time()}) + '\n'
            fd = os.open(STATUS_PATH,
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, rec.encode())
            finally:
                os.close(fd)
        except Exception:
            pass

    def _reset_memory(self):
        self.hist = []
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

            # 텔레포트 미러링 (P0-4): driver 가 래치를 지운 그 신호로 우리
            # 기억도 지운다. 첫 틱의 tp_seen=None 은 흡수만 한다.
            tp = int(lt.get('teleport_n', 0))
            if self.tp_seen is not None and tp != self.tp_seen:
                self._reset_memory()
            self.tp_seen = tp

            res = 0.0
            act = 0.0                     # 이번 틱에 실제로 적용한 정규화 행동
            if self.model is not None and lt.get('green'):
                # P1-10: frame_t 가 None 인 것도 stale 이다 (아직 프레임을
                # 한 장도 못 본 상태로 가속하면 안 된다)
                stale = (lt['frame_t'] is None
                         or time.monotonic() - lt['frame_t'] > STALE_NO_RESIDUAL)
                if stale:
                    # stale 동안 히스토리를 얼리지 않는다 — 얼렸다가 복구되면
                    # pre-stale row 가 새 row 와 섞인다 (P1-10)
                    self._reset_memory()
                else:
                    try:
                        a, _ = self.model.predict(self._obs(lt), deterministic=True)
                        act = float(np.clip(a[0], -1.0, 1.0))
                        res = act * self.res_max
                    except Exception:
                        self.predict_errors += 1
                        act, res = 0.0, 0.0
                        self._reset_memory()
                        if self.predict_errors == 1:
                            # 즉시 남긴다 — 100틱 주기만 믿으면 프로세스가
                            # 곧 죽는 경우 이 사실이 통째로 유실된다
                            self._write_status('predict_error')
            # 잔차를 0 으로 강제한 틱(초록 전·STALE·예외·텔레포트)은 prev_a 도
            # 0 이다 — 실제로 낸 행동이 0 이므로 관측이 사실과 어긋나면 안 된다
            self.prev_a = act
            cmd = float(np.clip(self.cap.base + res, 0.0, V_HARD))
            self.real_speed.publish(Float64(data=cmd))

            self.k += 1
            self.res_abs_sum += abs(res)
            if self.k % STATUS_EVERY == 0:
                self._write_status()
            if self.log_every and self.k % self.log_every == 0:
                print('[rl] tick %d base %.2f res %+.3f cmd %.2f (avg |res| %.3f)'
                      % (self.k, self.cap.base, res, cmd,
                         self.res_abs_sum / self.k), flush=True)


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
        r._write_status('end')
        try:
            r.real_speed.publish(Float64(data=0.0))
            r.node.pub_steer.publish(Float64(data=0.0))
        except Exception:
            pass


if __name__ == '__main__':
    main()
