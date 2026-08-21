#!/usr/bin/env python3
"""정책이 상황별로 어떤 잔차를 내는지 직접 물어본다 (시뮬 불필요, 몇 초).

    python3 rl/policy_probe.py rl/runs/s7/final.zip

합성 관측이라 근사치다 — 실제 관측 분포와 다를 수 있다. 그래도 "정책이
직선에서조차 감속하는가" 같은 방향 질문에는 충분히 답한다.
"""
import sys

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np
import time

import coweek_env as ce
from coweek_env import DEFAULT_LT, HIST, RES_MAX, obs_row

# 대표 상황들: (이름, lt 덮어쓰기, prev_a)
now = time.monotonic()
CASES = [
    ('직선 V_MAX (0.90)',      dict(green=True, speed=0.90, steer=0.02, aim=0.05,
                                    arc_safe=0.96, fmin=4.0, frame_t=now), 0.0),
    ('직선 V_MAX, 직전 +0.5',   dict(green=True, speed=0.90, steer=0.02, aim=0.05,
                                    arc_safe=0.96, fmin=4.0, frame_t=now), 0.5),
    ('완만 코너 alat (0.75)',   dict(green=True, speed=0.75, steer=0.15, aim=0.3,
                                    arc_safe=0.96, fmin=2.0, frame_t=now), 0.0),
    ('급코너 풀락 (0.70)',      dict(green=True, speed=0.70, steer=0.33, aim=0.8,
                                    arc_safe=0.80, fmin=1.2, frame_t=now), 0.0),
    ('콘 회피 CONE_SLOW (0.45)', dict(green=True, speed=0.45, steer=0.10, aim=0.2,
                                    arc_safe=0.60, fmin=1.0, lbias=0.3,
                                    dodging=True, mode='arc', ncone=1,
                                    carea=3000, frame_t=now), 0.0),
    ('콘 접근 0.60',            dict(green=True, speed=0.60, steer=0.05, aim=0.1,
                                    arc_safe=0.90, fmin=1.3, ncone=1,
                                    carea=6000, frame_t=now), 0.0),
    ('라이다 프리브레이크 0.55', dict(green=True, speed=0.55, steer=0.05, aim=0.1,
                                    arc_safe=0.96, fmin=1.4, frame_t=now), 0.0),
    ('결승 밴드 checker 0.45',  dict(green=True, speed=0.45, steer=0.02, aim=0.05,
                                    arc_safe=0.96, fmin=2.0, checker=True,
                                    frame_t=now), 0.0),
    ('짧은 원호 arc_short 0.50', dict(green=True, speed=0.50, steer=0.20, aim=0.5,
                                    arc_safe=0.50, fmin=1.5, mode='arc',
                                    frame_t=now), 0.0),
]


def main():
    from stable_baselines3 import SAC
    model = SAC.load(sys.argv[1] if len(sys.argv) > 1
                     else 'rl/runs/s7/final.zip', device='cpu')
    print('%-26s %8s %8s %8s' % ('상황(base)', '잔차', '최종속도', '결정론'))
    for name, over, prev_a in CASES:
        lt = dict(DEFAULT_LT)
        lt.update(over)
        row = obs_row(lt, prev_a)
        obs = np.concatenate([row] * HIST).astype(np.float32)
        a_det, _ = model.predict(obs, deterministic=True)
        res = float(np.clip(a_det[0], -1, 1)) * RES_MAX
        # 탐색 분포도 참고로 (10회 표본 평균)
        samp = [float(np.clip(model.predict(obs, deterministic=False)[0][0],
                              -1, 1)) * RES_MAX for _ in range(10)]
        print('%-26s %+8.3f %8.3f  (탐색 평균 %+.3f, sd %.3f)'
              % (name, res, lt['speed'] + res,
                 float(np.mean(samp)), float(np.std(samp))))


if __name__ == '__main__':
    main()
