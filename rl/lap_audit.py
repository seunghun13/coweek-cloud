#!/usr/bin/env python3
"""정책 한 랩을 실제로 돌리며 **어느 상한 구간에서 시간을 잃는지** 귀속한다.

    python3 rl/lap_audit.py rl/runs/s7/final.zip [seed]

같은 seed 로 정책 랩 → 잔차0 랩을 짝지어 돌리고, base 명령값(= 활성 상한의
지문)별로 시간 점유·평균 잔차·평균 속도를 비교한다. quick_gate 가 "얼마나
느린가"를 재면 이건 "**어디서** 느린가"를 잰다.
"""
import sys
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np

from coweek_env import CoweekResidualEnv

ZERO = np.zeros(1, np.float32)

# base 명령값 -> 상한 이름 (driver.py 상수의 지문)
BUCKETS = [(0.88, 'vmax(0.90)'), (0.58, 'cone_near(0.60)'),
           (0.53, 'lidar1(0.55)'), (0.48, 'arc_short(0.50)'),
           (0.43, 'cone/checker(0.45)'), (0.38, 'lidar2(0.40)'),
           (0.33, 'sharp(0.35)'), (-1, 'alat/기타')]


def bucket(base):
    for lo, name in BUCKETS:
        if base >= lo:
            return name
    return '기타'


def run(env, model, seed):
    obs, _ = env.reset(seed=seed)
    rows = []
    term = trunc = False
    info = {}
    while not (term or trunc):
        if model is None:
            a = ZERO
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        rows.append((info['base'], info['res'], info['cmd'],
                     int(info['off']), info['n_hit'], info['extra_ticks']))
    return rows, info


def digest(rows):
    agg = {}
    for base, res, cmd, off, hit, extra in rows:
        b = bucket(base)
        a = agg.setdefault(b, [0, 0.0, 0.0, 0, 0])
        a[0] += 1 + extra          # 시간(틱)
        a[1] += res
        a[2] += cmd
        a[3] += off
        a[4] += hit
    return agg


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'rl/runs/s7/final.zip'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 91000
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device='cpu')
    env = CoweekResidualEnv(seed=0, dyn=False)   # 참 플랜트, 본선 분포(cone6 고정)
    try:
        pol_rows, pol_info = run(env, model, seed)
        base_rows, base_info = run(env, None, seed)
    finally:
        env.close()

    pa, ba = digest(pol_rows), digest(base_rows)
    names = sorted(set(pa) | set(ba),
                   key=lambda n: -(pa.get(n, [0])[0]))
    print('%-18s | %6s %6s %7s | %6s %6s | %7s' %
          ('구간(base 지문)', '정책틱', '기준틱', 'Δ시간s', '정책잔차', '정책cmd', '기준cmd'))
    for n in names:
        p = pa.get(n, [0, 0, 0, 0, 0])
        b = ba.get(n, [0, 0, 0, 0, 0])
        print('%-18s | %6d %6d %+7.2f | %+6.3f %6.3f | %7.3f' %
              (n, p[0], b[0], (p[0] - b[0]) * 0.05,
               p[1] / max(p[0], 1), p[2] / max(p[0], 1),
               b[2] / max(b[0], 1)))
    print()
    print('정책: t_s %.2f, off %d, hit %d | 기준: t_s %.2f, off %d, hit %d'
          % (pol_info['t_s'], sum(r[3] for r in pol_rows),
             sum(r[4] for r in pol_rows),
             base_info['t_s'], sum(r[3] for r in base_rows),
             sum(r[4] for r in base_rows)))


if __name__ == '__main__':
    main()
