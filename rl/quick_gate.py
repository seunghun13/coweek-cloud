#!/usr/bin/env python3
"""정책 vs 잔차 0 을 **같은 조건에서 짝지어** 몇 분 만에 비교한다 (트리아지).

    . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
    cd ~/physicar_ws/coweek
    python3 rl/quick_gate.py --model rl/runs/s6/final.zip --episodes 8

## 이건 채택 근거가 **아니다**
이 프로젝트에서 오프라인 지표가 역대 최고를 찍고 배치에서 2.2~6.7 초 손해를 낸
전례가 여러 건이다 (v59 median/meanT2, v68 체커). 채택은 `eval_policy.py` 의
공식 심판 게이트로만 한다 — 기준은 `README.md` 8b 장에 미리 못박아 뒀다.

## 그래도 필요한 이유
공식 게이트는 예선·본선 두 조건에 각 30런이라 **약 70분**이다. 정책이 잔차 0보다
**나쁜지**를 가리는 데 70분을 쓸 이유가 없다. 이건 그 앞단의 몇 분짜리 체다.

## 왜 "짝지어" 비교하나
`reset(seed=S)` 가 env 의 rng 를 통째로 다시 깐다. 그래서 같은 S 를 두 arm 에
주면 **콘 배치 · 플랜트 · 스폰 위치가 완전히 동일**해진다. 본선 배치의 편차가
같은 빌드에서 페널티 130/170/200 을 만드는 이 프로젝트에서, 짝짓기는 사치가
아니라 필수다.
"""
import argparse
import sys
import os

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np

from coweek_env import CoweekResidualEnv, RES_MAX

ZERO = np.zeros(1, np.float32)


def run_arm(env, model, seeds):
    """각 seed 로 한 에피소드씩. model=None 이면 잔차 0."""
    ds = off = hit = 0.0
    res_abs, res_sum, n = 0.0, 0.0, 0
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        for _ in range(env.episode_steps):
            if model is None:
                a = ZERO
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, _r, _term, trunc, info = env.step(a)
            ds += info['ds']
            off += int(info['off'])
            hit += int(info['hit'])
            res_abs += abs(info['res'])
            res_sum += info['res']
            n += 1
            if trunc:
                break
    return {'ds': ds, 'off': int(off), 'hit': int(hit), 'n': n,
            'res_abs': res_abs / max(n, 1), 'res_mean': res_sum / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=8,
                    help='arm 당 랩 수. 랩이 약 45 초라 8 이면 두 arm 합쳐 '
                         '약 12 분이다')
    ap.add_argument('--seed0', type=int, default=90000)
    ap.add_argument('--dyn', action='store_true',
                    help='동역학 랜덤화를 켜고 잰다. **기본은 끔** — 이 도구는 '
                         '공식 게이트(참 플랜트)의 전단계 체라, 게이트와 같은 '
                         '플랜트에서 재야 예측력이 있다')
    ap.add_argument('--cone-reshuffle', type=int, default=1)
    a = ap.parse_args()

    assert os.path.isfile(a.model), '모델이 없다: ' + a.model
    from stable_baselines3 import SAC

    env = CoweekResidualEnv(seed=0, dyn=a.dyn,
                            cone_reshuffle=a.cone_reshuffle)
    model = SAC.load(a.model, device='cpu')
    seeds = [a.seed0 + k for k in range(a.episodes)]
    print('짝지은 %d 랩 × 2 arm · 동역학 랜덤화 %s · 콘 재배치 %s'
          % (a.episodes, 'ON' if a.dyn else 'OFF (참 플랜트)',
             ('%d 랩마다' % a.cone_reshuffle) if a.cone_reshuffle else 'OFF'),
          flush=True)
    try:
        pol = run_arm(env, model, seeds)
        base = run_arm(env, None, seeds)
    finally:
        env.close()

    # 페널티는 공식 계수(5 초)로 환산, 진행량은 K_PROG=1.0 s/m 그대로
    def cost(r):
        return 5.0 * (r['off'] + r['hit']) - r['ds']

    print()
    print('%-10s %8s %6s %6s %10s %10s' %
          ('', '진행 m', '이탈', '히트', '평균|잔차|', '비용(낮을수록)'))
    for tag, r in (('정책', pol), ('잔차0', base)):
        print('%-10s %8.2f %6d %6d %10.4f %10.2f'
              % (tag, r['ds'], r['off'], r['hit'], r['res_abs'], cost(r)))
    print()
    d_ds = pol['ds'] - base['ds']
    d_pen = 5.0 * ((pol['off'] + pol['hit']) - (base['off'] + base['hit']))
    print('진행량 차 %+.2f m · 페널티 차 %+.1f 초 · 순 %+.2f'
          % (d_ds, d_pen, d_ds - d_pen))
    print('평균 잔차 %+.4f m/s (범위 ±%.2f)' % (pol['res_mean'], RES_MAX))
    print()
    if d_ds - d_pen <= 0:
        print('=> 잔차 0 보다 낫지 않다. 공식 게이트에 70분을 쓰기 전에 원인부터.')
    else:
        print('=> 트리아지 통과. 이제 eval_policy.py 로 공식 게이트를 재라.')
        print('   (이 숫자는 채택 근거가 아니다 — README 8b 장)')


if __name__ == '__main__':
    main()
