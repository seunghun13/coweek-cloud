#!/usr/bin/env python3
"""정책 vs 잔차 0 을 **같은 조건에서 짝지어** 몇 분 만에 비교한다 (트리아지).

    . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
    cd ~/physicar_ws/coweek
    python3 rl/quick_gate.py --model rl/runs/s7/final.zip --episodes 8

## 이건 채택 근거가 **아니다**
이 프로젝트에서 오프라인 지표가 역대 최고를 찍고 배치에서 2.2~6.7 초 손해를 낸
전례가 여러 건이다 (v59 median/meanT2, v68 체커). 채택은 `eval_policy.py` 의
공식 심판 게이트로만 한다 — 기준은 `README.md` 8b 장에 미리 못박아 뒀다.

## 그래도 필요한 이유
공식 게이트는 예선·본선 두 조건에 각 15런 × 2 arm 이라 **약 70분**이다. 정책이
잔차 0보다 **나쁜지**를 가리는 데 70분을 쓸 이유가 없다. 이건 그 앞단의
십수 분짜리 체다.

## 무엇을 재나 (전수검수 P0-7 반영)
종전 비용 `5·(off+hit) − 진행거리` 는 **랩 시간을 안 쟀다** — 완주끼리는
진행거리가 같아서 40초 정책과 60초 정책이 같은 점수를 받았다 (s6 의 "무차이"
가 그렇게 나온 수치다). 이제 에피소드가 공식 결승선으로 끝나고 환경이 공식
시간(`t_s`, 페널티 회복 틱 포함)을 재므로, 심판과 같은 구조의 점수를 쓴다:

    score = t_s + 5·(off + hit)          (낮을수록 좋다)

seed 별로 정책·기준선을 **짝지어** 차이 d_i = 정책_i − 기준_i 를 본다.
`reset(seed=S)` 가 env 의 rng 를 통째로 다시 깔아 같은 S 면 콘 배치·플랜트·
스폰이 완전히 동일하다. 본선 배치 편차(같은 빌드 페널티 130/170/200)를
상쇄하는 유일한 방법이다.
"""
import argparse
import statistics as st
import sys
import os

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np

from coweek_env import CoweekResidualEnv, RES_MAX

ZERO = np.zeros(1, np.float32)


def run_episode(env, model, seed):
    """seed 하나로 한 에피소드. 공식 구조의 점수를 돌려준다."""
    obs, _ = env.reset(seed=int(seed))
    off = hit = 0
    res_abs = 0.0
    n = 0
    term = trunc = False
    info = {}
    while not (term or trunc):
        if model is None:
            a = ZERO
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        off += int(info['off'])
        hit += int(info['n_hit'])      # 심판처럼 콘 개수로 (P2-5)
        res_abs += abs(info['res'])
        n += 1
    return {'t_s': info.get('t_s', 0.0),
            'off': off, 'hit': hit,
            'finished': bool(term),
            'timeout': bool(trunc and not term),
            'score': info.get('t_s', 0.0) + 5.0 * (off + hit),
            'res_abs': res_abs / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--episodes', type=int, default=8,
                    help='seed(=짝) 수. 랩이 약 50 초라 8 이면 두 arm 합쳐 '
                         '약 15 분이다')
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
    print('짝지은 %d 쌍 · 동역학 랜덤화 %s · 콘 재배치 %s (pin 없음 = 본선 분포)'
          % (a.episodes, 'ON' if a.dyn else 'OFF (참 플랜트)',
             ('%d 랩마다' % a.cone_reshuffle) if a.cone_reshuffle else 'OFF'),
          flush=True)
    pairs = []
    try:
        # seed 마다 정책 → 기준선 순으로 붙여 돈다. arm 을 통째로 나누면
        # 시뮬 상태 드리프트가 한쪽 arm 에만 실릴 수 있다.
        for s in seeds:
            pol = run_episode(env, model, s)
            base = run_episode(env, None, s)
            pairs.append((s, pol, base))
            print('  seed %d | 정책 %6.2f (t %5.1f, off %d, hit %d%s) | '
                  '기준 %6.2f (t %5.1f, off %d, hit %d%s) | d %+6.2f'
                  % (s, pol['score'], pol['t_s'], pol['off'], pol['hit'],
                     ', TIMEOUT' if pol['timeout'] else '',
                     base['score'], base['t_s'], base['off'], base['hit'],
                     ', TIMEOUT' if base['timeout'] else '',
                     pol['score'] - base['score']), flush=True)
    finally:
        env.close()

    d = [p['score'] - b['score'] for _, p, b in pairs]
    pol_to = sum(p['timeout'] for _, p, _ in pairs)
    base_to = sum(b['timeout'] for _, _, b in pairs)
    res_abs = st.mean(p['res_abs'] for _, p, _ in pairs)
    print()
    print('짝지은 평균차 %+.2f 초 (sd %.2f, n=%d) · 평균 |잔차| %.4f (범위 ±%.2f)'
          % (st.mean(d), st.stdev(d) if len(d) > 1 else 0.0, len(d),
             res_abs, RES_MAX))
    print('타임아웃(150초 안전망): 정책 %d / 기준 %d' % (pol_to, base_to))
    print()
    if pol_to > base_to:
        print('=> 정책이 타임아웃을 더 낸다 — 즉시 원인부터. 공식 게이트 가지 마라.')
    elif st.mean(d) >= 0:
        print('=> 잔차 0 보다 낫지 않다. 공식 게이트에 70분을 쓰기 전에 원인부터.')
    else:
        print('=> 트리아지 통과. 이제 eval_policy.py 로 공식 게이트를 재라.')
        print('   (이 숫자는 채택 근거가 아니다 — README 8b 장)')


if __name__ == '__main__':
    main()
