#!/usr/bin/env python3
"""학습된 잔차 정책을 **공식 채점기로** 평가한다.

학습 곡선은 채택 근거가 아니다. 이 프로젝트에서 오프라인 지표가 역대 최고를
찍고도 실주행에서 손해를 낸 전례가 여러 건이다. 채택은 여기서만 한다.

    python3 rl/eval_policy.py --model rl/runs/s1/final.zip --runs 15
    python3 rl/eval_policy.py --model rl/runs/s1/final.zip --runs 15 --finals
    python3 rl/eval_policy.py --baseline --runs 15            # 잔차 0 대조군

게이트 (GPT 권고 + 우리 실측 기준선)
    Gate F  결승선 놓침       기준선 대비 증가하지 않을 것
    Gate P  페널티 합         기준선 대비 증가하지 않을 것
    Gate S  공식 점수 평균    기준선보다 낮을 것
모두 **같은 배치 안에서** 재야 한다. 배치 간 절대 비교는 못 믿는다
(같은 빌드가 세 배치에서 페널티 130/170/200 이었다).
"""
import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys

WS = '/home/physicar/physicar_ws'
ACTIVE = os.path.join(WS, 'coweek/rl/ACTIVE_MODEL')
LAP = 30.505


def drive_dist(tr):
    d = 0.0
    for i in range(1, len(tr)):
        s = math.hypot(tr[i][1] - tr[i - 1][1], tr[i][2] - tr[i - 1][2])
        if s <= 0.25:            # 그 이상은 주행이 아니라 심판 텔레포트다
            d += s
    return d


def analyse(path, tag):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    clean = [r for r in runs if drive_dist(r.get('trace') or []) <= 1.30 * LAP]
    miss = len(runs) - len(clean)
    pure = [r['score'] - r['penalty'] for r in clean]
    out = {
        'tag': tag, 'n': len(runs),
        'mean': st.mean(r['score'] for r in runs),
        'miss': miss,
        'clean_mean': st.mean(r['score'] for r in clean) if clean else float('nan'),
        'pure': st.mean(pure) if pure else float('nan'),
        'pen': sum(r['penalty'] for r in clean),
        'nopen': sum(1 for r in clean if r['penalty'] == 0),
    }
    print('%-14s n=%d | 총평균 %6.2f | 놓침 %d/%d | 놓침제외 %6.2f | '
          '순수랩 %6.2f | 페널티 %5.1f | 무페널티 %d'
          % (tag, out['n'], out['mean'], miss, out['n'], out['clean_mean'],
             out['pure'], out['pen'], out['nopen']))
    return out


def run_batch(runs, finals, log, traces, seed=None):
    """한 arm 을 돌린다. `seed` 를 주면 두 arm 이 **같은 콘 배치열**을 본다.

    seed 없이 돌리면 arm 마다 배치가 달라 비교가 짝지어지지 않는다. 같은 v64
    빌드가 세 배치에서 페널티 130/170/200 이었으므로, 짝짓기 없이 15런으로는
    웬만한 차이를 못 가른다."""
    cmd = [sys.executable, '-u', 'coweek/referee.py', '--runs', str(runs)]
    if finals:
        cmd += ['--random-cones', '--pin', '']
    if seed is not None:
        cmd += ['--seed', str(seed)]
    subprocess.run([sys.executable, 'coweek/kill_orphans.py'], cwd=WS)
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, cwd=WS, stdout=f, stderr=subprocess.STDOUT)
    if rc.returncode != 0:
        print('!! 심판이 비정상 종료했다 (rc=%d) — %s 확인'
              % (rc.returncode, log), flush=True)
    subprocess.run([sys.executable, 'coweek/kill_orphans.py'], cwd=WS)
    if not os.path.exists('/tmp/ref_traces.jsonl'):
        raise RuntimeError('심판이 트레이스를 안 남겼다 — 배치가 실패했다')
    subprocess.run(['cp', '/tmp/ref_traces.jsonl', traces])


def set_active(val):
    if val is None:
        if os.path.exists(ACTIVE):
            os.remove(ACTIVE)
    else:
        with open(ACTIVE, 'w') as f:
            f.write(val + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--baseline', action='store_true',
                    help='잔차 0 러너로 대조군을 함께 돌린다')
    ap.add_argument('--runs', type=int, default=15)
    ap.add_argument('--finals', action='store_true')
    ap.add_argument('--seed', type=int, default=1234,
                    help='두 arm 이 같은 콘 배치열을 보게 한다 (짝지은 A/B). '
                         '0 을 주면 짝짓기를 끄고 옛 방식으로 돈다')
    a = ap.parse_args()

    seed = a.seed if a.seed else None
    cond = '본선(랜덤콘)' if a.finals else '예선맵'
    print('배치 짝짓기: %s'
          % ('seed %d — 두 arm 이 동일한 배치열' % seed if seed is not None
             else 'OFF (arm 마다 다른 배치)'), flush=True)
    prev = open(ACTIVE).read().strip() if os.path.exists(ACTIVE) else None
    results = []
    try:
        if a.model:
            m = os.path.abspath(a.model)
            assert os.path.isfile(m), '모델이 없다: ' + m
            print('=== %s | 정책 %s ===' % (cond, m), flush=True)
            set_active(m)
            run_batch(a.runs, a.finals, '/tmp/eval_pol.log',
                      '/tmp/eval_pol_traces.jsonl', seed)
            results.append(analyse('/tmp/eval_pol_traces.jsonl', 'policy'))
        if a.baseline or not a.model:
            print('=== %s | 잔차 0 (= racer-v64) ===' % cond, flush=True)
            set_active('none')
            run_batch(a.runs, a.finals, '/tmp/eval_base.log',
                      '/tmp/eval_base_traces.jsonl', seed)
            results.append(analyse('/tmp/eval_base_traces.jsonl', 'baseline'))
    finally:
        set_active(prev)
        print('\nACTIVE_MODEL 복원:', prev or '(삭제 — 기본 driver.py)')

    if len(results) == 2:
        p, b = results[0], results[1]
        print()
        print('=== 게이트 (같은 배치 안 비교) ===')
        for name, key, better_low in (('Gate F 결승선 놓침', 'miss', True),
                                      ('Gate P 페널티 합', 'pen', True),
                                      ('Gate S 공식 점수', 'mean', True)):
            pv, bv = p[key], b[key]
            ok = pv <= bv if better_low else pv >= bv
            print('  %-20s 정책 %8.2f  vs  기준 %8.2f   %s'
                  % (name, pv, bv, 'PASS' if ok else 'FAIL'))
        print()
        print('  ⚠️ n=%d 는 결승선 놓침(기준선 3 %%)을 판정하기에 부족하다.'
              % p['n'])
        print('     놓침은 전용 벤치로 수백 회 재야 한다.')


if __name__ == '__main__':
    main()
