#!/usr/bin/env python3
"""학습 추세를 monitor.csv 에서 읽는다 (학습 중에도 안전 — 읽기만 한다).

    python3 rl/train_progress.py rl/runs/s7
    python3 rl/train_progress.py rl/runs/s7 --bin 20

## 보상에서 페널티 개수를 역산할 수 있다
계약이 `r = -W(Δt + 5·off + 5·hit)`, `W=0.1` 이므로 에피소드 return 과 길이만
있으면 페널티 건수가 나온다:

    -r/W  =  t_s + 5N        (t_s = 랩 소요 초, N = 이탈+히트 건수)
    t_s   ≈  DT · l          (l = 스텝 수. 페널티 회복 틱 extra 는 여기 포함돼
                              있지 않아 t_s 를 아주 조금 과소평가한다)
    N     ≈  (-r/W - DT·l) / 5

검산(실측): l=933, r=-5.17 -> -51.7 되돌리면 t_s 46.65, N=1.01 = 이탈 1건.
`quick_gate`·심판 점수와 같은 구조라, 학습 곡선을 **공식 점수 단위**로 읽는다.

⚠️ 이 수치는 **탐색 중인 stochastic 정책**의 것이다 (SAC 는 수집 때 잡음을
넣는다). 채택 판정은 결정론적 정책 + 공식 심판으로만 한다 — README 8b 장.
"""
import argparse
import csv
import os
import sys

W = 0.1
DT = 0.05


def load(run_dir):
    p = os.path.join(run_dir, 'monitor.csv')
    if not os.path.isfile(p):
        sys.exit('monitor.csv 가 없다: ' + p)
    rows = []
    with open(p) as f:
        first = f.readline()          # '#{"t_start": ...}'
        if not first.startswith('#'):
            f.seek(0)
        for r in csv.DictReader(f):
            try:
                rows.append((float(r['r']), int(r['l']), float(r['t'])))
            except (KeyError, ValueError):
                continue
    return rows


def stats(rows):
    out = []
    for r, l, t in rows:
        t_s = DT * l
        cost = -r / W                 # = t_s + 5N
        n_pen = max(0.0, (cost - t_s) / 5.0)
        out.append({'r': r, 'l': l, 't': t, 't_s': t_s,
                    'cost': cost, 'n_pen': n_pen})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--bin', type=int, default=10, help='몇 랩씩 묶어 볼지')
    a = ap.parse_args()

    rows = load(a.run_dir)
    if not rows:
        sys.exit('에피소드가 아직 없다 (첫 랩이 끝나야 한 줄 생긴다)')
    S = stats(rows)
    print('%s — 에피소드 %d개 (학습 %.1f분)'
          % (a.run_dir, len(S), S[-1]['t'] / 60.0))
    print()
    print('%-12s %7s %8s %8s %8s' % ('랩 구간', '랩수', '랩초', '페널티', '공식점수'))
    for i in range(0, len(S), a.bin):
        ch = S[i:i + a.bin]
        n = len(ch)
        print('%-12s %7d %8.2f %8.2f %8.2f'
              % ('%d-%d' % (i + 1, i + n), n,
                 sum(x['t_s'] for x in ch) / n,
                 sum(x['n_pen'] for x in ch) / n,
                 sum(x['cost'] for x in ch) / n))
    print()
    last = S[-a.bin:]
    first = S[:a.bin]
    d = (sum(x['cost'] for x in last) / len(last)
         - sum(x['cost'] for x in first) / len(first))
    print('최근 %d랩 평균 공식점수 %.2f (첫 %d랩 대비 %+.2f 초)'
          % (len(last), sum(x['cost'] for x in last) / len(last), len(first), d))
    print('  랩초 %.2f · 페널티 %.2f건 · 최고 %.2f'
          % (sum(x['t_s'] for x in last) / len(last),
             sum(x['n_pen'] for x in last) / len(last),
             min(x['cost'] for x in S)))
    print()
    print('※ 탐색 중 stochastic 정책의 수치다. 채택 판정은 공식 심판으로만.')


if __name__ == '__main__':
    main()
