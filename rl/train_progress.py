#!/usr/bin/env python3
"""학습 추세를 에피소드 요약 CSV 에서 읽는다 (학습 중에도 안전 — 읽기만 한다).

    python3 rl/train_progress.py rl/runs/s8
    python3 rl/train_progress.py rl/runs/s8 --bin 20

원자료는 env 가 직접 남기는 `episodes.csv` 다 (`t_s,off,hit,lap_done,timeout,
ret_official,ret_shape`). v2_pbrs 계약부터 return 에 진행 정형이 섞여
monitor.csv 의 return 만으로는 페널티를 역산할 수 없기 때문에, 역산 대신
원자료를 쓴다. 공식 점수 = t_s + 5·(off+hit).

⚠️ 이 수치는 **탐색 중인 stochastic 정책**의 것이다. 채택 판정은 결정론적
정책 + 공식 심판으로만 한다 — README 8b 장.
"""
import argparse
import csv
import os
import sys


def load(run_dir):
    p = os.path.join(run_dir, 'episodes.csv')
    if not os.path.isfile(p):
        sys.exit('episodes.csv 가 없다: ' + p + '  (v2_pbrs 이전 런이면 구버전 '
                 'train_progress 로 monitor.csv 를 읽어야 한다)')
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({'t_s': float(r['t_s']),
                             'off': int(r['off']), 'hit': int(r['hit']),
                             'done': int(r['lap_done']),
                             'to': int(r['timeout']),
                             'cost': (float(r['t_s'])
                                      + 5.0 * (int(r['off']) + int(r['hit'])))})
            except (KeyError, ValueError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--bin', type=int, default=10, help='몇 랩씩 묶어 볼지')
    a = ap.parse_args()

    S = load(a.run_dir)
    if not S:
        sys.exit('에피소드가 아직 없다 (첫 랩이 끝나야 한 줄 생긴다)')
    print('%s — 에피소드 %d개 (완주 %d, 타임아웃 %d)'
          % (a.run_dir, len(S), sum(x['done'] for x in S),
             sum(x['to'] for x in S)))
    print()
    print('%-12s %7s %8s %8s %8s %4s' %
          ('랩 구간', '랩수', '랩초', '페널티', '공식점수', 'TO'))
    for i in range(0, len(S), a.bin):
        ch = S[i:i + a.bin]
        n = len(ch)
        print('%-12s %7d %8.2f %8.2f %8.2f %4d'
              % ('%d-%d' % (i + 1, i + n), n,
                 sum(x['t_s'] for x in ch) / n,
                 sum(x['off'] + x['hit'] for x in ch) / n,
                 sum(x['cost'] for x in ch) / n,
                 sum(x['to'] for x in ch)))
    print()
    last = S[-a.bin:]
    first = S[:a.bin]
    d = (sum(x['cost'] for x in last) / len(last)
         - sum(x['cost'] for x in first) / len(first))
    print('최근 %d랩 평균 공식점수 %.2f (첫 %d랩 대비 %+.2f 초) · 최고 %.2f'
          % (len(last), sum(x['cost'] for x in last) / len(last),
             len(first), d, min(x['cost'] for x in S)))
    print()
    print('※ 탐색 중 stochastic 정책의 수치다. 채택 판정은 공식 심판으로만.')


if __name__ == '__main__':
    main()
