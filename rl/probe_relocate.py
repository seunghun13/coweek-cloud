#!/usr/bin/env python3
"""이탈 복귀가 **뒤로** 보낼 수 있는가 — 순수 기하, 시뮬 불필요.

    python3 rl/probe_relocate.py

## 왜
사용자가 화면에서 "돌다가 뒤로 이동하는 케이스"를 봤다(2026-08-21). 랩은
정상 완주하고 있으므로(에피소드 11개가 전부 lap_done) 후보는 **이탈 복귀
텔레포트**다. 복귀 지점을 `argmin |wp - p|` 로 찾는데, 이 트랙은 서펜타인에서
도로가 자기 자신과 나란히 붙어 있어 **잔디로 벗어난 차가 옆 구간 웨이포인트에
잡힐 수 있다.** 그러면 복귀가 전진이 아니라 후진이 된다.

심판도 같은 방식(`nearestIdx`)이라 만약 이게 실재하면 **평가에서도 일어난다** —
즉 env 의 결함이 아니라 트랙·규칙의 성질이다. 어느 쪽인지부터 가린다.
"""
import json
import math

ROUTE = ('/mnt/c/Users/정승훈/Desktop/coweek/자율주행 해커톤 정보/'
         '시뮬레이터 자료/api-dumps-AMET2026/route.json')

r = json.load(open(ROUTE))
wp = r['waypoints']
inner = r['inner']
outer = r['outer']
N = len(wp)

seg = [math.hypot(wp[(i + 1) % N][0] - wp[i][0],
                  wp[(i + 1) % N][1] - wp[i][1]) for i in range(N)]
S = [0.0]
for i in range(N - 1):
    S.append(S[-1] + seg[i])
TOTAL = S[-1] + seg[N - 1]

# 좌우 법선 (inner->outer 방향), Track.u 와 같은 정의
U = []
half = []
for i in range(N):
    vx = outer[i][0] - inner[i][0]
    vy = outer[i][1] - inner[i][1]
    L = math.hypot(vx, vy) or 1.0
    U.append((vx / L, vy / L))
    half.append(L / 2.0)


def nearest_idx(x, y):
    best, bd = 0, 1e18
    for i in range(N):
        d = (wp[i][0] - x) ** 2 + (wp[i][1] - y) ** 2
        if d < bd:
            bd, best = d, i
    return best


def darc(a, b):
    """b - a, 트랙을 감아서 부호 있는 최단 차이."""
    d = b - a
    if d < -TOTAL / 2:
        d += TOTAL
    elif d > TOTAL / 2:
        d -= TOTAL
    return d


print('트랙 길이 %.3f m, 웨이포인트 %d, 반폭 %.4f~%.4f'
      % (TOTAL, N, min(half), max(half)))
print()

# 이탈 순간의 차 위치를 흉내낸다: 중심선에서 lat 만큼 벗어난 점.
# 심판 한계는 반폭+0.12 (약 0.47). 그 언저리와 조금 더 바깥까지 훑는다.
BAD = 1.0          # 이 이상 어긋나면 "엉뚱한 구간에 잡혔다" 로 본다
for lat in (0.47, 0.55, 0.70, -0.47, -0.55, -0.70):
    rows = []
    for i in range(N):
        x = wp[i][0] + U[i][0] * lat
        y = wp[i][1] + U[i][1] * lat
        j = nearest_idx(x, y)
        err = darc(S[i], S[j])
        if abs(err) > BAD:
            rows.append((i, j, err, x, y))
    back = [t for t in rows if t[2] < 0]
    print('lat %+.2f m : 오배정 %3d / %d   그중 **후진** %d'
          % (lat, len(rows), N, len(back)))
    for i, j, err, x, y in sorted(back, key=lambda t: t[2])[:5]:
        print('    wp%-4d (%.2f,%.2f) -> wp%-4d  %+.2f m  (s %.2f -> %.2f)'
              % (i, x, y, j, err, S[i], S[j]))

print()
print('※ 후진 0 이면 복귀는 항상 전진이고, 화면의 뒤로 이동은 다른 원인이다.')
