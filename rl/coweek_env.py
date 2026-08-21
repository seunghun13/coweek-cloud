#!/usr/bin/env python3
"""코윅윅 잔차 RL 환경 — 1단계: 속도 잔차만 학습.

## 왜 속도만인가
상한 귀속 실측: V_MAX 34.6 % / 콘 0.45 그룹 26.5 % / 라이다 30.8 % 인데
**횡가속 곡선은 1.7 %** 다. 우리를 느리게 하는 것은 곡률 물리가 아니라
보수적 상한 상수들이다. 그래서 행동은 (steer, speed) 2차원이 아니라
**속도 잔차 1차원**이다. 조향 잔차는 결승선 신뢰성 게이트 통과 후에만 붙인다.

## driver.py 를 건드리지 않는다
`/speed` 퍼블리셔만 가로채 base 를 읽고, env 가 base+residual 을 대신 낸다.
residual=0 이면 **정확히 racer-v64** 다. 실차에서 이상하면 잔차만 끄면 된다.
(driver.py 에는 계측 dict `last_tick` 과 텔레포트 카운터 `teleport_n` 만
추가돼 있고, 둘 다 driver 안에서 아무도 읽지 않는 순수 대입이다.)

## 계약 (2026-08-21 전수검수 반영 — `official_cost_v1` / `referee_finish_v1`)
- **종료 = 공식 결승선** (P0-1): 심판과 같은 판정(앞점 선분 교차 + 어드밴스
  N/2)을 그대로 이식했다 (`FinishDetector`). 누적 arc 는 진단값일 뿐이다.
  선분 끝점 바깥으로 지나가면 랩이 인정되지 않고 **계속 돈다** — 실측된
  1순위 병목(결승선 놓침, 1회 ≈ 50초)이 이제 학습에 존재한다.
- **완주 = terminated** (P0-2): 목표 달성 종료라 SB3 가 부트스트랩하지
  않는다. 150 초 안전망만 truncated 다.
- **보상 = 공식 비용 그대로** (P0-6):
      r = -W · (Δt + 5·I_off + 5·I_cone)
  진행 보상·배리어·스무딩 항은 전부 뺐다. K_PROG 논증은 할인 때문에
  성립하지 않았고(어떤 스칼라도 Σγᵗ·Δs 를 상수로 못 만든다), 결승선 놓침을
  모델링하는 순간 raw 진행 보상은 "추가 한 바퀴 30.5 m = 30.5 초 이득"으로
  실제 50 초 하방을 상쇄해 버린다. Δt 는 페널티 회복에 쓴 추가 틱까지 센다
  (심판의 시계는 그동안에도 돈다).
- **보상용 상태는 항상 새 상태** (P0-3): `/state` 폴에 seq 를 매기고, 행동
  발행 이전 seq 의 상태는 쓰지 않는다. 폴 실패·vehicle 누락은 (0,0,0) 으로
  대체하지 않고 세며, 새 상태가 제때 안 오면 **크게 죽는다** — 낡은 상태로
  전이를 쌓는 것이 조용한 실패의 뿌리다.
- **텔레포트 후 관측은 새 장면** (P0-4): 페널티 텔레포트 뒤 새 카메라
  프레임이 올 때까지 base 로만 달리고(배포의 잔차 0 구간과 동일), 관측
  히스토리와 `prev_a` 를 리셋한다. driver v34c 가 텔레포트를 감지하면
  (`teleport_n` 증가) 같은 리셋을 미러링한다 — 배포 러너와 동일 규칙.
- **학습 콘 분포 = 평가 콘 분포** (P0-8): 랜덤 배치를 쓰되 **cone6 만 예선맵
  자리에 고정**한다. 이건 대회 규칙이 아니라 측정 전략이다 — cone6 가 아직
  미해결 병목이라 흩어버리면 성적이 실제보다 좋아 보인다(사용자 판단).
  중요한 것은 학습과 평가가 **같은 분포**를 본다는 것이고, 지금 둘 다 그렇다.

## 텔레포트 + **콘 복원**
심판은 이탈·콘히트 때 차를 중심선 0.3 m 앞으로 옮기고 **맞은 콘을 원위치**
시킨다. env 도 **둘 다** 한다.

⚠️ **콘 복원을 빠뜨려 22k 스텝 학습 하나를 통째로 버렸다 (2026-08-21).**
6개 중 5개가 밀려나고 cone4 는 1.78 m 굴러 누웠는데, 히트 판정이 거리
기준이라 그 22k 스텝 내내 히트를 **0건** 보고했다. 자세한 건 `Cones` 참조.

## 동역학 랜덤화 (`dyn=True`, 기본값 — 사용자 승인 2026-08-21)
이 시뮬의 구동계는 **속도 소스**라 감속이 폴 1개 안에 끝난다(≥4.5 m/s² 실측).
**감속이 공짜인 플랜트에서 속도 잔차를 학습하면 정책은 반드시 그걸 착취한다.**
그래서 에피소드마다 가감속 한계·구동 지연을 다시 뽑는다. 근거와 범위는
`dyn.py` (조향·관측 지연 축은 전수검수로 제거됨), 검증은 `test_dyn.py`.

**평가·배포에서는 끈다.** `eval_policy.py` 는 공식 심판을 쓰고
`run_policy.py` 는 참 플랜트에 그대로 낸다. `dyn=False` 면 이 파일의 동작이
랜덤화 도입 전과 **한 비트도 다르지 않다** (`test_env_dyn.py` 가 확인한다).
"""
import json
import math
import os
import random
import sys
import threading
import time
import urllib.request

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from std_msgs.msg import Float64

import driver as drv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dyn import Dyn                       # noqa: E402  (동역학 도메인 랜덤화)

BASE = 'http://localhost/sim/api'
ROUTE = ('/mnt/c/Users/정승훈/Desktop/coweek/자율주행 해커톤 정보/'
         '시뮬레이터 자료/api-dumps-AMET2026/route.json')

# ── 채점 상수 (공식 스크립트와 동일) ──
OFF_MARGIN = 0.12          # 이탈: |e| > 반폭 + 0.12  (차량 중심 한 점)
FRONT_M = 0.12             # 결승선 판정용 앞점 (referee.py FRONT)
CONE_HIT_R = 0.1525        # 차 반폭 0.0975 + 콘 반경 0.055 (스폰 여유 계산용)
TELE_AHEAD = 0.30          # 페널티 후 중심선 0.3 m 앞으로
CONE_MOVE_TOL = 0.01       # 히트 판정: 콘이 1 cm 이상 움직였는가 (심판과 동일)
CONE_Z_TOL = 0.04          # 〃 4 cm 이상 내려앉았는가 (넘어짐)
# 심판은 콘마다 **latch + rearm 1 초**를 둔다 (referee.py 헤더). 없으면 접촉
# 한 번이 복원이 전파될 때까지 여러 스텝에서 반복 계상돼 페널티·텔레포트가
# 겹친다. 같은 값을 쓴다.
CONE_REARM_S = 1.0
SPAWN_CONE_CLEAR = 0.35    # 리셋 스폰이 콘 위에 떨어지지 않게

# 콘 정본 배치. **심판의 기준이 아니다** — 심판은 각 런 시작 시점의 월드를
# 기준으로 삼으므로(`snapshot_base`), 월드가 오염된 채 심판을 돌리면 틀어진
# 위치를 정상으로 굳혀버린다. 그래서 정본은 이 파일에서 온다.
STATE_JSON = ('/mnt/c/Users/정승훈/Desktop/coweek/자율주행 해커톤 정보/'
              '시뮬레이터 자료/api-dumps-AMET2026/state.json')

# ── 보상 계수 (`official_cost_v1`) ──
W_SCALE = 0.1
P_OFF = 5.0
P_CONE = 5.0
# K_PROG / 배리어 / 스무딩 계수는 전수검수 P0-6 으로 제거됐다 (모듈 독스트링).

# ── 에피소드 ──
DT = 1.0 / drv.RATE_HZ     # 0.05 s
# 워밍업 = 텔레포트 직후 last_tick·새 프레임을 채우는 시간. **차는 세워 둔다**
# (P1-7): 베이스로 미리 달리는 "숨은 주행 구간"은 심판이 시간을 재는 구간을
# 정책에게 가리고, 워밍업이 없는 배포(run_policy.py)와 어긋난다. 정지 출발은
# 심판과 같은 진짜 상태다.
WARMUP_S = 0.25
EPISODE_S = 8.0
RES_MAX = 0.35             # 속도 잔차 범위 [-RES_MAX, +RES_MAX] (대칭)
#   대칭으로 두는 이유: action 0 이 정확히 잔차 0 = racer-v64 가 된다.
#   SAC 초기 정책이 0 근처이므로 학습 시작점이 검증된 베이스와 같고,
#   zero residual 이 v64 를 재현하는지 그대로 검증할 수 있다.
V_HARD = 1.40              # 안전 상한. 정책이 이 위로 못 간다

CAP_NAMES = ('vmax', 'alat', 'cone_slow', 'cone_near', 'arc_short',
             'checker', 'lidar1', 'lidar2', 'sharp', 'crawl')
HIST = 3

# 계약 버전 — 모델 매니페스트와 평가기가 대조한다. 종료·보상·관측의 의미가
# 바뀌면 반드시 올릴 것 (기존 모델·버퍼가 조용히 오작동하는 것을 막는다).
CONTRACT = 'official_cost_v1+referee_finish_v1'


def api(path, payload=None, timeout=10):
    req = urllib.request.Request(BASE + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header('Content-Type', 'application/json')
        req.method = 'POST'
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def cap_onehot(lt):
    """이번 틱의 speed 를 만든 상한을 판정한다 (driver.py 의 min() 순서 그대로).

    같은 0.45 라도 콘 회피인지 결승 밴드인지 정책이 구분하지 못하면 같은
    행동을 내게 된다. 그래서 원인을 one-hot 으로 명시해 준다."""
    v = np.zeros(len(CAP_NAMES), np.float32)
    sp, eps = lt['speed'], 1e-3
    kap = abs(math.tan(lt['steer'])) / drv.WHEELBASE_M
    alat = drv.V_MAX if kap < 1e-6 else min(drv.V_MAX, math.sqrt(drv.A_LAT / kap))
    idx = CAP_NAMES.index
    if lt['crawl']:
        v[idx('crawl')] = 1
    elif lt['fmin'] < drv.LID_PREBRAKE2 and abs(sp - drv.LID_PREBRAKE2_V) < eps:
        v[idx('lidar2')] = 1
    elif lt['fmin'] < drv.LID_PREBRAKE1 and abs(sp - drv.LID_PREBRAKE1_V) < eps:
        v[idx('lidar1')] = 1
    elif lt['checker'] and abs(sp - 0.45) < eps:
        v[idx('checker')] = 1
    elif (lt['mode'] in ('dodge', 'hold') or lt['lbias'] != 0.0
          or (lt['mode'] == 'arc' and lt['dodging'])) and abs(sp - drv.CONE_SLOW) < eps:
        v[idx('cone_slow')] = 1
    elif lt['carea'] >= drv.CONE_ENGAGE_AREA and abs(sp - 0.6) < eps:
        v[idx('cone_near')] = 1
    elif lt['mode'] == 'arc' and lt['arc_safe'] < 0.60 and abs(sp - 0.50) < eps:
        v[idx('arc_short')] = 1
    elif lt['sharp'] and abs(sp - 0.35) < eps:
        v[idx('sharp')] = 1
    elif alat < drv.V_MAX - eps and abs(sp - alat) < eps:
        v[idx('alat')] = 1
    elif abs(sp - drv.V_MAX) < eps:
        v[idx('vmax')] = 1
    return v


def obs_row(lt, prev_a=0.0):
    """정책이 보는 한 틱. **그라운드트루스는 하나도 없다.**

    절대좌표·오도메트리는 물론이고 route.json 유래 값(경계까지 여유 등)도
    넣지 않는다 — 실차에 없는 정보다. 전부 베이스 제어기가 카메라·라이다에서
    이미 뽑아내는 값이거나 자기 명령 이력이다.

    `prev_a` = **직전 행동**(잔차, [-1,1] 정규화). 자기 명령 이력이다.
    `lt['speed']` 는 **base 명령**이라 잔차가 얹히기 전 값이므로 대체가 안 된다.
    ⚠️ `run_policy.py` 가 **똑같이** 채워야 한다. 학습과 배포가 다른 관측을
    쓰면 조용히 성능만 무너진다."""
    age = 0.0
    if lt['frame_t'] is not None:
        age = min(1.0, time.monotonic() - lt['frame_t'])
    row = np.array([
        lt['speed'] / drv.V_MAX,
        lt['steer'] / drv.MAX_STEER,
        np.clip(lt['aim'], -2, 2),
        np.clip(lt['carea'] / 5000.0, 0, 4),
        lt['arc_safe'] / 0.96,
        min(lt['fmin'], 4.0) / 4.0,
        lt['lbias'],
        float(lt['pass_hold']) / 8.0,
        lt['pass_side'],
        min(lt['ncone'], 3) / 3.0,
        age / 0.35,
        float(lt['green']),
        float(np.clip(prev_a, -1.0, 1.0)),
    ], np.float32)
    return np.concatenate([row, cap_onehot(lt)])


DEFAULT_LT = {
    'ticks': 0, 'green': False, 'speed': 0.0, 'steer': 0.0, 'aim': 0.0,
    'mode': 'wait', 'carea': 0.0, 'fmin': drv.LID_RANGE_MAX, 'lbias': 0.0,
    'dodging': False, 'sharp': False, 'crawl': False, 'checker': False,
    'arc_safe': 1.0, 'pass_hold': 0, 'pass_side': 0.0, 'ncone': 0,
    'band_px': 0, 'frame_t': None, 'teleport_n': 0,
}
OBS_ROW_DIM = 13 + len(CAP_NAMES)   # 12 관측 + 직전 행동 1 + 상한 one-hot


class Track:
    """진행량·이탈 계산 전용. **정책은 이 값을 보지 못한다** (무상태 유지)."""

    def __init__(self, path=ROUTE):
        r = json.load(open(path))
        self.wp = np.array(r['waypoints'], float)
        inner = np.array(r['inner'], float)
        outer = np.array(r['outer'], float)
        d = np.linalg.norm(np.diff(self.wp, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        self.total = float(self.s[-1] + np.linalg.norm(self.wp[0] - self.wp[-1]))
        self.half = np.linalg.norm(outer - inner, axis=1) / 2.0
        v = outer - inner
        self.u = v / np.linalg.norm(v, axis=1)[:, None]
        # 결승 선분 (심판과 동일: inner[0] ~ outer[0]) + 심판의 N
        self.inner0 = (float(inner[0, 0]), float(inner[0, 1]))
        self.outer0 = (float(outer[0, 0]), float(outer[0, 1]))
        self.N = len(self.wp) - 1
        # ⚠️ route.json 은 **닫힌 루프**다 — wp[N] 이 wp[0] 과 완전히 같은
        # 중복점이다 (실측 closing gap 0.000000). 그래서 np.roll 로 그냥
        # 계산하면 yaw[N] = atan2(0,0) = 0 (동쪽) 이 되는데, 그 지점의 진짜
        # 트랙 헤딩은 -1.5601 rad (남쪽) 이라 **89.4° 오차**다. 결승선 바로
        # 그 자리라, 텔레포트가 그 인덱스를 고르면 차를 트랙과 거의 수직으로
        # 놓는다 — 1순위 병목 구간의 학습 분포가 통째로 오염된다.
        # 심판은 fwd_index 를 `% N` 으로 돌려 중복점을 절대 쓰지 않는다.
        nxt = np.roll(self.wp[:self.N], -1, axis=0)
        yaw = np.arctan2(nxt[:, 1] - self.wp[:self.N, 1],
                         nxt[:, 0] - self.wp[:self.N, 0])
        self.yaw = np.concatenate([yaw, yaw[:1]])   # 중복점은 wp[0] 의 헤딩

    def off_track(self, x, y):
        """심판 `off_track()` 의 1:1 이식 — **방사거리**로 잰다.

        ⚠️ 종전에는 `|e|`(법선 사영 횡오차)로 판정했는데, 심판은 최근접
        웨이포인트까지의 **유클리드 거리**를 본다. 사영은 항상 그 거리 이하라
        env 가 **한 방향으로만 관대**했다 (실측: 관대한 폭 중앙 2.4 mm,
        p90 4.3 cm, 최대 31 cm, env 가 더 엄격한 표본 0/1032). 이 프로젝트의
        실측 이탈 초과분이 **중앙값 7 mm** 라, 심판이 +5 를 주는 이탈의 상당수를
        보상이 무벌로 넘겼다는 뜻이다 — 보상이 유일한 학습 신호이므로 정책은
        심판이 처벌하는 라인을 학습하고 배포에서 그대로 반납한다.
        """
        d2 = ((self.wp[:self.N, 0] - x) ** 2 + (self.wp[:self.N, 1] - y) ** 2)
        i = int(np.argmin(d2))
        return math.sqrt(float(d2[i])) > float(self.half[i]) + OFF_MARGIN

    def fwd_index(self, i, dist):
        """심판 `fwd_index()` 1:1 — 중복 종점을 쓰지 않도록 `% N` 으로 돈다."""
        d, N = 0.0, self.N
        start = i
        while d < dist:
            j = (i + 1) % N
            d += float(np.linalg.norm(self.wp[j] - self.wp[i]))
            i = j
            if i == start:
                break
        return i

    def nearest_idx_near(self, x, y, from_idx, back, fwd):
        """심판 `nearest_idx_near()` 1:1 (히트 텔레포트에서 쓴다)."""
        N = self.N
        best_i, best_d = from_idx, 1e18
        for k in range(-back, fwd + 1):
            i = (from_idx + k) % N
            d = (self.wp[i, 0] - x) ** 2 + (self.wp[i, 1] - y) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def locate(self, x, y):
        """(가장 가까운 인덱스, arc 위치 s, 부호 있는 횡오차, 반폭)

        s 는 **선분 위로 사영**해서 낸다. 웨이포인트 간격이 약 0.1 m 라
        s[i] 를 그대로 쓰면 진행량이 0 / 0.1 / 0.2 로 계단이 되고,
        20 Hz 한 스텝의 실제 진행(0.03~0.05 m)이 대부분 0 으로 뭉개진다.
        """
        d2 = (self.wp[:, 0] - x) ** 2 + (self.wp[:, 1] - y) ** 2
        i = int(np.argmin(d2))
        n = len(self.wp)
        best_s, best_d2 = float(self.s[i]), float(d2[i])
        # 앞뒤 두 선분에 사영해 더 가까운 쪽을 택한다
        for a in (i - 1, i):
            b = (a + 1) % n
            if a < 0:
                a = n - 1
            ax, ay = self.wp[a]
            bx, by = self.wp[b]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1e-12:
                continue
            t = ((x - ax) * vx + (y - ay) * vy) / L2
            t = min(1.0, max(0.0, t))
            px, py = ax + t * vx, ay + t * vy
            dd = (px - x) ** 2 + (py - y) ** 2
            if dd <= best_d2:
                best_d2 = dd
                seg = float(self.s[b] - self.s[a]) if b > a else \
                    float(self.total - self.s[a])
                best_s = float(self.s[a]) + t * seg
        e = ((x - self.wp[i, 0]) * self.u[i, 0]
             + (y - self.wp[i, 1]) * self.u[i, 1])
        return i, best_s % self.total, float(e), float(self.half[i])

    def ds(self, s0, s1):
        d = s1 - s0
        if d < -self.total / 2:
            d += self.total
        elif d > self.total / 2:
            d -= self.total
        return d


class FinishDetector:
    """심판 결승선 판정의 1:1 이식 (referee.py `Referee`).

    종료가 누적 arc 거리였을 때는 "결승 선분 **끝점 바깥**으로 지나가 랩이
    인정되지 않는" 사고 — 실측된 1순위 병목, 1회 ≈ 50초 — 가 학습에 존재할
    수 없었다 (전수검수 P0-1). 판정은 심판과 동일하게만 한다:

      앞점(pose + 0.12 m, heading 방향)의 이동 선분이 결승 선분
      (inner[0]~outer[0], 폭 0.70 m)과 교차 **그리고** 누적 웨이포인트
      어드밴스 > N/2.

    텔레포트(앞점 점프 > 0.5 m 또는 `note_teleport()`)는 prev 를 지운다 —
    순간이동 선분이 결승선을 가로지르는 가짜 크로싱을 막는 심판의 규칙
    그대로다.
    """

    def __init__(self, track):
        self.tr = track
        self.reset()

    def reset(self):
        self.prev = None
        self.last_idx = None
        self.adv = 0

    def note_teleport(self):
        # 심판도 페널티 텔레포트 직후 prev 를 지운다 (move_to_center 뒤
        # `self.prev = None`)
        self.prev = None

    def _nearest(self, x, y):
        d2 = ((self.tr.wp[:self.tr.N, 0] - x) ** 2
              + (self.tr.wp[:self.tr.N, 1] - y) ** 2)
        return int(np.argmin(d2))

    def _nearest_near(self, x, y, from_idx, back, fwd):
        N = self.tr.N
        best_i, best_d = from_idx, 1e18
        for k in range(-back, fwd + 1):
            i = (from_idx + k) % N
            d = ((self.tr.wp[i, 0] - x) ** 2 + (self.tr.wp[i, 1] - y) ** 2)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _crossed(self, f):
        if self.prev is None:
            return False
        a, b = self.tr.inner0, self.tr.outer0
        px, py = self.prev
        d = (f[0] - px) * (b[1] - a[1]) - (f[1] - py) * (b[0] - a[0])
        if abs(d) < 1e-12:
            return False
        t = ((a[0] - px) * (b[1] - a[1]) - (a[1] - py) * (b[0] - a[0])) / d
        u = ((a[0] - px) * (f[1] - py) - (a[1] - py) * (f[0] - px)) / d
        return 0 <= t <= 1 and 0 <= u <= 1

    def update(self, x, y, yaw):
        """이번 pose 로 판정을 한 틱 진행. 완주면 True.

        심판 `evaluate()` 와 같은 순서다: 크로싱을 먼저 보고, 완주면
        어드밴스 갱신 없이 즉시 반환한다."""
        f = (x + math.cos(yaw) * FRONT_M, y + math.sin(yaw) * FRONT_M)
        if self.prev is not None and math.hypot(f[0] - self.prev[0],
                                                f[1] - self.prev[1]) > 0.5:
            self.prev = None
        crossed = self._crossed(f)
        self.prev = f
        if crossed and self.adv > self.tr.N / 2:
            return True
        i = (self._nearest(x, y) if self.last_idx is None
             else self._nearest_near(x, y, self.last_idx, 10, 15))
        if self.last_idx is not None:
            d = i - self.last_idx
            if d > self.tr.N / 2:
                d -= self.tr.N
            elif d < -self.tr.N / 2:
                d += self.tr.N
            self.adv += d
        self.last_idx = i
        return False


class _SpeedCapture:
    """driver 의 /speed 퍼블리시를 가로챈다. 유일한 개입 지점."""

    def __init__(self):
        self.base = 0.0
        self.n = 0

    def publish(self, msg):
        self.base = float(msg.data)
        self.n += 1


class Cones:
    """콘 정본 포즈를 들고 있다가 **심판과 같은 기준으로** 히트를 재고,
    맞은 콘을 **원위치로 되돌린다.**

    ## 이게 없어서 22k 스텝 학습 하나를 통째로 버렸다 (2026-08-21, 사용자가
    ## 화면을 보고 발견)
    심판은 히트 시 콘을 원위치시키는데 env 는 차만 옮기고 콘은 그대로 뒀다.
    22k 스텝 뒤 6개 중 **5개가 밀려나 있었고 cone4 는 1.78 m 굴러 누워
    있었다**(z 0.19 → 0.09). 정책은 그동안 트랙이 아닌 것을 학습하고 있었다.
    CLAUDE.md 에 **이미 같은 사고가 적혀 있었다** (finish_bench 콘 복원 누락으로
    측정 4회 무효): *"심판을 거치지 않는 자작 측정 도구는 심판이 해주던 월드
    상태 복원을 반드시 스스로 해야 한다."* env 도 그런 도구다.

    ## 히트를 거리가 아니라 **변위**로 재는 이유
    옛 기준 `hypot(차중심 − 콘) < 0.1525` 는 차 **중심**과의 거리라 차 길이
    방향(앞·뒤 범퍼)으로 치는 접촉을 통째로 놓친다 — 차는 폭보다 길다.
    그 기준은 콘 5개가 밀려나는 22k 스텝 동안 히트를 **0건** 보고했다.
    심판은 처음부터 변위(1 cm xy / 4 cm z)로 판정하고 있었다. 그쪽이 맞다.
    """

    ## 배치 랜덤화
    ##   생성기는 **심판의 `random_cone_layout` 을 그대로** 쓴다 — 학습과 평가가
    ##   같은 분포를 쓰지 않으면 게이트가 무엇을 재는지 알 수 없다. 새 배치
    ##   생성기를 발명하지 않는다 (이 프로젝트에서 그 부류는 전부 기각됐다).
    ##   기본 pin 은 **cone6** — 대회 규칙이 아니라 **우리 측정 전략**이다
    ##   (사용자 판단, 2026-08-22 재확인). 실제 본선은 예선에서 콘 위치가
    ##   바뀔 뿐이라 6개가 다 움직일 수 있다. 그런데 우리는 아직 cone6 를
    ##   통과하지 못하고 있고, **그 자리(결승 1.40 m 전, 코너 출구)가 최대
    ##   병목**이라 랜덤으로 흩어버리면 병목이 사라져 성적이 실제보다 좋아
    ##   보인다. 그래서 그 콘만 예선맵 자리에 묶어 둔다.
    ##   전수검수 P0-8 의 요지("학습과 평가가 **같은 분포**여야 한다")는 그대로
    ##   유효하고, `eval_policy.py --finals` 도 같은 분포(pin=cone6)를 쓴다.
    ##   cone6 문제가 풀리면 양쪽 다 `pin=()`(6개 전부 랜덤)으로 옮긴다.

    def __init__(self, path=STATE_JSON, pin=('cone6',)):
        self.base = {n: dict(p) for n, p in
                     json.load(open(path))['objects'].items()
                     if n.startswith('cone')}
        if not self.base:
            raise RuntimeError('콘 정본을 못 읽었다: ' + path)
        self.home = {n: dict(p) for n, p in self.base.items()}   # 예선맵 원본
        self.pin = tuple(pin)
        self._ref = None            # 심판 모듈 (지연 임포트)
        self._track = None          # 심판 Track (/route 를 부르므로 필요할 때만)
        self.restore_all()

    def reshuffle(self, rng, settle=0.6):
        """`pin` 을 뺀 콘을 심판과 같은 규칙으로 다시 뿌리고 정본을 갱신한다.

        정본은 **실제로 안착한 위치**를 다시 읽어 잡는다 (심판 `snapshot_base`
        와 같은 방식). 의도한 좌표를 그대로 정본으로 삼으면 안착 오차가
        곧바로 가짜 히트가 된다."""
        if self._ref is None:
            import referee as _ref               # noqa: E402 (지연 임포트)
            self._ref = _ref
            self._track = _ref.Track()
        # 심판 생성기는 **전역 random** 을 쓴다. env 의 rng 로 씨를 주되
        # 전역 스트림은 저장·복원한다 — 매 랩 전역 시드를 갈아엎으면 같은
        # 프로세스의 다른 소비자(SB3 등)에게 조용히 영향을 준다.
        _st = random.getstate()
        random.seed(int(rng.integers(0, 2 ** 31 - 1)))
        try:
            free = [n for n in self.home if n not in self.pin]
            pinned_s = [self._track.arc_of(self.home[n]['x'], self.home[n]['y'])
                        for n in self.pin if n in self.home]
            layout = self._ref.random_cone_layout(self._track, free,
                                                  pinned_s=pinned_s)
        finally:
            random.setstate(_st)
        for n in self.pin:                       # 고정 콘은 항상 예선맵 자리
            if n in self.home:
                h = self.home[n]
                api('/models/%s/pose' % n,
                    {'x': h['x'], 'y': h['y'], 'yaw': h.get('yaw', 0.0)})
        for n, (x, y, _s, _off, yaw) in layout.items():
            api('/models/%s/pose' % n, {'x': x, 'y': y, 'yaw': yaw})
        time.sleep(settle)
        # 안착한 실제 위치를 새 정본으로
        objs = {n: p for n, p in (api('/state').get('objects') or {}).items()
                if n.startswith('cone')}
        for n in self.base:
            if n in objs:
                self.base[n] = dict(objs[n])
        return len(layout)

    def restore(self, name):
        b = self.base[name]
        api('/models/%s/pose' % name,
            {'x': b['x'], 'y': b['y'], 'yaw': b.get('yaw', 0.0)})

    def restore_all(self):
        for n in self.base:
            self.restore(n)

    def restore_home(self):
        """예선맵 원본 배치로 되돌린다. 학습이 끝나면 월드를 이 상태로 넘긴다
        — 심판·오프라인 도구가 전부 이 배치를 전제한다."""
        self.base = {n: dict(p) for n, p in self.home.items()}
        self.restore_all()

    def displaced(self, objects):
        """정본에서 벗어난 콘 이름들. 심판 `displaced()` 와 같은 판정."""
        out = []
        for n, b in self.base.items():
            p = objects.get(n)
            if not p:
                continue
            if (math.hypot(p['x'] - b['x'], p['y'] - b['y']) > CONE_MOVE_TOL
                    or abs(p.get('z', 0.0) - b.get('z', 0.0)) > CONE_Z_TOL):
                out.append(n)
        return out

    def clearance(self, x, y):
        """정본 기준 최근접 콘까지 거리. 콘은 항상 복원되므로 실제와 같다."""
        return min(math.hypot(x - b['x'], y - b['y'])
                   for b in self.base.values())


class _StatePoller(threading.Thread):
    """월드 포즈를 별도 스레드에서 폴링한다.

    20 Hz 제어 예산 50 ms 안에서 HTTP 왕복(~1 ms 실측 p50)을 그래도 본 루프
    밖에 두는 이유는 가끔 오는 느린 응답 하나가 드라이버 틱을 밀면 안 되기
    때문이다. **보상 계산 전용**이고 관측에는 안 들어간다.

    전수검수 P0-3 반영:
    - pose 마다 **seq** 를 매긴다. env 는 "이번 행동 이후에 새로 받은 상태"만
      쓴다 (`_fresh_pose`).
    - `/state` 실패·`vehicle` 필드 누락은 **세고**(fail_n), (0,0,0) 으로
      대체하지 않는다 — 이전 pose 는 이전 seq 그대로 남아 재사용이 차단된다.
    """
    daemon = True

    def __init__(self, hz=25.0):
        super().__init__()
        self.dt = 1.0 / hz
        self.lock = threading.Lock()
        self.pose = None
        self.seq = 0
        self.fail_n = 0
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                st = api('/state', timeout=5)
                v = st.get('vehicle')
                if (not isinstance(v, dict) or 'x' not in v or 'y' not in v
                        or 'yaw' not in v):
                    raise ValueError('vehicle 필드 누락')
                # 콘은 dict 통째로 넘긴다 — z 가 있어야 "넘어짐"을 본다
                objs = {n: p for n, p in (st.get('objects') or {}).items()
                        if n.startswith('cone')}
                with self.lock:
                    self.pose = (float(v['x']), float(v['y']),
                                 float(v['yaw']), objs)
                    self.seq += 1
            except Exception:
                self.fail_n += 1
            time.sleep(self.dt)

    def get(self):
        with self.lock:
            return self.pose

    def get_seq(self):
        with self.lock:
            return self.pose, self.seq


class CoweekResidualEnv(gym.Env):
    """base(v64) 위에 속도 잔차만 얹는 환경."""

    metadata = {'render_modes': []}

    def __init__(self, seed=0, episode_s=EPISODE_S, warmup_s=WARMUP_S,
                 start_lo=0, start_hi=None, log=None, dyn=True, dyn_ranges=None,
                 cone_reshuffle=1, cone_pin=('cone6',),
                 lap_episode=True, max_episode_s=180.0,   # = referee.TIME_LIMIT
                 start_at_line=True):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.track = Track()
        self.finish = FinishDetector(self.track)
        # 에피소드 = **한 바퀴**, 종료 = 공식 결승선 (P0-1). `episode_s` 는
        # lap_episode=False(배선 테스트용 슬라이스)일 때만 쓰이고, 랩 모드에선
        # max_episode_s 가 "차가 멈추거나 결승을 계속 놓쳐 영원히 안 끝나는"
        # 경우를 막는 안전망일 뿐이다.
        self.lap_episode = bool(lap_episode)
        self.max_episode_s = float(max_episode_s)
        self.episode_steps = int((max_episode_s if lap_episode
                                  else episode_s) / DT)
        # 랩은 **출발선에서 출발선까지**여야 화면·심판과 경계가 일치한다.
        self.start_at_line = bool(start_at_line)
        self.warmup_steps = int(warmup_s / DT)
        self.start_lo = start_lo
        self.start_hi = start_hi if start_hi is not None else len(self.track.wp) - 1
        self.log = log

        # 동역학 도메인 랜덤화. 에피소드마다 플랜트를 다시 뽑는다.
        # 끄면(dyn=False) 명령이 한 비트도 안 바뀐다 — 그게 참 플랜트다.
        self.dyn = Dyn(DT, enabled=dyn, ranges=dyn_ranges)
        # 콘 정본 + 복원. 생성 시점에 월드를 한 번 깨끗하게 만든다.
        self.cones = Cones(pin=cone_pin)
        # N 에피소드마다 콘 재배치 (0 = 예선맵 고정). 과적합 방지용.
        self.cone_reshuffle = int(cone_reshuffle)
        self._ep = 0

        if not rclpy.ok():
            rclpy.init()
        self.node = drv.CoweekDriver()
        # 초록불 대기를 건너뛴다 — 학습은 주행 구간만 본다
        self.node.green_latched = True
        self._real_speed = self.node.pub_speed
        self.cap = _SpeedCapture()
        self.node.pub_speed = self.cap
        # 조향은 건드리지 않는다 — 조향 랜덤화 축은 전수검수로 제거됐다
        # (dyn.py 독스트링). /steering 은 드라이버가 직접 낸다 = 참 플랜트.

        self.poll = _StatePoller()
        self.poll.start()

        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0,
                                            (OBS_ROW_DIM * HIST,), np.float32)
        self._hist = []
        self._prev_a = 0.0
        self._hit_state = {}
        self._tp_seen = 0
        self._time_s = 0.0

    # ── 내부 ────────────────────────────────────────────────────────
    def _spin_one_tick(self):
        """드라이버 타이머가 한 번 돌 때까지 spin. 20 Hz 벽시계에 묶인다.

        틱이 1 초 동안 없으면 **크게 죽는다** (P1-8) — 그 스텝을 0.05 초짜리
        전이로 학습하는 것이 조용한 실패이기 때문이다. 카메라가 1 초 이상
        멎은 것은 환경 고장이지 학습할 상태가 아니다."""
        n0 = self.cap.n
        t0 = time.monotonic()
        while self.cap.n == n0 and time.monotonic() - t0 < 2.0:
            rclpy.spin_once(self.node, timeout_sec=0.005)
        if self.cap.n == n0:
            raise RuntimeError('드라이버 틱이 2 초 동안 없다 — 환경 고장. '
                               '이 전이를 학습하지 않고 죽는다 (P1-8). '
                               '체크포인트에서 --resume 으로 이어갈 것')
        return True

    def _publish(self, speed):
        """플랜트(가감속 한계 + 구동 지연)를 통과시켜 실제로 낸다.

        랜덤화가 꺼져 있으면 `speed` 가 그대로 나간다."""
        out = self.dyn.speed(float(speed))
        self._real_speed.publish(Float64(data=out))
        return out

    def _features(self):
        lt = getattr(self.node, 'last_tick', None)
        if lt is None:
            lt = dict(DEFAULT_LT)
        return lt

    def _obs_row(self, lt):
        # 직전 행동을 함께 넣는다 (obs_row 독스트링 참조).
        # step() 은 self._prev_a 를 **관측을 만들기 전에** 갱신하므로
        # 여기 들어가는 값은 "방금 낸 행동"이다. reset()·텔레포트 직후엔 0 이다.
        return obs_row(lt, self._prev_a)

    def _stack(self, row):
        # 관측 지연 축은 제거됐다 (P1-2) — 항상 현재 row 다.
        self._hist.append(row)
        while len(self._hist) < HIST:
            self._hist.insert(0, row)
        self._hist = self._hist[-HIST:]
        return np.concatenate(self._hist).astype(np.float32)

    def _teleport_to(self, i, lat=0.0, dyaw=0.0, back=0.0):
        wx, wy = self.track.wp[i]
        ux, uy = self.track.u[i]
        base_yaw = float(self.track.yaw[i])
        yaw = base_yaw + dyaw
        api('/pose', {'x': float(wx + ux * lat - math.cos(base_yaw) * back),
                      'y': float(wy + uy * lat - math.sin(base_yaw) * back),
                      'yaw': yaw})

    def _wait_pose(self, timeout=2.0):
        """가장 최근 pose (신선도 요구 없음 — 진단·테스트용)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            p = self.poll.get()
            if p is not None:
                return p
            time.sleep(0.02)
        return (0.0, 0.0, 0.0, {})

    def _fresh_pose(self, seq0, timeout=2.0):
        """`seq0` **이후에 새로 받은** pose 만 돌려준다 (P0-3).

        제한 시간 안에 새 상태가 안 오면 그 전이를 만들지 않고 죽는다 —
        낡은 상태로 정상 속도로 쓸모없는 전이를 쌓는 것이 조용한 실패의
        뿌리다. 폴 주기 40 ms 에서 2 초는 폴 50 번이다."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            p, s = self.poll.get_seq()
            if p is not None and s > seq0:
                return p
            time.sleep(0.004)
        raise RuntimeError(
            '상태 폴러가 %.1f 초 동안 새 pose 를 못 줬다 (seq %d 정지, 누적 실패 %d)'
            ' — /state API 를 확인하라. 낡은 상태로 학습을 계속하지 않는다 (P0-3)'
            % (timeout, seq0, self.poll.fail_n))

    def _fresh_frame(self, t_after, hold, max_s=3.0):
        """`t_after` 이후의 카메라 프레임이 last_tick 에 실릴 때까지 드라이버를
        돌린다 (P0-4 — 텔레포트 전 장면으로 다음 행동을 만들지 않는다).

        hold=True 면 그동안 차를 세워 둔다(리셋 경로). False 면 base 로
        달린다 — 배포에서 텔레포트 직후 잔차가 0 으로 강제되는 구간과 같다.
        (소비한 틱 수, last_tick) 을 반환한다. 틱 수는 시간 비용에 넣는다 —
        심판의 시계는 그동안에도 돈다."""
        n = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_s:
            self._spin_one_tick()
            self._publish(0.0 if hold else self.cap.base)
            n += 1
            lt = self._features()
            if lt['frame_t'] is not None and lt['frame_t'] > t_after:
                return n, lt
        raise RuntimeError('텔레포트 후 %.1f 초 동안 새 프레임이 없다 — 환경 고장'
                           % max_s)

    # ── Gym API ────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 과적합 방지: N 에피소드마다 pin 을 뺀 콘을 다시 뿌린다.
        # 재배치가 곧 정본 갱신이라 아래 복원 단계는 건너뛴다.
        self._ep += 1
        n_shuf = 0
        if self.cone_reshuffle and self._ep % self.cone_reshuffle == 0:
            n_shuf = self.cones.reshuffle(self.rng)
        else:
            # 에피소드 시작 전에 월드를 정본으로 되돌린다. 콘이 밀린 채로
            # 학습하면 정책은 트랙이 아닌 것을 배운다 — 2026-08-21 에 그렇게
            # 22k 스텝을 버렸다.
            p0 = self.poll.get()
            moved = self.cones.displaced(p0[3]) if p0 else list(self.cones.base)
            for n in moved:
                self.cones.restore(n)
            if moved:
                time.sleep(0.15)

        if self.start_at_line:
            # **출발선에서 시작해 공식 결승선에서 끝난다.** 심판처럼 출발선
            # 0.10 m 뒤에 놓는다 (referee.py `one_run` 의 initialize 와 동일).
            # 랩 하나가 트랙 전체를 덮으므로 시작점을 흩뿌려 얻을 다양성은
            # 없고, 임의 시작이면 에피소드 경계가 화면·심판과 어긋난다.
            i = 0
            lat = float(self.rng.uniform(-0.10, 0.10))
            dyaw = float(self.rng.uniform(-0.10, 0.10))
            back = 0.10
        else:
            # 스폰이 콘 위에 떨어지면 첫 스텝부터 가짜 히트가 난다.
            i = int(self.rng.integers(self.start_lo, self.start_hi))
            lat = float(self.rng.uniform(-0.15, 0.15))
            for _ in range(20):
                wx, wy = self.track.wp[i]
                ux, uy = self.track.u[i]
                if self.cones.clearance(wx + ux * lat,
                                        wy + uy * lat) > SPAWN_CONE_CLEAR:
                    break
                i = int(self.rng.integers(self.start_lo, self.start_hi))
                lat = float(self.rng.uniform(-0.15, 0.15))
            dyaw = float(self.rng.uniform(-0.25, 0.25))
            back = 0.0
        # 플랜트도 에피소드마다 다시 뽑는다. 텔레포트가 속도를 0 으로
        # 만드므로(실측) 액추에이터 상태도 0 에서 출발한다 — sample() 이 함께 한다.
        plant = self.dyn.sample(self.rng)
        self._teleport_to(i, lat, dyaw, back=back)
        # 신선도 기준 시각은 API 가 **돌아온 뒤**다 — 적용 전에 도착한 이전
        # 장면 프레임을 새 프레임으로 오인하지 않게
        tele_t = time.monotonic()
        # 드라이버 내부 래치를 초기화 (텔레포트 안전).
        # **목록은 driver.py 의 v34c 텔레포트 핸들러와 정확히 같아야 한다**
        # (`jump > 45.0` 분기). 랩 시작은 출발선 -> 출발선이라 장면이 비슷해서
        # 그 자동 감지가 안 걸릴 수 있으므로 여기서 직접 지운다.
        self.node.pass_hold = 0
        self.node.pass_side = 0.0
        self.node.sharp_lock = False
        self.node.last_steer = 0.0
        self.node.last_arc = None
        self._hist = []
        self._prev_a = 0.0
        self._hit_state = {}
        self.finish.reset()

        # 워밍업: **차를 세워 둔 채** 새 프레임과 last_tick 만 채운다 (P1-7).
        # 베이스로 미리 달리는 숨은 주행 구간은 두지 않는다 — 심판이 시간을
        # 재는 구간이고, 배포(run_policy.py)에는 워밍업이 없다.
        n0, lt = self._fresh_frame(tele_t, hold=True)
        for _ in range(max(0, self.warmup_steps - n0)):
            self._spin_one_tick()
            self._publish(0.0)
        lt = self._features()
        # 리셋 텔레포트로 v34c 가 발화했을 수 있다 — 여기서 흡수해 첫 스텝이
        # 가짜 리셋을 미러링하지 않게 한다
        self._tp_seen = int(lt.get('teleport_n', 0))

        p = self._fresh_pose(self.poll.get_seq()[1])
        _, s, e, half = self.track.locate(p[0], p[1])
        self._s_prev = s
        self._lap_s = 0.0          # 진단값 (종료 판정에는 쓰지 않는다 — P0-1)
        self._time_s = 0.0         # 공식 비용에 넣은 누적 시간
        self._t = 0
        return self._stack(self._obs_row(lt)), {'dyn': plant, 'reshuffled': n_shuf}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        res = a * RES_MAX
        base_used = self.cap.base          # 이번 명령이 실제로 얹힌 base
        cmd = float(np.clip(base_used + res, 0.0, V_HARD))
        seq0 = self.poll.get_seq()[1]      # 행동 발행 전의 마지막 상태 seq
        cmd_out = self._publish(cmd)       # 플랜트 통과 후 실제로 나간 값

        self._spin_one_tick()
        self._t += 1
        extra = 0                          # 페널티 회복에 쓴 추가 틱

        p = self._fresh_pose(seq0)         # P0-3: 행동 이후의 새 상태만
        x, y, yaw, objs = p
        i_now, s, e, half = self.track.locate(x, y)
        ds = self.track.ds(self._s_prev, s)

        # ── 공식 결승선 (P0-1). 심판 evaluate() 와 같은 순서 — 완주 틱은
        # 페널티 처리 없이 즉시 끝난다 (심판이 그렇게 return 한다).
        lap_done = self.lap_episode and self.finish.update(x, y, yaw)

        off = hit = False
        moved = []
        fresh = []
        if not lap_done:
            # 심판과 **같은 메트릭**(최근접 wp 까지의 방사거리). 사영 |e| 로
            # 재면 한 방향으로만 관대해진다 — Track.off_track 독스트링 참조
            off = self.track.off_track(x, y)
            # 히트는 심판과 **같은 기준** — 콘이 움직였는가(1 cm / 4 cm).
            moved = self.cones.displaced(objs)
            # latch+rearm 도 심판과 **같은 의미**여야 한다. 심판의 latch 는
            # 시간이 지나면 풀리는 게 아니라 콘이 **제자리로 돌아온 것을 한 번
            # 관측해야** 풀린다: rearm 이 지났는데 아직 움직여 있으면 복원만
            # 하고 페널티는 주지 않는다. "1 초 지나면 재과금"으로 두면 복원이
            # 늦거나 재접촉이 있을 때 심판이 절대 주지 않는 두 번째 +5 를
            # 학습시킨다.
            now = time.monotonic()
            mset = set(moved)
            for n in self.cones.base:
                h = self._hit_state.get(n)
                if h and h['latched']:
                    if now < h['rearm']:
                        continue
                    if n not in mset:
                        h['latched'] = False      # 제자리 확인 -> 재무장
                        continue
                    self.cones.restore(n)         # 아직 움직여 있다: 복원만
                    h['rearm'] = now + CONE_REARM_S
                    continue
                if n in mset:
                    self._hit_state[n] = {'latched': True,
                                          'rearm': now + CONE_REARM_S}
                    fresh.append(n)
            hit = bool(fresh)

        # 랩 진행(진단값)은 실제로 나아간 거리로 센다
        self._lap_s += min(max(ds, -0.5), 0.5)
        # 텔레포트가 걸리는 스텝은 진행량 진단값을 0 으로 (순간이동 오염 방지)
        if off or hit or abs(ds) > 0.25:
            ds = 0.0

        teleported = False
        if off or hit:
            # 심판과 같은 복구 — **콘도 제자리로.** 이걸 빠뜨려 22k 스텝을 버렸다
            for n in moved:
                self.cones.restore(n)
            # 착지점도 심판과 같아야 한다 — 두 경우가 **다른 기준**을 쓴다:
            #   이탈  -> 차 기준 최근접 wp 에서 0.3 m 앞 (move_to_center)
            #   콘히트 -> **콘 기준** 최근접 wp 에서 0.3 m 앞. 차 기준으로 놓으면
            #            전면 접촉 시 착지점이 심판보다 ~0.17 m 뒤라 복원된 콘
            #            옆구리에 차가 걸친 채 출발한다 (재접촉 유발)
            car_idx = (self.finish.last_idx if self.finish.last_idx is not None
                       else i_now)
            if hit:
                b = self.cones.base[fresh[-1]]
                src = self.track.nearest_idx_near(b['x'], b['y'], car_idx, 5, 20)
            else:
                src = self.track.nearest_idx_near(x, y, car_idx, 10, 15)
            j = self.track.fwd_index(src, TELE_AHEAD)
            seq1 = self.poll.get_seq()[1]
            self._teleport_to(j)
            tele_t = time.monotonic()   # 신선도 기준은 API 반환 이후 (위 리셋과 동일)
            self.dyn.on_teleport()      # 텔레포트는 차를 세운다 — 모델도 세운다
            self.finish.note_teleport()  # 심판도 페널티 후 prev 를 지운다
            # P0-4: 새 장면이 올 때까지 base 로만 달린다 (배포의 잔차 0 구간).
            # 이 틱들은 아래에서 시간 비용에 들어간다.
            extra, _ = self._fresh_frame(tele_t, hold=False)
            p2 = self._fresh_pose(seq1)
            _, s2, e, half = self.track.locate(p2[0], p2[1])
            # 심판의 강제 이동도 트랙을 따라 앞으로 놓는다 — 진행 진단에 포함
            self._lap_s += min(max(self.track.ds(s, s2), -0.5), 0.5)
            s = s2
            teleported = True

        # ── 보상 = 공식 비용 그대로 (P0-6). 정형·배리어 없음.
        # ⚠️ 콘은 **개수만큼** 과금한다 — 심판은 콘마다 +5 를 주므로(콘 루프
        # 안에서 `penalty += 5`), 한 틱에 두 개를 치면 심판은 +10 이다.
        # `bool(hit)` 로 세면 그 차이만큼 정책이 무벌로 배운다 (전수검수 P2-5).
        step_t = DT * (1 + extra)
        self._time_s += step_t
        cost = step_t + P_OFF * off + P_CONE * len(fresh)
        reward = -W_SCALE * cost

        self._s_prev = s
        self._prev_a = a
        lt = self._features()
        # P0-4: 텔레포트면 관측 히스토리·행동 이력을 리셋한다. driver v34c 가
        # 스스로 감지한 텔레포트(teleport_n 증가)도 같은 리셋을 미러링한다 —
        # 배포 러너(run_policy.py)와 같은 규칙이어야 학습/배포가 일치한다.
        tp = int(lt.get('teleport_n', 0))
        if teleported or tp != self._tp_seen:
            self._hist = []
            self._prev_a = 0.0
        self._tp_seen = tp
        obs = self._stack(self._obs_row(lt))

        # ── 종료 계약 (P0-2): 공식 완주 = terminated (부트스트랩 없음),
        # 시간 안전망 = truncated (부트스트랩 있음)
        if self.lap_episode:
            timeout = (not lap_done) and self._time_s >= self.max_episode_s
        else:
            timeout = self._t >= self.episode_steps
        terminated = bool(lap_done)
        truncated = bool(timeout)
        info = {'ds': ds, 'off': bool(off), 'hit': bool(hit),
                'n_hit': len(fresh), 'moved': moved,
                'base': base_used,          # cmd 가 얹힌 그 base
                'base_next': self.cap.base,  # spin 이후의 새 base
                'cmd': cmd, 'cmd_out': cmd_out, 'res': res, 'e': e,
                'lap_s': self._lap_s, 'lap_done': bool(lap_done),
                'timeout': bool(timeout), 't_s': self._time_s,
                'extra_ticks': extra, 'adv': self.finish.adv}
        if self.log is not None:
            self.log.append(info)
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            # 정지는 플랜트를 우회해 즉시 낸다 (감속 한계에 걸려 계속 구르면 안 된다)
            self._real_speed.publish(Float64(data=0.0))
        except Exception:
            pass
        try:
            # 월드를 **예선맵 원본**으로 되돌려 다음 사용자(심판·오프라인 도구)
            # 에게 넘긴다. 셔플된 배치를 남기면 심판이 그걸 정본으로 굳힌다.
            self.cones.restore_home()
        except Exception:
            pass
        self.poll.stop_flag = True
        try:
            self.node.destroy_node()
        except Exception:
            pass
        # 이걸 빼면 종료 중 드라이버 타이머가 한 번 더 돌면서
        # "publisher's context is invalid" 트레이스백을 남긴다. 무해하지만
        # 로그 감시가 Traceback 을 잡아 가짜 경보를 낸다.
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
