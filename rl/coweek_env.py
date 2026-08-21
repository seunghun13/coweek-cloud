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
(driver.py 에는 v70 계측 dict 한 곳만 추가돼 있고, 그건 아무도 읽지 않는
순수 대입이라 주행이 불변임을 score_now ok235 + 예선 3런으로 검증했다.)

## 보상 (공식 점수의 초 단위 비용)
    r = W * (K_PROG * ds - C)
    C = dt + 5·I_off + 5·I_cone + dt·(2·B_edge + 3·B_cone) + 0.01·(Δa)²
공식 계수(5초/5초)는 고정하고 dense 항은 보조로만 쓴다.

## 텔레포트 + **콘 복원**
심판은 이탈·콘히트 때 차를 중심선 0.3 m 앞으로 옮기고 **맞은 콘을 원위치**
시킨다. env 도 **둘 다** 한다. 그 스텝의 진행량 ds 는 0 으로 만든다 —
안 그러면 정책이 "이탈해서 순간이동으로 전진" 을 배운다.

⚠️ **콘 복원을 빠뜨려 22k 스텝 학습 하나를 통째로 버렸다 (2026-08-21).**
6개 중 5개가 밀려나고 cone4 는 1.78 m 굴러 누웠는데, 히트 판정이 거리
기준이라 그 22k 스텝 내내 히트를 **0건** 보고했다. 자세한 건 `Cones` 참조.

## 동역학 랜덤화 (`dyn=True`, 기본값 — 사용자 승인 2026-08-21)
이 시뮬의 구동계는 **속도 소스**라 감속이 폴 1개 안에 끝난다(≥4.5 m/s² 실측).
**감속이 공짜인 플랜트에서 속도 잔차를 학습하면 정책은 반드시 그걸 착취한다.**
그래서 에피소드마다 가감속 한계·조향 서보 각속도·구동/관측 지연을 다시 뽑는다.
근거와 범위는 `dyn.py`, 검증은 `test_dyn.py`.

**평가·배포에서는 끈다.** `eval_policy.py` 는 공식 심판을 쓰고
`run_policy.py` 는 참 플랜트에 그대로 낸다 — 랜덤화는 학습 분포를 넓히는
도구지 평가 조건이 아니다. `dyn=False` 면 이 파일의 동작이 랜덤화 도입
전과 **한 비트도 다르지 않다** (`test_env_dyn.py` 가 시뮬에서 확인한다).
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
CONE_HIT_R = 0.1525        # 차 반폭 0.0975 + 콘 반경 0.055 (보상 정형용)
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

# ── 보상 계수 ──
W_SCALE = 0.1
K_PROG = 1.0               # s/m — 1 m 전진 = 1초 이득
P_OFF = 5.0
P_CONE = 5.0
W_EDGE = 2.0
W_CONEROOM = 3.0
W_SMOOTH = 0.01

# ── 에피소드 ──
DT = 1.0 / drv.RATE_HZ     # 0.05 s
# 워밍업은 **텔레포트 직후 last_tick 을 채우기 위한 최소값**만 남긴다 (5틱).
# 원래 1.2 초였던 이유는 8 초 슬라이스 시절 에피소드가 트랙 임의 지점에서
# 정지 상태로 시작해 "코너 한복판 정지출발"이라는 가짜 상태를 과학습하기
# 때문이었다. 랩을 출발선에서 시작하는 지금은 정지출발이 **심판과 같은 진짜
# 상태**이고 랩당 1회(0.1 %)뿐이라 편향이 될 수 없다.
# 오히려 1.2 초는 배포(run_policy.py 는 워밍업이 없다)와 어긋나고, 심판이
# 시간을 재는 구간을 정책에게 가린다.
WARMUP_S = 0.25
EPISODE_S = 8.0
RES_MAX = 0.35             # 속도 잔차 범위 [-RES_MAX, +RES_MAX] (대칭)
#   대칭으로 두는 이유: action 0 이 정확히 잔차 0 = racer-v64 가 된다.
#   SAC 초기 정책이 0 근처이므로 학습 시작점이 검증된 베이스와 같고,
#   zero residual 이 v64 를 재현하는지 그대로 검증할 수 있다 (GPT Test 1).
#   실차에서 이상하면 잔차를 0 으로 고정하는 것만으로 v64 복귀.
V_HARD = 1.40              # 안전 상한. 정책이 이 위로 못 간다

CAP_NAMES = ('vmax', 'alat', 'cone_slow', 'cone_near', 'arc_short',
             'checker', 'lidar1', 'lidar2', 'sharp', 'crawl')
HIST = 3


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

    `prev_a` = **직전 행동**(잔차, [-1,1] 정규화). 이게 없으면 보상의
    `W_SMOOTH·(a − prev_a)²` 항을 정책이 원리적으로 최적화할 수 없다 —
    자기가 직전에 뭘 냈는지 모르는 채로 그 차이를 벌점받기 때문이다.
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
    'band_px': 0, 'frame_t': None,
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
        nxt = np.roll(self.wp, -1, axis=0)
        self.yaw = np.arctan2(nxt[:, 1] - self.wp[:, 1], nxt[:, 0] - self.wp[:, 0])

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

    ## 배치 랜덤화 (사용자 지시 2026-08-21)
    ##   과적합 방지를 위해 **cone6 를 뺀 5개**를 에피소드마다 다시 뿌린다.
    ##   생성기는 **심판의 `random_cone_layout` 을 그대로** 쓴다 — 학습과 평가가
    ##   같은 분포를 쓰지 않으면 게이트가 무엇을 재는지 알 수 없다. 새 배치
    ##   생성기를 발명하지 않는다 (이 프로젝트에서 그 부류는 전부 기각됐다).
    ##   cone6 를 고정하는 이유는 심판 `--pin cone6` 와 같다: 결승 1.40 m 전
    ##   코너 출구라는 위치 자체가 최대 병목이라, 흩어버리면 병목이 사라진다.

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


class _SteerRelay:
    """driver 의 /steering 을 가로채 서보 각속도 한계를 통과시킨 뒤 낸다.

    드라이버의 `tick()` **안에서 즉시** 재퍼블리시한다. env 가 나중에 내면
    조향이 속도보다 한 틱 밀려 랜덤화가 아니라 버그가 된다.
    `Dyn.enabled=False` 면 값이 그대로 통과한다 (test_dyn TEST 1).
    """

    def __init__(self, real, dyn):
        self.real = real
        self.dyn = dyn
        self.last = 0.0

    def publish(self, msg):
        self.last = self.dyn.steer(float(msg.data))
        self.real.publish(Float64(data=self.last))


class _StatePoller(threading.Thread):
    """월드 포즈를 별도 스레드에서 폴링한다.

    20 Hz 제어 예산 50 ms 안에서 HTTP 왕복(~20 ms)을 하면 드라이버가
    카메라 프레임을 놓친다. **보상 계산 전용**이고 관측에는 안 들어간다."""
    daemon = True

    def __init__(self, hz=25.0):
        super().__init__()
        self.dt = 1.0 / hz
        self.lock = threading.Lock()
        self.pose = None
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            try:
                st = api('/state', timeout=5)
                v = st.get('vehicle') or {}
                # 콘은 dict 통째로 넘긴다 — z 가 있어야 "넘어짐"을 본다
                objs = {n: p for n, p in (st.get('objects') or {}).items()
                        if n.startswith('cone')}
                with self.lock:
                    self.pose = (float(v.get('x', 0.0)), float(v.get('y', 0.0)),
                                 float(v.get('yaw', 0.0)), objs)
            except Exception:
                pass
            time.sleep(self.dt)

    def get(self):
        with self.lock:
            return self.pose


class CoweekResidualEnv(gym.Env):
    """base(v64) 위에 속도 잔차만 얹는 환경."""

    metadata = {'render_modes': []}

    def __init__(self, seed=0, episode_s=EPISODE_S, warmup_s=WARMUP_S,
                 start_lo=0, start_hi=None, log=None, dyn=True, dyn_ranges=None,
                 cone_reshuffle=1, cone_pin=('cone6',),
                 lap_episode=True, max_episode_s=150.0,
                 start_at_line=True):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.track = Track()
        # 에피소드 = **한 바퀴** (사용자 지시 2026-08-21). `episode_s` 는
        # lap_episode=False 일 때만 쓰이고, 랩 모드에선 max_episode_s 가
        # "차가 멈춰 영원히 안 끝나는" 경우를 막는 안전망일 뿐이다.
        self.lap_episode = bool(lap_episode)
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
        # 조향은 가로채서 서보 한계만 태우고 바로 흘려보낸다 (값은 안 바꾼다)
        self._real_steer = self.node.pub_steer
        self.node.pub_steer = _SteerRelay(self._real_steer, self.dyn)

        self.poll = _StatePoller()
        self.poll.start()

        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0,
                                            (OBS_ROW_DIM * HIST,), np.float32)
        self._hist = []
        self._raw = []
        self._prev_a = 0.0
        self._hit_rearm = {}

    # ── 내부 ────────────────────────────────────────────────────────
    def _spin_one_tick(self):
        """드라이버 타이머가 한 번 돌 때까지 spin. 20 Hz 벽시계에 묶인다."""
        n0 = self.cap.n
        t0 = time.monotonic()
        while self.cap.n == n0 and time.monotonic() - t0 < 1.0:
            rclpy.spin_once(self.node, timeout_sec=0.005)
        return self.cap.n != n0

    def _publish(self, speed):
        """플랜트(가감속 한계 + 구동 지연)를 통과시켜 실제로 낸다.

        랜덤화가 꺼져 있으면 `speed` 가 그대로 나간다."""
        out = self.dyn.speed(float(speed))
        self._real_speed.publish(Float64(data=out))
        return out

    def _features(self):
        lt = getattr(self.node, 'last_tick', None)
        if lt is None:
            lt = {'speed': 0.0, 'steer': 0.0, 'aim': 0.0, 'mode': 'wait',
                  'carea': 0.0, 'fmin': drv.LID_RANGE_MAX, 'lbias': 0.0,
                  'dodging': False, 'sharp': False, 'crawl': False,
                  'checker': False, 'arc_safe': 1.0, 'pass_hold': 0,
                  'pass_side': 0.0, 'ncone': 0, 'band_px': 0, 'frame_t': None}
        return lt

    def _cap_onehot(self, lt):
        return cap_onehot(lt)

    def _obs_row(self, lt):
        # 직전 행동을 함께 넣는다 (obs_row 독스트링 참조).
        # step() 은 self._prev_a = a 를 **관측을 만들기 전에** 갱신하므로
        # 여기 들어가는 값은 "방금 낸 행동"이다. reset() 에선 0 이다.
        return obs_row(lt, self._prev_a)

    def _stack(self, row):
        """관측 지연은 여기서 준다 — **정책이 보는 것만** 늦어진다.

        베이스 제어기는 자기 카메라를 그대로 쓴다. 이 지연이 상정하는 것은
        정책 쪽 센싱·추론 파이프라인이고, 배포(`run_policy.py`)는 지연 0 이라
        범위 하단(0)이 실제 배포 조건이다."""
        self._raw.append(row)
        self._raw = self._raw[-(HIST + 4):]
        k = self.dyn.obs_delay
        row = self._raw[-1 - k] if len(self._raw) > k else self._raw[0]
        self._hist.append(row)
        while len(self._hist) < HIST:
            self._hist.insert(0, row)
        self._hist = self._hist[-HIST:]
        return np.concatenate(self._hist).astype(np.float32)

    def _teleport_to(self, i, lat=0.0, dyaw=0.0):
        wx, wy = self.track.wp[i]
        ux, uy = self.track.u[i]
        yaw = float(self.track.yaw[i]) + dyaw
        api('/pose', {'x': float(wx + ux * lat), 'y': float(wy + uy * lat),
                      'yaw': yaw})

    def _wait_pose(self, timeout=2.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            p = self.poll.get()
            if p is not None:
                return p
            time.sleep(0.02)
        return (0.0, 0.0, 0.0, {})

    # ── Gym API ────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 과적합 방지: N 에피소드마다 cone6 를 뺀 콘을 다시 뿌린다.
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
            # **출발선에서 시작해 출발선에서 끝난다.** 랩 단위인데 시작점이
            # 임의면 에피소드 경계가 눈에 보이는 출발선과 어긋나서, 도중에
            # 결승선을 지나고 엉뚱한 지점에서 리셋·콘재배치가 일어난다
            # (사용자가 화면에서 잡아낸 문제, 2026-08-21).
            # 랩 하나가 이미 트랙 전체를 덮으므로 시작점을 흩뿌려 얻을
            # 다양성도 없다 — 8 초 슬라이스 시절의 잔재였다.
            i = 0
            lat = float(self.rng.uniform(-0.10, 0.10))
            dyaw = float(self.rng.uniform(-0.10, 0.10))
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
        # 플랜트도 에피소드마다 다시 뽑는다. 텔레포트가 속도를 0 으로
        # 만드므로(실측) 액추에이터 상태도 0 에서 출발한다 — sample() 이 함께 한다.
        plant = self.dyn.sample(self.rng)
        self._teleport_to(i, lat, dyaw)
        # 드라이버 내부 래치를 초기화 (텔레포트 안전).
        # **목록은 driver.py 의 v34c 텔레포트 핸들러와 정확히 같아야 한다**
        # (`jump > 45.0` 분기). 랩 시작은 출발선 -> 출발선이라 장면이 비슷해서
        # 그 자동 감지가 안 걸릴 수 있으므로 여기서 직접 지운다.
        # 그쪽 주석: "stale frame-scale memory ... ~0.5 m drift on the fresh
        # straight -> finish-miss cascade" — 우리가 쫓는 바로 그 병목이다.
        self.node.pass_hold = 0
        self.node.pass_side = 0.0
        self.node.sharp_lock = False
        self.node.last_steer = 0.0
        self.node.last_arc = None
        self._hist = []
        self._raw = []
        self._prev_a = 0.0
        self._hit_rearm = {}
        time.sleep(0.12)                      # 새 프레임 도착 (실측 78 ms)

        # 워밍업: 베이스 제어기로만 달려 비영 속도 상태를 만든다.
        # (정지출발 상태만 학습하면 인공적 가속 구간을 과학습한다)
        for _ in range(self.warmup_steps):
            self._spin_one_tick()
            self._publish(self.cap.base)
        p = self._wait_pose()
        _, s, e, half = self.track.locate(p[0], p[1])
        self._s_prev = s
        self._lap_s = 0.0          # 이번 에피소드에 실제로 나아간 arc 길이
        self._t = 0
        lt = self._features()
        return self._stack(self._obs_row(lt)), {'dyn': plant, 'reshuffled': n_shuf}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        res = a * RES_MAX
        base_used = self.cap.base          # 이번 명령이 실제로 얹힌 base
        cmd = float(np.clip(base_used + res, 0.0, V_HARD))
        cmd_out = self._publish(cmd)       # 플랜트 통과 후 실제로 나간 값

        self._spin_one_tick()
        self._t += 1

        p = self._wait_pose()
        x, y, _yaw, objs = p[0], p[1], p[2], p[3]
        _, s, e, half = self.track.locate(x, y)
        ds = self.track.ds(self._s_prev, s)

        off = abs(e) > half + OFF_MARGIN
        # 히트는 심판과 **같은 기준** — 콘이 움직였는가(1 cm / 4 cm).
        # 거리 기준은 차 중심만 봐서 앞·뒤 범퍼 접촉을 놓친다 (Cones 독스트링)
        moved = self.cones.displaced(objs)
        # 심판과 같은 latch+rearm: 복원이 전파되기 전 몇 스텝 동안 같은 콘이
        # 계속 "움직인 상태"로 보이는데, 그걸 매번 새 히트로 세면 안 된다
        now = time.monotonic()
        fresh = [n for n in moved if now >= self._hit_rearm.get(n, 0.0)]
        hit = bool(fresh)
        for n in fresh:
            self._hit_rearm[n] = now + CONE_REARM_S

        # 랩 진행은 **보상과 따로** 센다. 보상 쪽 ds 는 순간이동 착취를 막으려고
        # 0 으로 죽이지만, 랩이 끝났는지는 실제로 나아간 거리로 판단해야 한다.
        self._lap_s += min(max(ds, -0.5), 0.5)

        # 텔레포트가 걸리는 스텝은 진행량을 0 으로 — 순간이동 착취 차단
        if off or hit or abs(ds) > 0.25:
            ds = 0.0

        edge_room = (half + OFF_MARGIN) - abs(e)
        buf = 0.04
        b_edge = max(0.0, (buf - edge_room) / buf) ** 2
        # 콘은 항상 정본으로 복원되므로 정본 거리 = 실제 거리다
        cone_room = self.cones.clearance(x, y) - CONE_HIT_R
        b_cone = max(0.0, (0.10 - cone_room) / 0.10) ** 2 if cone_room < 0.10 else 0.0

        cost = (DT + P_OFF * off + P_CONE * hit
                + DT * (W_EDGE * b_edge + W_CONEROOM * b_cone)
                + W_SMOOTH * (a - self._prev_a) ** 2)
        reward = W_SCALE * (K_PROG * ds - cost)

        if off or hit:
            # 심판과 같은 복구 — **콘도 제자리로.** 이걸 빠뜨려 22k 스텝을 버렸다
            for n in moved:
                self.cones.restore(n)
            # 그리고 차는 중심선 위 0.3 m 앞
            i, _, _, _ = self.track.locate(x, y)
            j = i
            acc = 0.0
            while acc < TELE_AHEAD:
                j = (j + 1) % len(self.track.wp)
                acc += float(np.linalg.norm(self.track.wp[j] - self.track.wp[j - 1]))
            self._teleport_to(j)
            self.dyn.on_teleport()      # 텔레포트는 차를 세운다 — 모델도 세운다
            time.sleep(0.10)
            p = self._wait_pose()
            _, s2, e, half = self.track.locate(p[0], p[1])
            # 심판의 강제 이동도 트랙을 따라 앞으로 놓는다 — 랩 진행에 포함한다
            self._lap_s += min(max(self.track.ds(s, s2), -0.5), 0.5)
            s = s2

        self._s_prev = s
        self._prev_a = a
        lt = self._features()
        obs = self._stack(self._obs_row(lt))
        lap_done = self.lap_episode and self._lap_s >= self.track.total
        truncated = bool(lap_done or self._t >= self.episode_steps)
        info = {'ds': ds, 'off': bool(off), 'hit': bool(hit), 'moved': moved,
                'base': base_used,          # cmd 가 얹힌 그 base
                'base_next': self.cap.base,  # spin 이후의 새 base
                'cmd': cmd, 'cmd_out': cmd_out, 'res': res, 'e': e,
                'lap_s': self._lap_s, 'lap_done': bool(lap_done)}
        if self.log is not None:
            self.log.append(info)
        return obs, float(reward), False, bool(truncated), info

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
