#!/usr/bin/env python3
"""동역학 도메인 랜덤화 — **학습에서만** 켠다.

## 왜 필요한가 (측정된 이유, 추측 아님)
시뮬 구동계(`gz AckermannSteering`, `max_velocity 3.0 / max_acceleration ±10`)는
**속도 소스**다. 그 결과:

- 감속이 폴 1개(~0.07 s) 안에 끝난다 — 하한 **≥4.5 m/s²** 실측
- 3랩 **실주행거리/명령거리 = 0.9968** — 차는 명령을 오차 없이 따라간다
- 전 구간 실측 최고속 0.9076 m/s = `V_MAX` 그 자체 (물리가 아니라 상수가 상한)

즉 **이 시뮬에서 감속은 공짜다.** 속도 잔차를 학습시키면 정책은 반드시 이걸
착취한다 — "코너 직전까지 최고속 → 순간 감속". 실차 모터는 그렇게 안 선다.
작년 RL 모델이 실차에서 전멸한 것이 정확히 이 형태다.

**콘 위치만 랜덤화해서는 이 계통 오차를 못 막는다.** 도메인 랜덤화는
랜덤화한 축에 대해서만 효과가 있다. 그래서 축을 명시적으로 고른다.

## 무엇을 흔드는가 (3축, 사용자 승인 2026-08-21)
| 축 | 범위 | 시뮬 참값 | 근거 |
|---|---|---|---|
| 가속 한계 | 1.5~9.0 m/s² | 사실상 무한 (≥4.5 실측) | 실차 모터 토크 한계 |
| 감속 한계 | 1.0~9.0 m/s² | 〃 | 코스트다운은 가속보다 느리다 |
| 조향 서보 각속도 | 3~12 rad/s | 10 (physicar-ros *"Limit steering joint velocity to 10"*) | 서보 부하·전압 |
| 구동 지연 | 0~3 틱 (0~150 ms) | 0 | 시리얼·ESC 지연 |
| 관측 지연 | 0~2 틱 (0~100 ms) | 0 | 카메라 파이프라인 |

가속·감속을 **따로 뽑는다** — 실차는 미는 힘과 멈추는 힘이 다르다.

**범위 상단이 시뮬 참값을 포함한다.** 참 플랜트가 학습 분포 안에 있어야
정책이 지금 성적을 통째로 버리지 않는다. 하단은 실차 열화를 상정한 값이다.

## 어디에 안 켜는가
`run_policy.py`(배포)와 `eval_policy.py`(공식 채점)는 **참 플랜트**를 그대로
쓴다. 랜덤화는 학습 분포를 넓히는 도구지 평가 조건이 아니다 — 평가까지
흔들면 게이트가 무엇을 재는지 알 수 없게 된다.

`enabled=False` 면 전 경로가 **순수 통과**다 (`test_dyn.py` 가 검증한다).
`NOMINAL` 파라미터(무한 가속 / 0 지연)로 켜도 마찬가지로 순수 통과다 —
랜덤화 배선 자체가 동작을 바꾸지 않음을 이 성질로 확인할 수 있다.

## 다음 후보 축 (지금은 안 넣는다)
모터 게인 오차(`v_gain`), 조향 트림 바이어스. 둘 다 실차에서 실재하지만
**1차원 속도 잔차로는 보상할 수 없는 축**이라 보상 신호에 잡음만 더한다.
조향 잔차(2단계)를 붙일 때 같이 열 것.
"""
from collections import deque

# 각 축의 (하한, 상한). 상한 = 시뮬 참값 쪽, 하한 = 실차 열화 쪽.
DYN_RANGES = {
    'accel':      (1.5, 9.0),      # m/s^2
    'decel':      (1.0, 9.0),      # m/s^2
    'steer_rate': (3.0, 12.0),     # rad/s
    'act_delay':  (0, 3),          # ticks (20 Hz -> 0~150 ms)
    'obs_delay':  (0, 2),          # ticks (20 Hz -> 0~100 ms)
}

# 참 플랜트 = 무한 가감속 / 무한 서보 / 무지연. 이 값이면 켜져 있어도 통과다.
NOMINAL = {
    'accel': float('inf'), 'decel': float('inf'),
    'steer_rate': float('inf'), 'act_delay': 0, 'obs_delay': 0,
}


class Dyn:
    """에피소드 단위로 플랜트를 다시 뽑고, 틱 단위로 명령을 통과시킨다.

    지연을 먼저 걸고 그 다음에 가감속 한계를 적용한다 — 전송이 먼저이고
    액추에이터 응답이 나중이라는 물리 순서 그대로다.
    """

    def __init__(self, dt, enabled=True, ranges=None):
        self.dt = float(dt)
        self.enabled = bool(enabled)
        self.ranges = dict(DYN_RANGES)
        if ranges:
            self.ranges.update(ranges)
        self.p = dict(NOMINAL)
        self.reset_state()

    # ── 에피소드 ────────────────────────────────────────────────────
    def sample(self, rng):
        """새 플랜트를 뽑는다. `rng` 는 numpy Generator."""
        if self.enabled:
            r = self.ranges
            self.p = {
                'accel': float(rng.uniform(*r['accel'])),
                'decel': float(rng.uniform(*r['decel'])),
                'steer_rate': float(rng.uniform(*r['steer_rate'])),
                'act_delay': int(rng.integers(r['act_delay'][0],
                                              r['act_delay'][1] + 1)),
                'obs_delay': int(rng.integers(r['obs_delay'][0],
                                              r['obs_delay'][1] + 1)),
            }
        else:
            self.p = dict(NOMINAL)
        self.reset_state()
        return dict(self.p)

    def reset_state(self, v0=0.0, steer0=0.0):
        """액추에이터 상태 초기화. 텔레포트는 속도를 0 으로 만든다(실측)."""
        self.v_act = float(v0)
        self.st_act = float(steer0)
        n = int(self.p['act_delay']) + 1
        self.v_q = deque([float(v0)] * n, maxlen=n)
        self.st_q = deque([float(steer0)] * n, maxlen=n)

    def on_teleport(self):
        """심판 텔레포트는 차를 **세운다**(실측). 구동 액추에이터만 0 으로
        되돌린다 — 이걸 안 하면 텔레포트 직후 한 틱이 가속 한계를 우회해
        (모델 v_act 는 0.9 인데 차는 0) 랜덤화에 구멍이 생긴다.

        조향 서보는 건드리지 않는다. 차가 옮겨져도 서보 각도는 유지된다."""
        self.v_act = 0.0
        n = self.v_q.maxlen
        self.v_q = deque([0.0] * n, maxlen=n)

    @property
    def obs_delay(self):
        return int(self.p['obs_delay'])

    # ── 틱 ──────────────────────────────────────────────────────────
    def speed(self, target):
        """/speed 명령 하나를 플랜트에 통과시킨다."""
        if not self.enabled:
            return float(target)
        self.v_q.append(float(target))
        tgt = self.v_q[0]                       # act_delay 틱 전의 명령
        hi = self.v_act + self.p['accel'] * self.dt
        lo = self.v_act - self.p['decel'] * self.dt
        self.v_act = min(hi, max(lo, tgt))
        return float(self.v_act)

    def steer(self, target):
        """/steering 명령 하나를 서보 각속도 한계에 통과시킨다."""
        if not self.enabled:
            return float(target)
        self.st_q.append(float(target))
        tgt = self.st_q[0]
        m = self.p['steer_rate'] * self.dt
        self.st_act = min(self.st_act + m, max(self.st_act - m, tgt))
        return float(self.st_act)

    # ── 로깅 ────────────────────────────────────────────────────────
    def summary(self):
        if not self.enabled:
            return 'dyn OFF (참 플랜트)'
        p = self.p
        return ('dyn a%.1f/d%.1f m/s² · servo %.1f rad/s · act %d · obs %d 틱'
                % (p['accel'], p['decel'], p['steer_rate'],
                   p['act_delay'], p['obs_delay']))
