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

## 무엇을 흔드는가 (속도 축만 — 2026-08-21 전수검수 반영)
| 축 | 범위 | 시뮬 참값 | 근거 |
|---|---|---|---|
| 가속 한계 | 1.5~9.0 m/s² | 사실상 무한 (≥4.5 실측) | 실차 모터 토크 한계 |
| 감속 한계 | 1.0~9.0 m/s² | 〃 | 코스트다운은 가속보다 느리다 |
| 구동 지연 | 0~3 틱 (0~150 ms) | 0 | 시리얼·ESC 지연 |

가속·감속을 **따로 뽑는다** — 실차는 미는 힘과 멈추는 힘이 다르다.

### 제거된 축 (GPT 전수검수 P1-2 / P1-4 / Q8·Q9, 2026-08-21)
- **관측 지연**: 정책 관측 row 전체를 과거로 바꾸는 방식은 내부 모순이었다 —
  `prev_a`(자기 행동 이력)와 `frame_age` 까지 과거가 되고, 정책은 과거 base 를
  보는데 잔차는 현재 base 에 얹혔다. 실제 카메라 지연이면 베이스 제어기도
  같이 늦어야 하고, 정책 추론 지연이면 관측이 아니라 **행동 적용**이 늦어야
  한다. 실차(8/25)에서 각 구간 지연을 실측한 뒤 맞는 위치에 다시 모델링한다.
- **조향 서보 각속도 + 조향 지연**: 1단계 정책은 속도 잔차만 내고 실제 조향
  출력을 관측하지도 못하므로, 이 축은 학습에 보수성·분산만 만든다. 조향
  잔차(2단계)와 실차 실측 뒤에만 다시 연다.

## 범위에 대한 정직한 주석 (P1-3, 미해결로 남긴다)
가감속 상단 9.0 은 시뮬 참값("사실상 무한", 한 틱 완전 추종에는 18 m/s² 필요)을
**포함하지 않는다.** 연속 uniform 분포라 nominal passthrough 는 분포 안에 없다.
범위 재설정은 8/25 실차 step response 실측 후에만 의미가 있어 지금은 유지한다
(사용자 승인 범위, v71). 평가·배포는 어차피 참 플랜트다.

## 어디에 안 켜는가
`run_policy.py`(배포)와 `eval_policy.py`(공식 채점)는 **참 플랜트**를 그대로
쓴다. 랜덤화는 학습 분포를 넓히는 도구지 평가 조건이 아니다 — 평가까지
흔들면 게이트가 무엇을 재는지 알 수 없게 된다.

`enabled=False` 면 전 경로가 **순수 통과**다 (`test_dyn.py` 가 검증한다).
`NOMINAL` 파라미터(무한 가속 / 0 지연)로 켜도 마찬가지로 순수 통과다 —
랜덤화 배선 자체가 동작을 바꾸지 않음을 이 성질로 확인할 수 있다.
"""
from collections import deque

# 각 축의 (하한, 상한). 상한 = 시뮬 참값 쪽, 하한 = 실차 열화 쪽.
DYN_RANGES = {
    'accel':     (1.5, 9.0),      # m/s^2
    'decel':     (1.0, 9.0),      # m/s^2
    'act_delay': (0, 3),          # ticks (20 Hz -> 0~150 ms), 속도 명령에만
}

# 참 플랜트 = 무한 가감속 / 무지연. 이 값이면 켜져 있어도 통과다.
NOMINAL = {
    'accel': float('inf'), 'decel': float('inf'), 'act_delay': 0,
}


class Dyn:
    """에피소드 단위로 플랜트를 다시 뽑고, 틱 단위로 속도 명령을 통과시킨다.

    지연을 먼저 걸고 그 다음에 가감속 한계를 적용한다 — 전송이 먼저이고
    액추에이터 응답이 나중이라는 물리 순서 그대로다.
    조향은 건드리지 않는다 (모듈 독스트링의 "제거된 축" 참조).
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
                'act_delay': int(rng.integers(r['act_delay'][0],
                                              r['act_delay'][1] + 1)),
            }
        else:
            self.p = dict(NOMINAL)
        self.reset_state()
        return dict(self.p)

    def reset_state(self, v0=0.0):
        """액추에이터 상태 초기화. 텔레포트는 속도를 0 으로 만든다(실측)."""
        self.v_act = float(v0)
        n = int(self.p['act_delay']) + 1
        self.v_q = deque([float(v0)] * n, maxlen=n)

    def on_teleport(self):
        """심판 텔레포트는 차를 **세운다**(실측). 구동 액추에이터를 0 으로
        되돌린다 — 이걸 안 하면 텔레포트 직후 한 틱이 가속 한계를 우회해
        (모델 v_act 는 0.9 인데 차는 0) 랜덤화에 구멍이 생긴다."""
        self.v_act = 0.0
        n = self.v_q.maxlen
        self.v_q = deque([0.0] * n, maxlen=n)

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

    # ── 로깅 ────────────────────────────────────────────────────────
    def summary(self):
        if not self.enabled:
            return 'dyn OFF (참 플랜트)'
        p = self.p
        return ('dyn a%.1f/d%.1f m/s² · act %d 틱'
                % (p['accel'], p['decel'], p['act_delay']))
