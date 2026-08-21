#!/usr/bin/env python3
"""`dyn.py` 단위 테스트 — ROS·시뮬 없이 몇 초 만에 돈다.

    python3 rl/test_dyn.py

가장 중요한 것은 **TEST 1/2** 다: 랜덤화를 끄거나 참 플랜트 값으로 두면
배선이 명령을 **한 비트도 바꾸지 않는다**. 이게 깨지면 `eval_policy.py` 의
게이트가 무엇을 재는지 알 수 없게 된다.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from dyn import Dyn, DYN_RANGES, NOMINAL

DT = 0.05
FAIL = []


def check(name, ok, detail=''):
    print('%-46s %s %s' % (name, 'PASS' if ok else '!! FAIL', detail))
    if not ok:
        FAIL.append(name)


# 실제 랩에서 나오는 종류의 명령열 (상한 계단 + 콘 감속)
SEQ = [0.0, 0.9, 0.9, 0.45, 0.45, 0.9, 0.6, 0.4, 0.55, 0.9, 0.9, 0.0, 0.9]
STEER_SEQ = [0.0, 0.349, -0.349, 0.1, 0.0, -0.2, 0.349, 0.0]


def test_passthrough(enabled, tag):
    d = Dyn(DT, enabled=enabled)
    if enabled:
        d.p = dict(NOMINAL)
        d.reset_state()
    ev = max(abs(d.speed(v) - v) for v in SEQ)
    es = max(abs(d.steer(s) - s) for s in STEER_SEQ)
    check('TEST %s speed 순수통과' % tag, ev == 0.0, '|out-in| max %.1e' % ev)
    check('TEST %s steer 순수통과' % tag, es == 0.0, '|out-in| max %.1e' % es)


# ── TEST 1  랜덤화 OFF -> 순수 통과 ──
test_passthrough(False, '1 (enabled=False)')

# ── TEST 2  랜덤화 ON + 참 플랜트 값 -> 순수 통과 ──
test_passthrough(True, '2 (enabled, NOMINAL)')

# ── TEST 3  가속 한계가 실제로 걸린다 ──
d = Dyn(DT, enabled=True)
d.p = dict(NOMINAL, accel=2.0, decel=4.0)
d.reset_state()
# 0 -> 0.9 는 틱당 0.10 이므로 9 틱이 걸린다 (여유 있게 12 틱 돌린다)
up = [d.speed(0.9) for _ in range(12)]
check('TEST 3 가속 2.0 m/s² -> 틱당 +0.10', abs(up[0] - 0.1) < 1e-9,
      'first=%.4f' % up[0])
check('TEST 3 가속 단조 증가 + 목표 수렴',
      up == sorted(up) and abs(up[-1] - 0.9) < 1e-9,
      'ticks-to-target=%d' % (up.index(max(up)) + 1))
top = up[-1]
dn = [d.speed(0.0) for _ in range(3)]
check('TEST 3 감속 4.0 m/s² -> 틱당 -0.20',
      abs((top - dn[0]) - 0.2) < 1e-9, '%.4f -> %.4f' % (top, dn[0]))
check('TEST 3 가속≠감속이 독립적으로 적용됨',
      abs((up[1] - up[0]) - 0.1) < 1e-9 and abs((dn[1] - dn[0]) + 0.2) < 1e-9,
      '+%.2f / %.2f 틱당' % (up[1] - up[0], dn[1] - dn[0]))

# ── TEST 4  가속 한계 위반이 어떤 구간에서도 없다 ──
rng = np.random.default_rng(7)
worst = 0.0
for _ in range(200):
    d = Dyn(DT, enabled=True)
    p = d.sample(rng)
    prev = 0.0
    for v in SEQ * 4:
        out = d.speed(v)
        if out > prev:
            worst = max(worst, (out - prev) / DT - p['accel'])
        else:
            worst = max(worst, (prev - out) / DT - p['decel'])
        prev = out
check('TEST 4 200 에피소드 가감속 한계 위반 0', worst < 1e-9,
      'worst 초과 %.2e m/s²' % worst)

# ── TEST 5  구동 지연이 정확히 N 틱이다 ──
for k in (0, 1, 2, 3):
    d = Dyn(DT, enabled=True)
    d.p = dict(NOMINAL, act_delay=k)
    d.reset_state()
    src = [0.0, 0.3, 0.7, 0.2, 0.9, 0.5, 0.1, 0.8]
    out = [d.speed(v) for v in src]
    want = [0.0] * k + src[:len(src) - k]
    check('TEST 5 구동 지연 %d 틱' % k, out == want, '%s' % [round(v, 2) for v in out])

# ── TEST 6  서보 각속도 한계 ──
d = Dyn(DT, enabled=True)
d.p = dict(NOMINAL, steer_rate=4.0)      # 틱당 0.20 rad
d.reset_state()
s = [d.steer(0.349) for _ in range(3)]
check('TEST 6 서보 4.0 rad/s -> 틱당 0.20 rad',
      abs(s[0] - 0.2) < 1e-9 and abs(s[1] - 0.349) < 1e-9,
      'seq=%s' % [round(v, 4) for v in s])

# ── TEST 7  sample() 이 범위를 지키고 재현 가능하다 ──
r1 = np.random.default_rng(3)
r2 = np.random.default_rng(3)
a = [Dyn(DT).sample(r1) for _ in range(50)]
b = [Dyn(DT).sample(r2) for _ in range(50)]
check('TEST 7 같은 seed -> 같은 플랜트열', a == b)
ok_range = all(DYN_RANGES[k][0] <= p[k] <= DYN_RANGES[k][1]
               for p in a for k in DYN_RANGES)
check('TEST 7 전 축이 지정 범위 안', ok_range)
spread = len({(round(p['accel'], 3), p['act_delay'], p['obs_delay']) for p in a})
check('TEST 7 실제로 흔들린다 (50개 중 고유 %d)' % spread, spread >= 40)

# ── TEST 8  참 플랜트가 학습 분포 안에 있다 (상단이 시뮬 값을 덮는가) ──
check('TEST 8 가속 상단 ≥ 실측 하한 4.5', DYN_RANGES['accel'][1] >= 4.5,
      'accel hi=%.1f' % DYN_RANGES['accel'][1])
check('TEST 8 서보 범위가 시뮬 참값 10 을 포함',
      DYN_RANGES['steer_rate'][0] <= 10.0 <= DYN_RANGES['steer_rate'][1])
check('TEST 8 지연 하한이 0 (참 플랜트 포함)',
      DYN_RANGES['act_delay'][0] == 0 and DYN_RANGES['obs_delay'][0] == 0)

print()
if FAIL:
    print('!! 실패 %d 건: %s' % (len(FAIL), FAIL))
    sys.exit(1)
print('전부 통과')
