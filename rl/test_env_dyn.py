#!/usr/bin/env python3
"""env 배선 검증 — 시뮬을 **실제로 돌려서** 잰다 (약 2분).

    . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
    cd ~/physicar_ws/coweek && python3 rl/test_env_dyn.py

`test_dyn.py` 가 `Dyn` 자체를 재는 단위 테스트라면, 이 파일은 **배선**을 잰다.

**TEST A 가 가장 중요하다.** `dyn=False` 면 랜덤화 도입 전과 명령이 한 비트도
달라지지 않아야 한다 — 평가(`eval_policy.py`)와 배포(`run_policy.py`)가 그
경로를 쓰기 때문이다. 여기가 깨지면 게이트가 무엇을 재는지 알 수 없게 된다.

⚠️ 심판·`run.sh` 드라이버가 떠 있으면 안 된다. 먼저 `python3 kill_orphans.py`.
"""
import sys
import os
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np

from coweek_env import CoweekResidualEnv, DT

FAIL = []
ZERO = np.zeros(1, np.float32)


def check(name, ok, detail=''):
    print('%-44s %s %s' % (name, 'PASS' if ok else '!! FAIL', detail), flush=True)
    if not ok:
        FAIL.append(name)


def episode(env):
    """한 에피소드를 잔차 0 으로 굴리고 (플랜트, 스텝기록) 을 준다."""
    _, info0 = env.reset()
    plant = info0.get('dyn', {})
    rows = []
    prev = float(env.dyn.v_act)          # 워밍업 직후의 액추에이터 상태
    for _ in range(env.episode_steps):
        _, _, term, trunc, info = env.step(ZERO)
        rows.append((prev, info))
        prev = float(env.dyn.v_act)      # 텔레포트가 걸리면 여기서 0 이 된다
        if term or trunc:                # 완주(terminated)도 에피소드 끝이다
            break
    return plant, rows


print('env 생성 중 (드라이버 노드 + 상태 폴러)...', flush=True)
# 배선 테스트는 4 초 슬라이스로 돈다 (랩 모드면 한 번에 50 초라 너무 길다).
# 랩 종료 자체는 TEST G 에서 따로 잰다.
env = CoweekResidualEnv(seed=11, episode_s=4.0, warmup_s=1.2, dyn=False,
                        lap_episode=False, start_at_line=False)

# ── TEST A  dyn=False 는 순수 통과여야 한다 ──────────────────────────
print('\n=== TEST A  dyn=False (참 플랜트) — 2 에피소드 ===', flush=True)
rows_a = []
for _ in range(2):
    _, r = episode(env)
    rows_a += r
inf_a = [i for _, i in rows_a]

d_base = max(abs(i['cmd'] - i['base']) for i in inf_a)
d_out = max(abs(i['cmd_out'] - i['cmd']) for i in inf_a)
check('A1 잔차 0 -> cmd == base (v64 동일)', d_base == 0.0,
      '|cmd-base| max %.1e (n=%d)' % (d_base, len(inf_a)))
check('A2 배선이 명령을 안 바꾼다 (cmd_out == cmd)', d_out == 0.0,
      '|cmd_out-cmd| max %.1e' % d_out)
ds_a = sum(i['ds'] for i in inf_a)
off_a = sum(int(i['off']) for i in inf_a)
hit_a = sum(int(i['hit']) for i in inf_a)
print('    참고: 진행 %.2f m | off %d | hit %d' % (ds_a, off_a, hit_a))

# ── TEST B  dyn=True 는 실제로 물고, 한계를 안 넘는다 ────────────────
print('\n=== TEST B  dyn=True (랜덤 플랜트) — 3 에피소드 ===', flush=True)
env.dyn.enabled = True
worst = -1e9
bite = []
plants = []
ds_b = off_b = hit_b = 0
for _ in range(3):
    p, rows = episode(env)
    plants.append(p)
    print('    플랜트: %s' % env.dyn.summary(), flush=True)
    for prev, info in rows:
        out = float(info['cmd_out'])
        bite.append(abs(out - float(info['cmd'])))
        # 가감속 한계 위반량 (양수면 위반)
        if out > prev:
            worst = max(worst, (out - prev) / DT - p['accel'])
        else:
            worst = max(worst, (prev - out) / DT - p['decel'])
        ds_b += info['ds']
        off_b += int(info['off'])
        hit_b += int(info['hit'])

mean_bite = float(np.mean(bite))
check('B1 가감속 한계 위반 0 (전 스텝)', worst < 1e-9,
      'worst 초과 %.2e m/s² (n=%d)' % (worst, len(bite)))
check('B2 플랜트가 실제로 물고 있다', mean_bite > 1e-3,
      '평균 |cmd-cmd_out| = %.4f m/s' % mean_bite)
check('B3 에피소드마다 다른 플랜트를 뽑는다',
      len({(round(p['accel'], 4), p['act_delay']) for p in plants}) == len(plants),
      '%d 개 전부 고유' % len(plants))
check('B4 랜덤화해도 차가 굴러간다 (진행 > 0)', ds_b > 0.0,
      '진행 %.2f m | off %d | hit %d' % (ds_b, off_b, hit_b))

# ── TEST C  다시 끄면 순수 통과로 돌아온다 (되돌릴 수 있는가) ────────
print('\n=== TEST C  dyn 을 다시 끈다 — 1 에피소드 ===', flush=True)
env.dyn.enabled = False
_, rows_c = episode(env)
inf_c = [i for _, i in rows_c]
d_out_c = max(abs(i['cmd_out'] - i['cmd']) for i in inf_c)
check('C1 끄면 즉시 순수 통과로 복귀', d_out_c == 0.0,
      '|cmd_out-cmd| max %.1e' % d_out_c)

# ── TEST D  관측 모양 ────────────────────────────────────────────────
obs, _ = env.reset()
from coweek_env import OBS_ROW_DIM, HIST, obs_row, DEFAULT_LT   # noqa: E402
check('D1 관측 차원이 선언과 일치', obs.shape == (OBS_ROW_DIM * HIST,),
      '%s vs %d' % (obs.shape, OBS_ROW_DIM * HIST))
check('D2 관측이 전부 유한', bool(np.isfinite(obs).all()))
check('D3 obs_row 가 직전 행동을 받는다',
      obs_row(DEFAULT_LT, 0.5).shape[0] == OBS_ROW_DIM
      and obs_row(DEFAULT_LT, 0.5)[12] == np.float32(0.5),
      'row %d, prev_a 칸 %.2f' % (obs_row(DEFAULT_LT, 0.5).shape[0],
                                  obs_row(DEFAULT_LT, 0.5)[12]))
# 학습/배포 관측 불일치는 예외 없이 조용히 성능만 무너뜨린다. 소스로 막는다.
_rp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'run_policy.py'), encoding='utf-8').read()
check('D4 배포 러너도 직전 행동을 넣는다',
      'obs_row(lt, self.prev_a)' in _rp,
      'run_policy.py 가 obs_row 를 1인자로 부르면 학습과 관측이 어긋난다')

# ── TEST E  콘을 밀면 히트로 잡고 되돌려 놓는가 ─────────────────────
# 22k 스텝을 버리게 만든 바로 그 결함의 회귀 테스트다.
print('\n=== TEST E  콘 변위 감지 + 복원 ===', flush=True)
from coweek_env import api, CONE_MOVE_TOL       # noqa: E402
import math as _m                               # noqa: E402

name = sorted(env.cones.base)[0]
b = env.cones.base[name]
api('/models/%s/pose' % name,
    {'x': b['x'] + 0.40, 'y': b['y'] + 0.30, 'yaw': b.get('yaw', 0.0)})
time.sleep(0.6)

objs = env._wait_pose()[3]
det = env.cones.displaced(objs)
check('E1 밀린 콘을 변위로 감지', name in det, '감지=%s' % det)

_, _, _, _, info_e = env.step(ZERO)
check('E2 그 스텝이 hit 으로 잡힌다', info_e['hit'] is True,
      "hit=%s moved=%s" % (info_e['hit'], info_e['moved']))
check('E3 히트 스텝의 진행량은 0', info_e['ds'] == 0.0, 'ds=%s' % info_e['ds'])

time.sleep(0.6)
objs = env._wait_pose()[3]
p = objs.get(name, {})
d = _m.hypot(p.get('x', 9) - b['x'], p.get('y', 9) - b['y'])
check('E4 콘이 정본으로 복원됐다', d <= CONE_MOVE_TOL, '잔여 %.4f m' % d)
check('E5 다른 콘은 안 건드렸다', env.cones.displaced(objs) == [],
      '%s' % env.cones.displaced(objs))

# ── TEST F  콘 재배치 — 랜덤 + cone6 고정 (측정 전략) ─────────────
print('\n=== TEST F  콘 재배치 — 랜덤 5개 + cone6 고정 ===', flush=True)
home = {n: (p['x'], p['y']) for n, p in env.cones.home.items()}
# 기본 pin = cone6. **대회 규칙이 아니라 측정 전략**이다 — 미해결 병목을
# 테스트에 남긴다. eval_policy --finals 도 같은 분포를 써야 한다 (P0-8).
check('F0 기본 pin 이 cone6 (평가와 같은 분포)', env.cones.pin == ('cone6',),
      'pin=%s' % (env.cones.pin,))
layouts = []
for k in range(3):
    n_shuf = env.cones.reshuffle(env.rng)
    cur = {n: (p['x'], p['y']) for n, p in env.cones.base.items()}
    layouts.append(tuple(sorted((n, round(x, 3), round(y, 3))
                                for n, (x, y) in cur.items())))
    if k == 0:
        check('F2 기본 분포는 5개만 배치 (cone6 제외)', n_shuf == 5,
              'n=%d' % n_shuf)
        d6 = _m.hypot(cur['cone6'][0] - home['cone6'][0],
                      cur['cone6'][1] - home['cone6'][1])
        check('F2b cone6 는 예선맵 자리 그대로', d6 <= CONE_MOVE_TOL,
              '이동 %.4f m' % d6)
        movedn = [n for n in cur if n != 'cone6'
                  and _m.hypot(cur[n][0] - home[n][0],
                               cur[n][1] - home[n][1]) > 0.05]
        check('F3 나머지 5개는 실제로 움직였다', len(movedn) == 5,
              '움직인 콘 %s' % sorted(movedn))
check('F4 매번 다른 배치가 나온다', len(set(layouts)) == 3,
      '고유 배치 %d/3' % len(set(layouts)))
objs_f = {n: p for n, p in api('/state').get('objects', {}).items()
          if n.startswith('cone')}
check('F5 재배치 직후 가짜 히트가 없다', env.cones.displaced(objs_f) == [],
      '%s' % env.cones.displaced(objs_f))
# 6개 전부 랜덤은 **옵션**으로만 남는다 (더 어려운 스트레스 분포)
env.cones.pin = ()
check('F6 pin 을 비우면 6개 전부 랜덤 (옵션)',
      env.cones.reshuffle(env.rng) == 6)
env.cones.pin = ('cone6',)               # 본선 분포로 복귀

# ── TEST G  랩 단위 에피소드 — 한 바퀴를 실제로 돈다 ───────────────
# 사용자 지시(2026-08-21): 이탈·콘히트가 나도 에피소드를 끊지 말고 한 바퀴를
# 계속 돌아야 한다. 평가가 랩이므로 학습도 랩이어야 분포가 맞는다.
# ⚠️ G1 은 이제 **공식 결승선 판정**(P0-1)과 종료 계약(P0-2)을 함께 잰다:
# 완주는 terminated 로 끝나야 하고, lap_done 은 앞점이 결승 선분을 실제로
# 교차했다는 뜻이다 (누적 arc 도달이 아니라).
print('\n=== TEST G  랩 단위 에피소드 (실제 한 바퀴, 약 1분) ===', flush=True)
env.lap_episode = True
env.start_at_line = True
env.episode_steps = int(150.0 / DT)
env.reset()
# 출발선에서 시작해야 에피소드 경계가 화면·심판과 일치한다. 리셋은 심판처럼
# 출발선 0.10 m **뒤**에 세워 두고(P1-7 — 워밍업은 정지 상태로 프레임만
# 채운다), 주행은 첫 step 부터다. arc 는 출발선 직전(음수쪽)이어야 한다.
p_start = env._wait_pose()
_, s_start, _, _ = env.track.locate(p_start[0], p_start[1])
s_signed = s_start if s_start < env.track.total / 2 else s_start - env.track.total
check('G0 랩이 출발선(0.1 m 뒤 정지)에서 시작한다', -0.5 < s_signed < 0.5,
      'arc %+.3f m (임의 시작이면 기대값 15 m)' % s_signed)
steps, offs, hits, first_pen, last = 0, 0, 0, None, None
term = trunc = False
while steps < env.episode_steps:
    _, _, term, trunc, last = env.step(ZERO)
    steps += 1
    offs += int(last['off'])
    hits += int(last['hit'])
    if first_pen is None and (last['off'] or last['hit']):
        first_pen = steps
    if term or trunc:
        break
check('G1 공식 결승선 완주 = terminated', bool(term and last['lap_done']),
      'term=%s trunc=%s adv=%s' % (term, trunc, last.get('adv')))
check('G1b 완주가 truncated 가 아니다 (P0-2)', not trunc,
      'SB3 가 완주를 timeout 으로 보면 부트스트랩한다')
check('G2 진행 arc >= 트랙 길이', bool(last and last['lap_s'] >= env.track.total - 0.3),
      'lap_s %.2f / total %.2f' % (last['lap_s'], env.track.total))
check('G2b 어드밴스가 심판 조건을 만족', last.get('adv', 0) > env.track.N / 2,
      'adv %s / N %d' % (last.get('adv'), env.track.N))
check('G3 랩 소요가 그럴듯하다 (30~120 초)', 600 <= steps <= 2400,
      '%d 스텝 = %.1f 초 (t_s %.1f)' % (steps, steps * DT, last.get('t_s', -1)))
if first_pen is not None:
    check('G4 페널티가 에피소드를 끊지 않는다', steps > first_pen + 5,
          '첫 페널티 %d 스텝 -> 총 %d 스텝 (off %d, hit %d)'
          % (first_pen, steps, offs, hits))
else:
    print('%-44s (이번 랩은 페널티가 없었다 — 검사 생략)' % 'G4')

env.close()
time.sleep(0.8)
# close() 가 월드를 **예선맵 원본**으로 넘겼는지
try:
    objs_g = {n: p for n, p in api('/state').get('objects', {}).items()
              if n.startswith('cone')}
    worst = max(_m.hypot(objs_g[n]['x'] - home[n][0],
                         objs_g[n]['y'] - home[n][1]) for n in home)
    check('E6 close() 후 월드가 예선맵 원본', worst <= CONE_MOVE_TOL,
          '최대 이탈 %.4f m' % worst)
except Exception as e:
    check('E6 close() 후 월드가 예선맵 원본', False, str(e))
print()
if FAIL:
    print('!! 실패 %d 건: %s' % (len(FAIL), FAIL))
    sys.exit(1)
print('전부 통과 — 랜덤화 배선이 참 플랜트를 건드리지 않는다')
