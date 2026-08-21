#!/usr/bin/env python3
"""전수검수 P0 계약 회귀 테스트 (GPT Test 1/2/4/7/12 상당).

    . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
    cd ~/physicar_ws/coweek
    python3 rl/test_contract.py --unit     # 시뮬 불필요 (기하 + fail-closed)
    python3 rl/test_contract.py            # 전부 (시뮬 필요, 약 2분)

무엇을 지키는 테스트인가:
  U*  결승선 판정이 심판과 같은 기하다 — 중앙 통과는 완주, **선분 끝점
      바깥 통과는 완주가 아니다** (P0-1 의 핵심: 그 사고가 학습에 존재해야
      한다), 텔레포트 선분은 크로싱이 아니다
  E*  평가 fail-closed 헬퍼가 실제로 닫혀 있다 (P0-5)
  S*  [SIM] 완주 = terminated / 안전망 = truncated (P0-2), 보상 합 = 공식
      비용 (P0-6/Test 12), 텔레포트 후 관측이 새 장면 + 히스토리 리셋 (P0-4),
      pose seq 가 매 스텝 전진 (P0-3)

⚠️ 시뮬 파트는 심판·run.sh 드라이버가 떠 있으면 안 된다. 먼저 kill_orphans.
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np

FAIL = []


def check(name, ok, detail=''):
    print('%-52s %s %s' % (name, 'PASS' if ok else '!! FAIL', detail),
          flush=True)
    if not ok:
        FAIL.append(name)


ap = argparse.ArgumentParser()
ap.add_argument('--unit', action='store_true', help='시뮬 없는 부분만')
args = ap.parse_args()

from coweek_env import (Track, FinishDetector, W_SCALE, DT,  # noqa: E402
                        CoweekResidualEnv)
import eval_policy as ep                                      # noqa: E402

# ════════════════════════════════════════════════════════════════════
# U — 결승선 판정 기하 (P0-1). 시뮬 불필요 (route.json 만 읽는다).
# ════════════════════════════════════════════════════════════════════
print('=== U  결승선 판정 = 심판 기하 ===', flush=True)
tr = Track()
N = tr.N
ax, ay = tr.inner0
bx, by = tr.outer0
line_len = math.hypot(bx - ax, by - ay)
ldx, ldy = (bx - ax) / line_len, (by - ay) / line_len   # 선분 방향 단위벡터
yaw0 = float(tr.yaw[0])                                  # 출발선에서의 트랙 방향
tdx, tdy = math.cos(yaw0), math.sin(yaw0)
mx, my = (ax + bx) / 2, (ay + by) / 2                    # 선분 중점


def fd_with_lap():
    """어드밴스가 찬 detector — 트랙을 성기게 한 바퀴 순회시킨다."""
    fd = FinishDetector(tr)
    done = False
    for i in range(0, N - 6, 3):        # 윈도 (10,15) 안의 스텝
        done = fd.update(float(tr.wp[i, 0]), float(tr.wp[i, 1]),
                         float(tr.yaw[i])) or done
    return fd, done


fd, premature = fd_with_lap()
check('U1 순회 중 완주 오발 없음', not premature, 'adv=%d' % fd.adv)
check('U1b 순회 후 adv > N/2', fd.adv > N / 2, 'adv=%d N=%d' % (fd.adv, N))

# U2: 중앙 통과 -> 완주. 선분 중점을 트랙 방향으로 곧게 가로지른다.
crossed = False
for k in (-4, -2, 2, 4):
    px = mx + tdx * (0.1 * k) - tdx * 0.12   # front 가 pose+0.12 이므로 보정
    py = my + tdy * (0.1 * k) - tdy * 0.12
    crossed = fd.update(px, py, yaw0) or crossed
check('U2 중앙 통과 -> 완주', crossed)

# U3: **선분 끝점 바깥** 통과 -> 완주가 아니다. 같은 궤적을 outer 끝점 밖
# 0.15 m 에서 반복한다 — 실측된 결승선 놓침 사고의 기하 그대로다.
fd3, _ = fd_with_lap()
ox, oy = bx + ldx * 0.15, by + ldy * 0.15
missed_finish = False
for k in (-4, -2, 2, 4):
    px = ox + tdx * (0.1 * k) - tdx * 0.12
    py = oy + tdy * (0.1 * k) - tdy * 0.12
    missed_finish = fd3.update(px, py, yaw0) or missed_finish
check('U3 끝점 밖 0.15 m 통과 -> 완주 아님 (놓침)', not missed_finish,
      'adv=%d — 이 사고가 이제 학습에 존재한다' % fd3.adv)
# U3b: 놓친 뒤에도 계속 돌 수 있고, 다음 번 중앙 통과에서 완주된다
recovered = False
for k in (-4, -2, 2, 4):
    px = mx + tdx * (0.1 * k) - tdx * 0.12
    py = my + tdy * (0.1 * k) - tdy * 0.12
    recovered = fd3.update(px, py, yaw0) or recovered
check('U3b 놓친 뒤 중앙 재통과 -> 완주', recovered)

# U4: 텔레포트 가드 — note_teleport() 뒤 첫 이동 선분은 크로싱이 아니다
fd4, _ = fd_with_lap()
fd4.update(mx - tdx * 0.4, my - tdy * 0.4, yaw0)     # 선 앞 0.4 m
fd4.note_teleport()
tele_cross = fd4.update(mx + tdx * 0.4, my + tdy * 0.4, yaw0)
check('U4 텔레포트 직후 선분은 크로싱 아님', not tele_cross)

# U5: 0.5 m 점프 자동 가드 (심판의 prev 리셋과 동일)
fd5, _ = fd_with_lap()
fd5.update(mx - tdx * 2.0, my - tdy * 2.0, yaw0)     # 선 앞 2.0 m
jump_cross = fd5.update(mx + tdx * 0.3, my + tdy * 0.3, yaw0)
check('U5 0.5 m 넘는 점프는 크로싱 아님', not jump_cross)

# ── U6  닫힌 루프의 중복 종점 yaw (적대 리뷰, 2026-08-22) ──────────
# route.json 은 wp[N]==wp[0] 인 닫힌 루프라 np.roll 로 yaw 를 계산하면
# yaw[N]=atan2(0,0)=0(동쪽)이 된다. 그 지점이 하필 결승선이라, 텔레포트가
# 그 인덱스를 고르면 차를 트랙과 89° 틀어 놓는다.
true_last = math.atan2(tr.wp[N, 1] - tr.wp[N - 1, 1],
                       tr.wp[N, 0] - tr.wp[N - 1, 0])
err_deg = abs(math.degrees(float(tr.yaw[N]) - true_last))
err_deg = min(err_deg, 360 - err_deg)
check('U6 중복 종점 yaw 가 트랙 방향', err_deg < 2.0,
      'yaw[N]=%.4f vs 실제 %.4f -> 오차 %.2f°' % (tr.yaw[N], true_last, err_deg))
# 텔레포트 walk 는 중복점을 아예 안 밟아야 한다 (심판 fwd_index 는 % N)
walk = {tr.fwd_index(i, 0.30) for i in range(N)}
check('U6b fwd_index 가 중복 종점을 안 쓴다', N not in walk,
      '도달 인덱스 최대 %d (N=%d)' % (max(walk), N))

# ── U7  이탈 판정 메트릭 = 심판 (적대 리뷰, critical) ──────────────
# 심판은 최근접 wp 까지의 **방사거리**, 종전 env 는 법선 사영 |e| 를 썼다.
# 사영은 항상 그 거리 이하라 env 가 한 방향으로만 관대했다.
def ref_off(x, y):
    """referee.off_track 그대로 (방사거리)."""
    d2 = ((tr.wp[:N, 0] - x) ** 2 + (tr.wp[:N, 1] - y) ** 2)
    i = int(np.argmin(d2))
    return math.sqrt(float(d2[i])) > float(tr.half[i]) + 0.12


mismatch = 0
tested = 0
for i in range(0, N, 7):
    for f in (0.0, 0.5):
        j = (i + 1) % N
        bx = tr.wp[i, 0] + (tr.wp[j, 0] - tr.wp[i, 0]) * f
        by = tr.wp[i, 1] + (tr.wp[j, 1] - tr.wp[i, 1]) * f
        for lat in (0.30, 0.40, 0.44, 0.46, 0.47, 0.48, 0.50, 0.60):
            for sgn in (1, -1):
                px = bx + tr.u[i, 0] * lat * sgn
                py = by + tr.u[i, 1] * lat * sgn
                tested += 1
                if tr.off_track(px, py) != ref_off(px, py):
                    mismatch += 1
check('U7 이탈 판정이 심판과 완전 일치', mismatch == 0,
      '불일치 %d / %d 표본' % (mismatch, tested))

# ════════════════════════════════════════════════════════════════════
# E — 평가 fail-closed (P0-5). 시뮬 불필요.
# ════════════════════════════════════════════════════════════════════
print('\n=== E  평가 fail-closed 헬퍼 ===', flush=True)
tmpd = tempfile.mkdtemp(prefix='coweek_contract_')


_n = [0]


def _status(*recs):
    """append 로그(jsonl)를 만든다 — 러너는 배치당 프로세스가 여러 개다."""
    _n[0] += 1
    p = os.path.join(tmpd, 'status_%d.jsonl' % _n[0])
    with open(p, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    return p


t0 = time.time() - 60


def rec(**kw):
    d = {'pid': 1000 + len(kw), 'event': 'end', 'model': '/x/final.zip',
         'loaded': True, 'ticks': 900, 'res_abs_mean': 0.05,
         'res_max': ep.RES_MAX_CODE, 'contract': ep.CONTRACT,
         'predict_errors': 0, 't': time.time()}
    d.update(kw)
    return d


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
        return False
    except ep.EvalError:
        return True


check('E1 상태 로그 없음 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             os.path.join(tmpd, 'none.jsonl')))
check('E2 로드 실패 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(loaded=False))))
check('E3 predict 예외 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(predict_errors=3))))
check('E4 잔차가 사실상 0 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(res_abs_mean=0.0))))
check('E5 다른 모델 로드 -> 무효',
      raises(ep.check_runner_status, '/y/final.zip', t0, _status(rec())))
check('E6 정상 상태 -> 통과',
      not raises(ep.check_runner_status, '/x/final.zip', t0, _status(rec())))
check('E7 기준선 arm 에 모델 로드 -> 무효',
      raises(ep.check_runner_status, None, t0, _status(rec())))
check('E8 기준선 arm 정상 -> 통과',
      not raises(ep.check_runner_status, None, t0,
                 _status(rec(model='', loaded=False, res_abs_mean=0.0))))
# ── 적대 리뷰가 짚은 구멍: 한 프로세스만 고장나도 잡혀야 한다 ──
check('E8b 중간 프로세스 1개만 로드 실패해도 무효 (덮임 방지)',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(pid=1, ticks=900), rec(pid=2, loaded=False, ticks=10),
                     rec(pid=3, ticks=900))),
      '종전에는 마지막 프로세스가 파일을 덮어 통과했다')
check('E8c 중간 프로세스의 predict 예외도 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(pid=1), rec(pid=2, predict_errors=7), rec(pid=3))))
check('E8d 이전 배치 레코드만 있으면 무효 (신선도)',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(t=t0 - 600))))
check('E8e 러너 프로세스 수 < 런 수 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(pid=1), rec(pid=2)), 3))
check('E8f 계약 불일치 -> 무효',
      raises(ep.check_runner_status, '/x/final.zip', t0,
             _status(rec(contract='old_contract_v0'))))

def _mf(**kw):
    """매니페스트를 가진 가짜 모델 디렉터리를 만든다."""
    d = {'status': 'READY', 'contract': ep.CONTRACT,
         'res_max': ep.RES_MAX_CODE, 'obs_row_dim': 23, 'hist': 3,
         'driver_md5': ep._md5(os.path.join(ep.WS, 'coweek/driver.py')),
         'arc_md5': ep._md5(os.path.join(ep.WS, 'coweek/arc_planner.py'))}
    d.update(kw)
    dd = tempfile.mkdtemp(dir=tmpd)
    with open(os.path.join(dd, 'manifest.json'), 'w') as f:
        json.dump(d, f)
    return os.path.join(dd, 'final.zip')


check('E12 정상 매니페스트 -> 통과',
      not raises(ep.check_manifest, _mf()))
check('E13 매니페스트 없음 -> 무효 (구계약 모델 차단)',
      raises(ep.check_manifest, os.path.join(tmpd, 'nomf', 'final.zip')))
check('E14 status=INCOMPLETE -> 무효',
      raises(ep.check_manifest, _mf(status='INCOMPLETE')))
check('E15 계약 불일치 -> 무효',
      raises(ep.check_manifest, _mf(contract='official_cost_v0')))
check('E16 잔차 스케일 불일치 -> 무효',
      raises(ep.check_manifest, _mf(res_max=0.25)))
check('E17 driver.py 변경 -> 무효',
      raises(ep.check_manifest, _mf(driver_md5='0' * 32)))
check('E18 arc_planner.py 변경 -> 무효 (관측 의미가 바뀐다)',
      raises(ep.check_manifest, _mf(arc_md5='0' * 32)))

tj = os.path.join(tmpd, 'traces.jsonl')
with open(tj, 'w') as f:
    for k in range(2):
        f.write(json.dumps({'run': k + 1, 'score': 50.0, 'penalty': 5.0,
                            'trace': []}) + '\n')
check('E9 트레이스 런 수 불일치 -> 무효',
      raises(ep.analyse, tj, 'x', 3))

ps = ep.paired_stats([50.0, 51.0, 49.0, 50.0], [52.0, 53.0, 51.0, 52.0])
check('E10 짝지은 통계 (평균 -2, CI<0)',
      abs(ps['mean'] + 2.0) < 1e-9 and ps['hi'] < 0,
      'mean %+.2f CI [%+.2f, %+.2f]' % (ps['mean'], ps['lo'], ps['hi']))
ps2 = ep.paired_stats([50.0, 54.0], [52.0, 52.0])
check('E11 불확정(CI 가 0 걸침)은 PASS 조건이 아니다',
      not (ps2['mean'] < 0 and ps2['hi'] < 0),
      'mean %+.2f hi %+.2f' % (ps2['mean'], ps2['hi']))

# ════════════════════════════════════════════════════════════════════
# R — 적대 리뷰 회귀 (2026-08-22). 전부 **구조**를 지키는 검사다:
#     되돌아가면 조용히 학습이 오염되거나 장시간 학습이 죽는 것들이라
#     동작 테스트로는 잡기 어렵고(희귀·타이밍 의존) 소스로 못박는다.
# ════════════════════════════════════════════════════════════════════
print('\n=== R  적대 리뷰 회귀 ===', flush=True)
import inspect                                          # noqa: E402
import re as _re                                        # noqa: E402
import coweek_env as ce                                 # noqa: E402

src_env = inspect.getsource(ce)
rs = inspect.getsource(ce.CoweekResidualEnv.reset)
ss = inspect.getsource(ce.CoweekResidualEnv.step)

# R1: reset() 이 월드를 건드리기 **전에** 차를 세우고 안전지대로 옮긴다
i_stop = rs.find('_real_speed.publish')
i_park = rs.find('_teleport_to(0,')
i_cone = min([x for x in (rs.find('cones.reshuffle'), rs.find('cones.restore'))
              if x >= 0] or [-1])
check('R1 정지가 콘 재배치보다 먼저', 0 <= i_stop < i_cone,
      'stop@%d cone@%d — 굴러가는 차 옆에서 콘을 뿌리면 정본이 오염된다'
      % (i_stop, i_cone))
check('R1b 출발선 파킹이 콘 재배치보다 먼저', 0 <= i_park < i_cone,
      'park@%d cone@%d — 파킹 지점은 3.4 m 콘 금지 구역이다' % (i_park, i_cone))

# R2: 장시간 학습 생존성 — API 재시도와 timeout 정합
check('R2 api() 에 409·네트워크 재시도', 'RETRY_409' in src_env
      and 'tries' in inspect.signature(ce.api).parameters,
      '재시도가 없으면 일시적 409 하나가 2.25 시간 학습을 끝낸다')
fp_to = inspect.signature(ce.CoweekResidualEnv._fresh_pose).parameters['timeout'].default
poll_to = [float(x) for x in _re.findall(r"api\('/state',\s*timeout=([\d.]+)\)", src_env)]
check('R2b _fresh_pose timeout > 폴러 HTTP timeout',
      bool(poll_to) and fp_to > max(poll_to),
      '_fresh_pose %.1f s vs 폴러 %.1f s — 역전되면 느린 응답 하나가 학습을 죽인다'
      % (fp_to, max(poll_to) if poll_to else -1))

# R3: 신선도를 seq 가 아니라 **폴 시작 시각**으로 판정한다
check('R3 폴 시작 시각 기반 신선도(get_after)',
      hasattr(ce._StatePoller, 'get_after') and 'get_after' in src_env
      and '_fresh_pose(t_cmd)' in ss,
      'seq 비교는 명령 순간 전송 중이던 폴(=명령 이전 세계)을 통과시킨다')

# R4: STALE 이면 학습도 잔차 0 — 배포와 같은 규칙
check('R4 env 에도 STALE 게이트', 'STALE_NO_RESIDUAL' in ss,
      '없으면 정책이 배포에서 무시되는 행동을 배우고 driver 안전상한을 뚫는다')
_rp2 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'run_policy.py'), encoding='utf-8').read()
check('R4b 배포가 그 상수를 env 에서 import',
      'STALE_NO_RESIDUAL' in _rp2.split('class ')[0]
      and 'from coweek_env import' in _rp2,
      '두 곳에 따로 적으면 조용히 어긋난다')

# R5: _lap_s 는 텔레포트 **전** 주행분을 먼저 세야 한다.
# 리뷰가 "이중 계상"이라며 순서를 뒤집으라고 했는데 **그게 틀렸다** — 두 항은
# 연속된 다른 구간이라, 뒤집으면 정상 주행분이 버려져 한 바퀴가 27.06 m 로
# 모자란다 (test_env_dyn G2 가 잡았다). 그 되돌림을 여기서 못박는다.
i_zero = ss.find('ds = 0.0')
i_lap = ss.find('self._lap_s +=')
check('R5 _lap_s 누적이 ds 오염 제거보다 먼저', 0 <= i_lap < i_zero,
      'lap@%d zero@%d — 뒤집으면 텔레포트 전 주행분이 통째로 버려진다'
      % (i_lap, i_zero))

if args.unit:
    print()
    if FAIL:
        print('!! 실패 %d 건: %s' % (len(FAIL), FAIL))
        sys.exit(1)
    print('유닛 파트 전부 통과 (시뮬 파트는 --unit 없이)')
    sys.exit(0)

# ════════════════════════════════════════════════════════════════════
# S — 시뮬 인더루프 (P0-2 / P0-3 / P0-4 / P0-6)
# ════════════════════════════════════════════════════════════════════
print('\n=== S  시뮬 계약 (약 2분) ===', flush=True)
ZERO = np.zeros(1, np.float32)
env = CoweekResidualEnv(seed=21, dyn=False, cone_reshuffle=0)

# S1: 완주 = terminated (P0-2) + 보상 합 = 공식 비용 (P0-6, GPT Test 12)
obs, _ = env.reset()
tot_r, offs, hits, steps = 0.0, 0, 0, 0
term = trunc = False
info = {}
while not (term or trunc):
    obs, r, term, trunc, info = env.step(ZERO)
    tot_r += r
    offs += int(info['off'])
    hits += int(info['n_hit'])     # 심판처럼 **콘 개수**로 센다 (P2-5)
    steps += 1
check('S1 완주 = terminated', term and info['lap_done'],
      'steps=%d t_s=%.1f adv=%d' % (steps, info['t_s'], info['adv']))
check('S1b 완주는 truncated 가 아니다', not trunc)
official = info['t_s'] + 5.0 * (offs + hits)
recon = abs(-tot_r / W_SCALE - official)
check('S1c 보상 합 = -(공식 비용) (Test 12)', recon < 1e-6,
      '|Σr/W + cost| = %.2e (cost %.2f)' % (recon, official))
check('S1d 랩 시간이 그럴듯하다', 30.0 <= info['t_s'] <= 120.0,
      't_s %.1f' % info['t_s'])

# S2: 안전망 = truncated (P0-2)
env.max_episode_s = 3.0
obs, _ = env.reset()
term = trunc = False
guard = 0
while not (term or trunc) and guard < 200:
    obs, r, term, trunc, info = env.step(ZERO)
    guard += 1
check('S2 안전망 = truncated (terminated 아님)', trunc and not term,
      't_s %.2f steps %d' % (info['t_s'], guard))
check('S2b info.timeout 표기', bool(info['timeout']))
env.max_episode_s = 150.0

# S3: pose 신선도 (P0-3) — 스텝마다 seq 가 전진한다
obs, _ = env.reset()
seqs = []
for _ in range(20):
    s0 = env.poll.get_seq()[1]
    env.step(ZERO)
    seqs.append(env.poll.get_seq()[1] - s0)
check('S3 스텝마다 새 pose seq (최소 전진 %d)' % min(seqs),
      all(d >= 1 for d in seqs))
check('S3b 폴 실패 없음', env.poll.fail_n == 0, 'fail_n=%d' % env.poll.fail_n)

# S4: 텔레포트 후 관측 리셋 (P0-4) — 차를 잔디로 옮겨 이탈을 강제한다
for _ in range(10):
    env.step(ZERO)
p = env._wait_pose()
i_now, _, _, _ = env.track.locate(p[0], p[1])
env._teleport_to((i_now + 8) % len(env.track.wp), lat=0.60)   # 반폭+0.12 밖
t_before = time.monotonic()
obs, r, term, trunc, info = env.step(ZERO)
check('S4 강제 이탈이 잡힌다', info['off'], 'off=%s e=%.3f' % (info['off'], info['e']))
check('S4b 회복 틱이 시간 비용에 들어간다', info['extra_ticks'] >= 1,
      'extra=%d' % info['extra_ticks'])
lt = env._features()
check('S4c 관측 프레임이 텔레포트 이후다',
      lt['frame_t'] is not None and lt['frame_t'] > t_before,
      'frame age %.3f s' % (time.monotonic() - (lt['frame_t'] or 0)))
check('S4d 행동 이력 리셋 (prev_a=0)', env._prev_a == 0.0)
rows = env._hist
same = all(np.array_equal(rows[0], rw) for rw in rows)
check('S4e 히스토리가 새 장면으로만 채워짐', len(rows) >= 1 and same,
      'rows=%d 동일=%s' % (len(rows), same))

env.close()
time.sleep(0.8)
# 월드 복원 확인 (test_env_dyn E6 상당) — close() 가 예선맵 원본으로 넘겼는가
from coweek_env import api as _api, STATE_JSON, CONE_MOVE_TOL   # noqa: E402
home = {n: p for n, p in json.load(open(STATE_JSON))['objects'].items()
        if n.startswith('cone')}
objs = {n: p for n, p in _api('/state').get('objects', {}).items()
        if n.startswith('cone')}
worst = max(math.hypot(objs[n]['x'] - home[n]['x'],
                       objs[n]['y'] - home[n]['y'])
            for n in home if n in objs) if objs else 9.9
check('S5 close() 후 월드가 예선맵 원본', worst <= CONE_MOVE_TOL,
      '최대 이탈 %.4f m' % worst)

print()
if FAIL:
    print('!! 실패 %d 건: %s' % (len(FAIL), FAIL))
    sys.exit(1)
print('전부 통과 — 계약(P0-1/2/3/4/6)이 코드에 박혀 있다')
