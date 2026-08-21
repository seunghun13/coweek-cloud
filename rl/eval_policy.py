#!/usr/bin/env python3
"""학습된 잔차 정책을 **공식 채점기로** 평가한다.

학습 곡선은 채택 근거가 아니다. 이 프로젝트에서 오프라인 지표가 역대 최고를
찍고도 실주행에서 손해를 낸 전례가 여러 건이다. 채택은 여기서만 한다.

    python3 rl/eval_policy.py --model rl/runs/s1/final.zip --runs 15 --baseline
    python3 rl/eval_policy.py --model rl/runs/s1/final.zip --runs 15 --finals --baseline
    python3 rl/eval_policy.py --baseline --runs 15            # 잔차 0 대조군만

## 평가는 fail-closed 다 (전수검수 P0-5)
종전에는 모델 로드가 실패해도 러너가 잔차 0(=v64)으로 달리고, 게이트가 `<=`
라 **동률도 PASS** 였다 — 고장난 정책이 "안전하게 전 게이트 통과"로 나올 수
있었다. 이제 다음 중 하나라도 걸리면 그 평가는 **무효**로 죽는다:
  모델 로드 실패 · predict 예외 · 러너 상태 파일 부재/낡음 · 잔차가 사실상
  0 만 나옴 · 이전 배치의 trace 재사용 · 심판 비정상 종료 · 고아 청소 실패 ·
  trace 런 수 불일치 · 모델 매니페스트의 driver 불일치

게이트 (GPT 권고 + 우리 실측 기준선)
    Gate F  결승선 놓침       기준선 대비 증가하지 않을 것 (지뢰선 — n 이
                              작아 "개선"의 증거로는 못 쓴다)
    Gate P  페널티 합         기준선 대비 증가하지 않을 것
    Gate S  공식 점수 평균    **짝지은 차이의 평균 < 0 이고 95 % CI 상한 < 0**
                              (동률·불확정은 불합격 — P0-5)
모두 **같은 배치 안에서**, 같은 `--seed` 로 짝지어 잰다. 배치 간 절대 비교는
못 믿는다 (같은 빌드가 세 배치에서 페널티 130/170/200 이었다).
"""
import argparse
import hashlib
import json
import math
import os
import statistics as st
import subprocess
import sys
import time

WS = '/home/physicar/physicar_ws'
ACTIVE = os.path.join(WS, 'coweek/rl/ACTIVE_MODEL')
LAP = 30.505
TRACES = '/tmp/ref_traces.jsonl'
RUNNER_STATUS = '/tmp/rl_runner_status.jsonl'

# 계약·잔차 스케일은 **코드에서** 읽는다 (매니페스트와 대조하기 위해).
sys.path.insert(0, os.path.join(WS, 'coweek'))
sys.path.insert(0, os.path.join(WS, 'coweek/rl'))
try:                       # driver/rclpy 없이도 상수는 읽히도록
    from coweek_env import CONTRACT, RES_MAX as RES_MAX_CODE
except Exception:          # ROS 환경이 아니면 소스에서 직접 긁는다
    import re as _re
    _src = open(os.path.join(WS, 'coweek/rl/coweek_env.py'), encoding='utf-8').read()
    CONTRACT = _re.search(r"^CONTRACT\s*=\s*'([^']+)'", _src, _re.M).group(1)
    RES_MAX_CODE = float(_re.search(r'^RES_MAX\s*=\s*([\d.]+)', _src, _re.M).group(1))

# t 분포 95 % 양측 임계값 (df -> t). 표에 없는 df 는 가장 가까운 아래 값을 쓴다.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045, 40: 2.021, 60: 2.000}


def t95(df):
    if df <= 0:
        return float('inf')
    keys = sorted(T95)
    best = keys[0]
    for k in keys:
        if k <= df:
            best = k
    return T95[best]


class EvalError(RuntimeError):
    """평가 무효 — fail-closed. 이 예외로 끝난 배치는 채택 판단에 쓰지 않는다."""


def drive_dist(tr):
    d = 0.0
    for i in range(1, len(tr)):
        s = math.hypot(tr[i][1] - tr[i - 1][1], tr[i][2] - tr[i - 1][2])
        if s <= 0.25:            # 그 이상은 주행이 아니라 심판 텔레포트다
            d += s
    return d


def analyse(path, tag, expect_runs=None):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    if expect_runs is not None and len(runs) != expect_runs:
        raise EvalError('트레이스가 %d 런인데 %d 런을 기대했다 (%s) — 배치 불완전'
                        % (len(runs), expect_runs, path))
    clean = [r for r in runs if drive_dist(r.get('trace') or []) <= 1.30 * LAP]
    miss = len(runs) - len(clean)
    pure = [r['score'] - r['penalty'] for r in clean if r['score'] is not None]
    # 짝지은 비교용 런별 점수. timeout(score None) 은 심판 한도 180 으로 친다.
    scores = [(r['score'] if r.get('score') is not None else 180.0)
              for r in runs]
    out = {
        'tag': tag, 'n': len(runs),
        'mean': st.mean(scores),
        'miss': miss,
        'clean_mean': st.mean(r['score'] for r in clean
                              if r['score'] is not None) if clean else float('nan'),
        'pure': st.mean(pure) if pure else float('nan'),
        'pen': sum(r['penalty'] for r in clean),
        'nopen': sum(1 for r in clean if r['penalty'] == 0),
        'scores': scores,
    }
    print('%-14s n=%d | 총평균 %6.2f | 놓침 %d/%d | 놓침제외 %6.2f | '
          '순수랩 %6.2f | 페널티 %5.1f | 무페널티 %d'
          % (tag, out['n'], out['mean'], miss, out['n'], out['clean_mean'],
             out['pure'], out['pen'], out['nopen']))
    return out


def _run_checked(cmd, what):
    rc = subprocess.run(cmd, cwd=WS)
    if rc.returncode != 0:
        raise EvalError('%s 실패 (rc=%d) — fail-closed' % (what, rc.returncode))


def run_batch(runs, finals, log, traces, seed=None):
    """한 arm 을 돌린다. `seed` 를 주면 두 arm 이 **같은 콘 배치열**을 본다.

    시작 시각을 반환한다 (러너 상태 파일 신선도 검사용)."""
    # P0-5: 이전 배치의 산출물을 재사용할 길을 먼저 끊는다
    for p in (TRACES, traces, RUNNER_STATUS):
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    _run_checked([sys.executable, 'coweek/kill_orphans.py'], '고아 청소(전)')
    cmd = [sys.executable, '-u', 'coweek/referee.py', '--runs', str(runs)]
    if finals:
        # 랜덤 배치 + **cone6 만 고정** (referee 의 `--pin` 기본값이 cone6 라
        # 넘기지 않는다). 대회 규칙이 아니라 측정 전략이다 — cone6 가 미해결
        # 병목이라 흩어버리면 성적이 실제보다 좋아 보인다. 학습 env 도 같은
        # 분포를 쓴다 (P0-8). 6개 전부 랜덤을 보려면 `--pin ''` 를 넣어라.
        cmd += ['--random-cones']
    if seed is not None:
        cmd += ['--seed', str(seed)]
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, cwd=WS, stdout=f, stderr=subprocess.STDOUT)
    if rc.returncode != 0:
        raise EvalError('심판 비정상 종료 (rc=%d) — %s 확인. 이 배치는 무효다'
                        % (rc.returncode, log))
    _run_checked([sys.executable, 'coweek/kill_orphans.py'], '고아 청소(후)')
    if not os.path.exists(TRACES):
        raise EvalError('심판이 트레이스를 안 남겼다 — 배치가 실패했다')
    if os.path.getmtime(TRACES) < t0:
        raise EvalError('트레이스가 이번 실행보다 오래됐다 — 이전 배치 재사용 차단')
    subprocess.run(['cp', TRACES, traces], check=True)
    return t0


def check_runner_status(model_path, t0, status_path=RUNNER_STATUS, runs=None):
    """이 arm 의 **모든 러너 프로세스**가 정말 그 정책으로 달렸는지 (P0-5).

    심판은 런마다 run.sh 를 새로 spawn 하고 run.sh 는 죽은 러너를 재시작하므로
    러너 프로세스는 배치당 여러 개다. 그래서 상태는 **append 로그**(jsonl)이고
    여기서는 **레코드 전부**를 본다 — 한 프로세스만 봤다면 3번째 런의 로드
    실패를 4번째 런의 깨끗한 상태가 덮어 무효 배치가 유효로 통과한다."""
    if not os.path.exists(status_path):
        raise EvalError('러너 상태 로그가 없다 — 정책이 실행됐는지 알 수 없다. '
                        'run.sh 디스패치를 확인하라')
    recs = []
    for line in open(status_path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            raise EvalError('러너 상태 로그가 깨졌다: ' + line[:120])
        if r.get('t', 0) >= t0 - 1.0:       # 이번 배치 것만
            recs.append(r)
    if not recs:
        raise EvalError('이번 배치의 러너 레코드가 없다 (로그는 있으나 전부 '
                        '이전 실행) — 러너가 안 떴다')
    pids = {r.get('pid') for r in recs}
    errs = max(r.get('predict_errors', 0) for r in recs)
    if errs:
        raise EvalError('predict 예외 %d 건 (프로세스 %d개 중) — 정책이 온전히 '
                        '돌지 않았다' % (errs, len(pids)))
    contracts = {r.get('contract') for r in recs}
    if contracts - {None} and contracts != {CONTRACT}:
        raise EvalError('러너 계약 불일치: %s (기대 %s)' % (contracts, CONTRACT))
    if runs is not None and len(pids) < runs:
        raise EvalError('러너 프로세스가 %d개뿐인데 %d런을 돌렸다 — 어떤 런은 '
                        '정책 없이 달렸을 수 있다' % (len(pids), runs))
    if model_path is None:                      # 기준선 arm (잔차 0)
        bad = [r for r in recs if r.get('loaded')]
        if bad:
            raise EvalError('기준선 arm 인데 모델이 로드된 프로세스가 있다: %s'
                            % bad[0].get('model'))
        return
    unloaded = [r for r in recs if not r.get('loaded')]
    if unloaded:
        raise EvalError('모델 로드 실패 프로세스 %d개 (pid %s) — 그 런들은 '
                        'v64 로 달렸다' % (len(unloaded), unloaded[0].get('pid')))
    wrong = [r for r in recs
             if os.path.abspath(r.get('model', '')) != os.path.abspath(model_path)]
    if wrong:
        raise EvalError('다른 모델이 로드됐다: %s (기대 %s)'
                        % (wrong[0].get('model'), model_path))
    res_max = {r.get('res_max') for r in recs} - {None}
    if res_max and res_max != {RES_MAX_CODE}:
        raise EvalError('러너 잔차 스케일이 코드 기본값과 다르다: %s (기대 %s)'
                        % (res_max, RES_MAX_CODE))
    last = max(recs, key=lambda r: (r.get('ticks', 0), r.get('t', 0)))
    if last.get('res_abs_mean', 0.0) < 1e-4:
        raise EvalError('잔차가 사실상 0 만 나왔다 (평균 |res| %.2e) — 고장난 '
                        '정책과 구분할 수 없어 무효 처리한다'
                        % last.get('res_abs_mean', 0.0))


def _md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def check_manifest(model_path):
    """모델이 **지금 코드와 같은 계약**으로 학습됐는지 대조한다 (P1-9).

    SB3 zip 은 shape 만 지킨다. 관측 차원이 같아도 계약(보상·종료)·잔차
    스케일·베이스 제어기가 바뀐 모델은 정상 로드되어 조용히 오작동한다.
    `driver.py` 뿐 아니라 `arc_planner.py` 도 본다 — 관측의 `aim`·`arc_safe`·
    `mode` 가 그쪽 산물이라 그게 바뀌면 관측 의미가 바뀐다."""
    mf = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                      'manifest.json')
    if not os.path.exists(mf):
        raise EvalError('manifest.json 이 없다 — 어떤 계약으로 학습된 모델인지 '
                        '알 수 없다. 계약 도입 이전 모델(s1~s6)은 재사용 금지다')
    m = json.load(open(mf))
    if m.get('status') != 'READY':
        raise EvalError('매니페스트 status=%s — 완료되지 않은 학습 산출물이다'
                        % m.get('status'))
    if m.get('contract') != CONTRACT:
        raise EvalError('계약 불일치: 모델 %s vs 코드 %s — 보상·종료 의미가 '
                        '다른 모델이다' % (m.get('contract'), CONTRACT))
    if abs(float(m.get('res_max', -1)) - RES_MAX_CODE) > 1e-9:
        raise EvalError('잔차 스케일 불일치: 모델 %s vs 코드 %s — 모든 행동이 '
                        '말없이 축소/확대된다' % (m.get('res_max'), RES_MAX_CODE))
    if int(m.get('obs_row_dim', -1)) * int(m.get('hist', -1)) <= 0:
        raise EvalError('매니페스트에 관측 스키마가 없다')
    for key, rel in (('driver_md5', 'coweek/driver.py'),
                     ('arc_md5', 'coweek/arc_planner.py')):
        cur = _md5(os.path.join(WS, rel))
        if m.get(key) and m[key] != cur:
            raise EvalError('%s 가 학습 때(%s)와 다르다(%s) — 관측 의미가 '
                            '어긋난 모델은 조용히 오작동한다'
                            % (rel, m[key][:8], cur[:8]))


def paired_stats(pol_scores, base_scores):
    """짝지은 런별 차이 (정책 − 기준선). 평균·95 % CI 를 준다."""
    d = [p - b for p, b in zip(pol_scores, base_scores)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d) if n > 1 else float('inf')
    half = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float('inf')
    return {'d': d, 'n': n, 'mean': m, 'sd': sd,
            'lo': m - half, 'hi': m + half}


def set_active(val):
    if val is None:
        if os.path.exists(ACTIVE):
            os.remove(ACTIVE)
    else:
        with open(ACTIVE, 'w') as f:
            f.write(val + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    ap.add_argument('--baseline', action='store_true',
                    help='잔차 0 러너로 대조군을 함께 돌린다')
    ap.add_argument('--runs', type=int, default=15)
    ap.add_argument('--finals', action='store_true')
    ap.add_argument('--seed', type=int, default=1234,
                    help='두 arm 이 같은 콘 배치열을 보게 한다 (짝지은 A/B). '
                         '0 을 주면 짝짓기를 끄고 옛 방식으로 돈다')
    a = ap.parse_args()

    seed = a.seed if a.seed else None
    cond = '본선(랜덤콘)' if a.finals else '예선맵'
    print('배치 짝짓기: %s'
          % ('seed %d — 두 arm 이 동일한 배치열' % seed if seed is not None
             else 'OFF (arm 마다 다른 배치)'), flush=True)
    prev = open(ACTIVE).read().strip() if os.path.exists(ACTIVE) else None
    results = []
    try:
        if a.model:
            m = os.path.abspath(a.model)
            if not os.path.isfile(m):
                raise EvalError('모델이 없다: ' + m)
            check_manifest(m)
            print('=== %s | 정책 %s ===' % (cond, m), flush=True)
            set_active(m)
            t0 = run_batch(a.runs, a.finals, '/tmp/eval_pol.log',
                           '/tmp/eval_pol_traces.jsonl', seed)
            check_runner_status(m, t0, runs=a.runs)
            results.append(analyse('/tmp/eval_pol_traces.jsonl', 'policy',
                                   expect_runs=a.runs))
        if a.baseline or not a.model:
            print('=== %s | 잔차 0 (= racer-v64) ===' % cond, flush=True)
            set_active('none')
            t0 = run_batch(a.runs, a.finals, '/tmp/eval_base.log',
                           '/tmp/eval_base_traces.jsonl', seed)
            check_runner_status(None, t0, runs=a.runs)
            results.append(analyse('/tmp/eval_base_traces.jsonl', 'baseline',
                                   expect_runs=a.runs))
    except EvalError as e:
        print('\n!! 평가 무효 (fail-closed): %s' % e, flush=True)
        set_active(prev)
        print('ACTIVE_MODEL 복원:', prev or '(삭제 — 기본 driver.py)')
        sys.exit(2)
    finally:
        set_active(prev)
    print('\nACTIVE_MODEL 복원:', prev or '(삭제 — 기본 driver.py)')

    if len(results) == 2:
        p, b = results[0], results[1]
        ps = paired_stats(p['scores'], b['scores'])
        print()
        print('=== 게이트 (같은 배치 안, seed 짝지은 비교) ===')
        okF = p['miss'] <= b['miss']
        print('  %-20s 정책 %8d  vs  기준 %8d   %s'
              % ('Gate F 결승선 놓침', p['miss'], b['miss'],
                 'PASS' if okF else 'FAIL'))
        okP = p['pen'] <= b['pen']
        print('  %-20s 정책 %8.1f  vs  기준 %8.1f   %s'
              % ('Gate P 페널티 합', p['pen'], b['pen'],
                 'PASS' if okP else 'FAIL'))
        # Gate S: 동률·불확정 불합격 (P0-5). 평균이 낮아도 CI 가 0 을 걸치면
        # "판정 불가"지 통과가 아니다.
        if ps['mean'] < 0 and ps['hi'] < 0:
            verdict = 'PASS'
        elif ps['mean'] < 0:
            verdict = 'INCONCLUSIVE (CI 가 0 을 걸침 — 불합격)'
        else:
            verdict = 'FAIL'
        print('  %-20s 짝지은 평균차 %+.2f 초, 95%% CI [%+.2f, %+.2f]   %s'
              % ('Gate S 공식 점수', ps['mean'], ps['lo'], ps['hi'], verdict))
        print()
        print('  런별 차이(정책−기준):',
              ' '.join('%+.1f' % v for v in ps['d']))
        print('  ⚠️ n=%d 는 결승선 놓침(기준선 3 %%)을 판정하기에 부족하다.'
              % p['n'])
        print('     놓침은 전용 벤치로 수백 회 재야 한다.')


if __name__ == '__main__':
    main()
