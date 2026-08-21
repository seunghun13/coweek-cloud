#!/usr/bin/env python3
"""GPT 검수용 소스 번들을 만든다.

    cd ~/physicar_ws/coweek && python3 rl/make_bundle.py
"""
import os
import re
import subprocess

WS = '/home/physicar/physicar_ws'
CW = os.path.join(WS, 'coweek')
OUT = ('/mnt/c/Users/정승훈/Desktop/coweek/GPT_첨부_RL소스.md')

FILES = ['rl/coweek_env.py', 'rl/dyn.py', 'rl/train_speed.py',
         'rl/run_policy.py', 'rl/eval_policy.py', 'rl/quick_gate.py',
         'rl/test_dyn.py', 'rl/test_env_dyn.py']


def read(p):
    with open(os.path.join(CW, p), encoding='utf-8') as f:
        return f.read()


def lines(p, a, b):
    return '\n'.join(read(p).splitlines()[a - 1:b])


def between(p, start_re, end_re):
    src = read(p).splitlines()
    out, on = [], False
    for ln in src:
        if not on and re.search(start_re, ln):
            on = True
        if on:
            out.append(ln)
            if len(out) > 1 and re.search(end_re, ln):
                break
    return '\n'.join(out)


tag = subprocess.run(['git', '-C', CW, 'describe', '--tags', '--always'],
                     capture_output=True, text=True).stdout.strip()

P = []
w = P.append
w('# 코윅윅 RL 소스 번들 (검수용)')
w('')
w('> `coweek/rl/` 전체 + `driver.py`·`run.sh` 관련 발췌.')
w('> 커밋 `%s` 기준, 2026-08-21.' % tag)
w('> 의뢰문은 `GPT_분석의뢰_RL전수검수.md` 다. 둘을 같이 봐야 한다.')
w('')
w('## 파일 목록')
w('')
w('| 파일 | 줄 | 역할 |')
w('|---|---|---|')
ROLE = {
    'rl/coweek_env.py': 'Gymnasium 환경 (핵심). 관측·보상·에피소드·콘/월드 관리',
    'rl/dyn.py': '동역학 도메인 랜덤화 (가감속·서보·지연)',
    'rl/train_speed.py': 'SAC 학습 스크립트',
    'rl/run_policy.py': '**배포 경로**. driver.py 위에 잔차를 얹어 실행',
    'rl/eval_policy.py': '공식 심판으로 채점하는 채택 게이트',
    'rl/quick_gate.py': '심판 전 몇 분짜리 트리아지 (정책 vs 잔차0, 짝지음)',
    'rl/test_dyn.py': '`dyn.py` 단위 테스트 20건 (ROS 불필요)',
    'rl/test_env_dyn.py': '환경 배선 시뮬 인더루프 테스트 24건',
}
for f in FILES:
    n = len(read(f).splitlines())
    w('| `coweek/%s` | %d | %s |' % (f, n, ROLE[f]))
w('')
w('---')
w('')

for f in FILES:
    w('## `coweek/%s`' % f)
    w('')
    w('```python')
    w(read(f))
    w('```')
    w('')

w('---')
w('')
w('# 참고 — `driver.py` 발췌')
w('')
w('베이스 제어기(규칙 기반, 1100+ 줄)라 전체는 뺐다. RL 이 닿는 곳만 싣는다.')
w('**RL 은 이 파일을 한 줄도 바꾸지 않는다** — 퍼블리셔만 가로챈다.')
w('')
w('## `last_tick` — env 가 관측을 만드는 유일한 소스')
w('')
w('```python')
w(between('driver.py', r'self\.last_tick = \{', r'^\s*\}'))
w('```')
w('')
w('## 매 틱 퍼블리시 (env 가 `pub_speed` / `pub_steer` 를 가로챈다)')
w('')
w('```python')
w(lines('driver.py', 1078, 1090))
w('```')
w('')
w('## v34c 텔레포트 감지 — 프레임이 확 바뀌면 래치를 스스로 지운다')
w('')
w('```python')
w(lines('driver.py', 264, 283))
w('```')
w('')
w('## 상한 상수 (정책이 이 상한들을 잔차로 완화하는 것이 1단계 목표다)')
w('')
w('```python')
src = read('driver.py').splitlines()
for ln in src:
    if re.match(r'^(RATE_HZ|MAX_STEER|V_MAX|V_MIN|A_LAT|WHEELBASE_M|CONE_SLOW'
                r'|LID_PREBRAKE\w*|CONE_ENGAGE_AREA|CAM_TILT|STALE_\w*)\s*=', ln):
        w(ln)
w('```')
w('')
w('## `run.sh` — 심판이 띄우는 진입점 (`ACTIVE_MODEL` 파일로 분기)')
w('')
w('```bash')
w(read('../run.sh'))
w('```')

txt = '\n'.join(P)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(txt)
print('wrote %s  (%d 줄, %.0f KB)' % (OUT, txt.count('\n') + 1, len(txt) / 1024))
