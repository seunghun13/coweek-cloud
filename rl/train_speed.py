#!/usr/bin/env python3
"""1단계 잔차 SAC 학습 — 속도 잔차만.

## 실행 (승인 후)
    . /opt/physicar/src/physicar-ros/deploy/sim/bashrc-append
    cd ~/physicar_ws/coweek
    python3 rl/train_speed.py --steps 60000 --out rl/runs/s1

## 예산 (실측 기반)
제어 20 Hz, RTF 1.0 이 하드 상한이므로 **시간당 72,000 스텝**이다.
시뮬 가속은 쓰지 않는다 — 제어 루프가 벽시계라 가속하면 시뮬 시간 기준
제어 주기가 줄어 다른 플랜트가 된다 (driver.py 는 use_sim_time 미설정).
    1시간 =  72k  |  2시간 = 144k  |  3시간 = 216k

## 왜 SAC 인가
연속 1차원 행동, 수십만 스텝 예산, off-policy 라 샘플 효율이 높다.
PPO 는 on-policy 라 이 예산에서 불리하다.

## 동역학 랜덤화 (기본 ON — 사용자 승인 2026-08-21)
이 시뮬은 구동계가 속도 소스라 **감속이 공짜다**(≥4.5 m/s² 실측). 그 위에서
속도 잔차를 학습하면 정책은 "코너 직전까지 최고속 → 순간 감속" 을 배우고
실차에서 그대로 죽는다. 그래서 에피소드마다 가감속 한계·조향 서보 각속도·
구동/관측 지연을 다시 뽑는다. 근거·범위는 `rl/dyn.py`, 검증은 `rl/test_dyn.py`.
끄려면 `--no-dyn` (참 플랜트 대조군용).

## 안전 장치
- action 0 == 잔차 0 == racer-v64. 초기 정책이 베이스 근처에서 시작한다.
- 최종 명령은 env 에서 [0, 1.40] 으로 하드 클립된다.
- 체크포인트마다 저장하고, 채택은 별도 `eval_gates.py` 로만 한다
  (학습 곡선은 채택 근거가 아니다 — 오프라인 지표가 배치를 예측하지 못한
   전례가 이 프로젝트에 여러 건 있다).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, '/home/physicar/physicar_ws/coweek')
sys.path.insert(0, '/home/physicar/physicar_ws/coweek/rl')

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from coweek_env import CoweekResidualEnv
from dyn import DYN_RANGES


class Progress(BaseCallback):
    """스텝률과 최근 성과를 주기적으로 찍는다. 학습 곡선은 진단용이다."""

    def __init__(self, every=2000):
        super().__init__()
        self.every = every
        self.t0 = time.time()
        self.off = 0
        self.hit = 0
        self.ds = 0.0
        self.lag = 0.0          # |명령 - 실제로 나간 값| 누적
        self.n = 0
        self.laps = 0           # 완주한 바퀴 수

    def _on_step(self):
        info = self.locals.get('infos', [{}])[0]
        self.off += int(info.get('off', False))
        self.hit += int(info.get('hit', False))
        self.ds += float(info.get('ds', 0.0))
        self.laps += int(info.get('lap_done', False))
        if 'cmd_out' in info:
            self.lag += abs(float(info['cmd']) - float(info['cmd_out']))
            self.n += 1
        if self.num_timesteps % self.every == 0:
            el = time.time() - self.t0
            # `plant` 은 랜덤화가 실제로 물고 있는지 보는 위생 지표다.
            # dyn 을 켰는데 0.000 이면 배선이 끊긴 것이다.
            print('[%7d] %5.1f min | %.1f steps/s | 랩 %d | ds %.1f m'
                  ' | off %d | hit %d | plant %.3f m/s'
                  % (self.num_timesteps, el / 60.0,
                     self.num_timesteps / max(el, 1e-9), self.laps,
                     self.ds, self.off, self.hit,
                     self.lag / max(self.n, 1)), flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--out', default='rl/runs/s1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episode-s', type=float, default=8.0)
    ap.add_argument('--warmup-s', type=float, default=0.25,
                    help='텔레포트 직후 last_tick 을 채울 최소 시간. '
                         '--start-anywhere 로 임의 지점에서 시작할 때만 '
                         '1.2 정도가 필요하다 (가짜 정지출발 편향 제거용)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--tau', type=float, default=0.01)
    ap.add_argument('--learning-starts', type=int, default=2000)
    # UTD(update-to-data). 데이터 수집이 실시간 시뮬에 하드 캡(18.6 steps/s)돼
    # 있는데 GPU 는 3 % 다 — 같은 54분 동안 학습 횟수만 늘릴 수 있는 유일한
    # 레버다. 다만 높은 UTD 는 SAC 의 Q 과대추정을 키우므로 올릴 때는
    # 반드시 A/B 로 재라 (GPT 검수 Q3 으로 넘긴 항목).
    ap.add_argument('--gradient-steps', type=int, default=1)
    ap.add_argument('--train-freq', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-lap', action='store_true',
                    help='랩 단위 에피소드를 끄고 --episode-s 초 슬라이스로 돈다. '
                         '기본은 랩 — 평가가 한 바퀴라 학습도 한 바퀴여야 '
                         '분포가 맞고, 결승선 접근은 랩 스케일 사건이다')
    ap.add_argument('--max-episode-s', type=float, default=150.0,
                    help='랩 모드 안전망: 차가 멈춰도 이 시간이면 끊는다')
    ap.add_argument('--start-anywhere', action='store_true',
                    help='랩을 트랙 임의 지점에서 시작한다. 기본은 출발선 — '
                         '임의 지점이면 에피소드 경계가 눈에 보이는 출발선과 '
                         '어긋나 도중에 리셋·콘재배치가 일어나는 것처럼 보인다')
    ap.add_argument('--cone-reshuffle', type=int, default=1,
                    help='N 에피소드마다 cone6 를 뺀 콘 5개를 다시 뿌린다 '
                         '(0 = 예선맵 고정). 과적합 방지 — 생성기는 심판의 '
                         'random_cone_layout 을 그대로 쓴다')
    ap.add_argument('--cone-pin', default='cone6',
                    help='재배치에서 제외할 콘 (쉼표 구분, 빈 문자열이면 전부 랜덤)')
    ap.add_argument('--resume', default='',
                    help='기존 zip 에서 이어서 학습한다. 같은 이름의 '
                         '_buffer.pkl 이 있으면 리플레이 버퍼도 복원한다. '
                         '예: --resume rl/runs/s1/final.zip --steps 60000')
    ap.add_argument('--no-dyn', action='store_true',
                    help='동역학 랜덤화를 끈다 (참 플랜트 = 시뮬 그대로). '
                         '기본은 켜짐 — 감속이 공짜인 플랜트만 보면 정책이 '
                         '그걸 착취하고 실차에서 죽는다. 근거는 rl/dyn.py')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pin = tuple(x for x in a.cone_pin.split(',') if x.strip())
    env = Monitor(CoweekResidualEnv(seed=a.seed, episode_s=a.episode_s,
                                    warmup_s=a.warmup_s, dyn=not a.no_dyn,
                                    cone_reshuffle=a.cone_reshuffle,
                                    cone_pin=pin,
                                    lap_episode=not a.no_lap,
                                    max_episode_s=a.max_episode_s,
                                    start_at_line=not a.start_anywhere),
                  filename=os.path.join(a.out, 'monitor.csv'))
    print('에피소드: %s · 시작 %s'
          % ('한 바퀴 (안전망 %.0f 초)' % a.max_episode_s if not a.no_lap
             else '%.1f 초 슬라이스' % a.episode_s,
             '트랙 임의 지점' if a.start_anywhere else '출발선'), flush=True)
    print('동역학 랜덤화: %s' % ('OFF — 참 플랜트' if a.no_dyn else
                                'ON  ' + str(DYN_RANGES)), flush=True)
    print('콘 재배치: %s (고정: %s)'
          % ('OFF — 예선맵 고정' if not a.cone_reshuffle
             else '%d 에피소드마다' % a.cone_reshuffle,
             ','.join(pin) or '없음(전부 랜덤)'), flush=True)

    # tensorboard 는 있으면 쓰고 없으면 그냥 안 쓴다. 없다고 학습이 죽으면 안 된다
    # — 진단은 monitor.csv 와 Progress 로그로 충분하고, 이 프로젝트에서
    # 학습 곡선은 애초에 채택 근거가 아니다 (채택은 eval_policy.py 로만).
    try:
        import tensorboard            # noqa: F401
        tb = os.path.join(a.out, 'tb')
    except ImportError:
        tb = None
        print('tensorboard 없음 — 로깅 생략 (monitor.csv 는 그대로 쌓인다)',
              flush=True)

    if a.resume:
        # 한 시간짜리 학습을 처음부터 다시 돌리지 않으려고 있는 경로다.
        # 리플레이 버퍼가 같이 저장돼 있으면 반드시 같이 싣는다 — 버퍼 없이
        # 이어가면 정책만 남고 off-policy 데이터가 사라져 사실상 재탐색이다.
        model = SAC.load(a.resume, env=env, device=a.device)
        print('이어서 학습:', a.resume, flush=True)
        buf = os.path.splitext(a.resume)[0] + '_buffer.pkl'
        if os.path.isfile(buf):
            model.load_replay_buffer(buf)
            print('리플레이 버퍼 복원: %s (%d 전이)'
                  % (buf, model.replay_buffer.size()), flush=True)
        else:
            print('!! 리플레이 버퍼가 없다 (%s) — 정책만 이어간다' % buf,
                  flush=True)
    else:
        model = SAC(
            'MlpPolicy', env,
            learning_rate=a.lr,
            buffer_size=300_000,
            batch_size=a.batch,
            gamma=a.gamma,          # 0.995 @20Hz -> 유효 지평 약 10 초
            tau=a.tau,
            train_freq=a.train_freq,
            gradient_steps=a.gradient_steps,
            learning_starts=a.learning_starts,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
            seed=a.seed,
            device=a.device,
            tensorboard_log=tb,
        )

    ck = CheckpointCallback(save_freq=5000, save_path=a.out,
                            name_prefix='sac')
    print('UTD: train_freq %d / gradient_steps %d -> 환경 1스텝당 학습 %.1f회'
          % (a.train_freq, a.gradient_steps,
             a.gradient_steps / max(a.train_freq, 1)), flush=True)
    print('학습 시작 — 목표 %d 스텝 (20 Hz 기준 약 %.1f 시간)'
          % (a.steps, a.steps / 72000.0), flush=True)
    try:
        model.learn(total_timesteps=a.steps, callback=[ck, Progress()],
                    log_interval=10, reset_num_timesteps=not bool(a.resume))
    finally:
        final = os.path.join(a.out, 'final')
        model.save(final)
        # 버퍼도 같이 남긴다. --resume 이 이걸 찾는다 (약 160 MB).
        try:
            model.save_replay_buffer(final + '_buffer')
        except Exception as e:
            print('버퍼 저장 실패(무시):', e, flush=True)
        env.close()
        print('저장 완료:', final + '.zip')


if __name__ == '__main__':
    main()
