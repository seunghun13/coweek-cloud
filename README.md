# coweek-cloud — 채점 전용 트리

로컬(WSL)이 **정본**이다. 이 저장소는 클라우드가 채점만 할 수 있도록
필요한 파일만 담은 단방향 사본이다. **클라우드에서 코드를 고치지 않는다.**

## 배치
```
~/physicar_ws/coweek        <- 이 저장소를 clone
~/physicar_ws/run.sh        <- 여기 run.sh 를 복사
```

## A/B (심판은 고정, 주행 빌드만 스왑)
```bash
./ab_practice.sh 15 racer-v64                     # 예선맵
./ab_practice.sh 15 racer-v64 --random-cones --pin ''   # 본선
```
`ab_practice.sh` 는 `driver.py`, `arc_planner.py` 두 파일만 바꾸고
매 arm 마다 md5 를 대조한다.

## 짝지은 A/B
`referee.py --seed <N>` 을 주면 두 arm 이 **같은 콘 배치열**을 본다.
기본값 None 이면 시딩하지 않아 이 플래그가 없던 때와 동작이 같다.

## ⚠️ 클라우드 특성
- RTF 약 0.90 (4 vCPU, `gz sim` 단독 110%). 심판이 `<<DEGRADED` 를 찍는다.
- 그 저하는 **결승선 놓침만** 3% → 23% 로 키운다. 랩타임·페널티는 거의 불변.
- 따라서 **속도 상한·결승 접근을 건드리는 변경은 로컬에서 판정**한다.
- 클라우드 절대 수치를 로컬과 합산하지 않는다. 클라우드 안 A/B 만 유효.
