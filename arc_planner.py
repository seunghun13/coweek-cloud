#!/usr/bin/env python3
"""Phase C: constant-curvature arc lattice over the visible legal region.

Pure, stateless, ROS-free: plan(mask, cones) → (delta, safe_len, dbg).
The mask is the own-lane component on the LOOK_TOP crop (driver.own_lane
output); cones are (D, y, r_inflate) in vehicle-ground meters.

Geometry (flat ground, calibrated): D(row_full) = K/(row_full - H0),
col = CX - y * FX / D  (y positive = left).

An arc for steering delta has curvature k = tan(delta)/L. Points:
  x(s) = sin(k s)/k,  y(s) = (1 - cos(k s))/k     (straight: y=0)
Legality per point: its image row's road run [c0,c1] widened by the
0.12 m margin must contain the projected column. Rows where the run is
clipped by the image border are UNKNOWN → the arc is neither penalized
nor extended there (horizon capped).

Score: prefer long safe arcs, endpoints near the far road center, small
steering. Inside cutting is not a special case — arcs that clip the
corner stay legal (margin) and score best on length.
"""
import math

import cv2
import numpy as np

H0, KC, FX, CX = 112.8, 47.9, 240.0, 240.0
LOOK_TOP_ROW = 162          # int(360 * 0.45) — crop offset of the mask
WHEELBASE = 0.18
MARGIN_M = 0.04             # legal beyond the paint (rule: 0.12) minus slack.
                            # v33g: 0.10→0.06 — the cone6 west-side pass
                            # threaded the boundary at (0.90,4.8) 5/5 and
                            # sometimes swallowed the finish crossing
                            # (+70 s); corner cuts showed zero grazes at
                            # 0.10 so they have headroom.
S_STEP = 0.08
S_MAX = 0.96               # (v35 horizon extension to 1.44 REJECTED:
                           # 117.8 — every safe-based speed threshold was
                           # calibrated to the 0.96 cap and the long
                           # sweeps broke the tuned line character)
PASS_ROOM_W = 0.15         # weight on the road left beside a cone pass.
                           # Capped at 0.35 m, so the most it can add is
                           # 0.052 m — under one S_STEP (0.08). It breaks
                           # ties between arcs of equal length (at S_MAX
                           # many are exactly tied) and never overrides a
                           # genuinely longer one. v52 tried this as an
                           # offset on the planner's OUTPUT and drove into
                           # cone4 15/15; the preference belongs here.
PASS_ROOM_CAP = 0.35
# v62: 원호 순위를 사전식에서 가중합으로. 사전식에서는 safe 가 절대 우선이라
# "조금 짧지만 잘 중앙에 놓인 원호"가 원리적으로 이길 수 없었다 — PASS_ROOM 보너스
# 상한 0.052 도 S_STEP 0.08 을 못 넘게 일부러 묶어둔 값이다. 그 자유도를 연다.
# 라벨 프레임 오프라인 스윕 (probe_tmp/p2_sweep.py):
#     W_END  0.04 -> ok 235 DRIFT 12 WR 9   (v60 기준선 ok 221 DRIFT 24 WR 11)
#            0.08 -> ok 231 DRIFT 14 WR 11
#            0.15 -> ok 232 DRIFT 14 WR 11
#            0.30 -> finish_zone 2/56 -> 6/56  ← 다른 팀 빌드의 원값. 유해하다
# 0.30 이 왜 나쁜지는 CLAUDE.md 의 "호 길이 편향이 곧 조기 턴인의 원천"과 같다 —
# AMET 코너 R 0.19~0.33 이 최소회전반경 0.4906 보다 작아 조기 턴인이 없으면
# 물리적으로 못 돈다. end_err 를 너무 세게 주면 그 편향을 지운다.
W_END = 0.04               # 끝점이 도로 중앙에서 벗어난 정도의 벌점
W_STEER = 0.06             # 큰 조향 소폭 벌점 (사전식의 -abs(delta) 자리)
DELTAS = np.linspace(-0.349, 0.349, 21)


def road_runs(mask):
    """Per-row largest contiguous run → {row: (c0, c1, clipped)}."""
    h, w = mask.shape
    out = {}
    for r in range(h):
        cols = np.nonzero(mask[r])[0]
        if len(cols) < 12:
            continue
        best, s, prev = None, cols[0], cols[0]
        for c in cols[1:]:
            if c - prev > 5:
                if best is None or prev - s > best[1] - best[0]:
                    best = (s, prev)
                s = c
            prev = c
        if best is None or prev - s > best[1] - best[0]:
            best = (s, prev)
        out[r] = (best[0], best[1], best[0] <= 1 or best[1] >= w - 2)
    return out


def bridge_checker(mask, checker_d, crop_row=LOOK_TOP_ROW, anchor=10,
                   tol_px=8, gap_max=8):
    """Repair the road span the start/finish checker band erased.

    The band's white squares are not road-coloured, so they DELETE the
    mask rows they occupy — measured on the finish approach: crop rows
    0-5 (D 0.88-0.97 m) gone entirely, rowmin=5. road_runs then has no
    far rows, the straight arc dies at safe=0.88, and a CURVED arc —
    which covers less depth per unit arc length — still reaches S_MAX
    and wins the safe-first tiebreak: delta +0.140 (endpoint on the road
    EDGE, err 0.99) beat +0.000 (centred, err 0.25) exactly where the car
    must be centred to cross the LINE SEGMENT. That is the +5/teleport
    loop behind the 4/10 finals-format completion rate.

    Row-local inner bridging (the first design) cannot fix it: the dead
    rows hold <2 road pixels, so there is nothing to bridge between —
    measured winner unchanged. Folding the band's white pixels into the
    mask is v38 again: the run clips to (0,479).

    Geometry only: a straight road edge projects to a straight image
    line, so fit c0(row), c1(row) on the clean rows just BELOW the band
    and extrapolate through it. Validated against an independent clean
    frame of the same geometry (reconstructed spans matched to 1-2 px).

    REPAIR, never overwrite (v39b). Forcing detection at every depth on
    every probe frame moved the plan on 15/119 combinations — a single
    false positive in a corner painted a straight road and flipped the
    steering (serpentine_mid +0.244 -> -0.035, west_u_exit +0.174 ->
    0.000). So each row is walked NEAR->FAR and:
      * a healthy row (its own run already ~full width) is left alone;
      * a row whose surviving pixels fall OUTSIDE the extrapolation is a
        road that genuinely goes elsewhere: stop, paint nothing beyond;
      * only damaged/empty rows are filled, and a run of empty rows is
        capped (gap_max) so the band cannot invent a road into a bend.
    Purely additive. Returns True if it painted.
    """
    h, w = mask.shape
    r_far = int(round(H0 + KC / (checker_d + 0.20))) - crop_row
    r_near = int(round(H0 + KC / (checker_d - 0.15))) - crop_row
    r_far, r_near = max(0, r_far), min(h - 1, r_near)
    if r_near < r_far:
        return False
    rs, c0s, c1s = [], [], []
    for r in range(r_near + 1, min(h, r_near + 1 + anchor)):
        cols = np.nonzero(mask[r])[0]
        if len(cols) < 12 or cols[0] <= 1 or cols[-1] >= w - 2:
            continue            # empty or border-clipped: biased edge
        rs.append(r)
        c0s.append(cols[0])
        c1s.append(cols[-1])
    if len(rs) < 4:
        return False            # no trustworthy anchor: leave the mask
    rs = np.array(rs, dtype=float)
    f0 = np.polyfit(rs, np.array(c0s, dtype=float), 1)
    f1 = np.polyfit(rs, np.array(c1s, dtype=float), 1)
    if max(float(np.abs(np.polyval(f0, rs) - c0s).max()),
           float(np.abs(np.polyval(f1, rs) - c1s).max())) > 6.0:
        return False            # anchor edges not straight: do not invent
    painted, gap, seen_perf = False, 0, False
    for r in range(r_near, r_far - 1, -1):          # near -> far
        e0 = int(round(np.polyval(f0, r)))
        e1 = int(round(np.polyval(f1, r)))
        ew = e1 - e0
        if ew < 12 or e1 <= 0 or e0 >= w:
            break
        cols = np.nonzero(mask[r])[0]
        if len(cols) >= 12:
            if cols[0] < e0 - tol_px or cols[-1] > e1 + tol_px:
                break           # road leaves the extrapolated corridor
            ext = cols[-1] - cols[0]
            if ext >= 0.70 * ew and _longest_run(cols) >= 0.70 * ew:
                gap = 0
                continue        # healthy row: nothing to repair
            if ext < 0.70 * ew and len(cols) >= 0.60 * ext:
                break           # narrow AND solid: a road that genuinely
                                # narrows or ends (far rows of a bend).
                                # Perspective shrinks the road ~3 px per
                                # row here, so a SOLID row that much
                                # narrower than the extrapolation is
                                # geometry, not a hole.
            seen_perf = True    # perforated: either full extent with the
            gap = 0             # middle eaten (measured run 98 of 206) or
                                # a sparse remnant (12 px inside a 71 px
                                # extent where the road is 186 wide) —
                                # both are the band's squares.
        else:
            if not seen_perf:
                break           # empty rows with no band evidence below
            gap += 1            # them: the road simply ended here
            if gap > gap_max:
                break           # too much unseen road to reconstruct
        mask[r, max(0, e0):min(w, e1 + 1)] = 255
        painted = True
    return painted


def _longest_run(cols):
    """Longest contiguous span (px) in a sorted column index array."""
    best, s, prev = 0, cols[0], cols[0]
    for c in cols[1:]:
        if c - prev > 5:
            best = max(best, prev - s)
            s = c
        prev = c
    return max(best, prev - s)


def _bottom_comp(drv):
    """Bottom-anchored largest connected component of a drivable mask."""
    n, lab = cv2.connectedComponents(drv)
    if n <= 1:
        return drv
    c = np.bincount(lab[-8:, :].ravel(), minlength=n)
    c[0] = 0
    o = int(c.argmax())
    return (lab == o).astype(np.uint8) * 255 if c[o] else drv


def band_bridge(drv, gap_max=60, anchor=12, tol=10, min_overlap=0.45):
    """Reconnect the road across the start/finish checker band.

    The band is a horizontal stripe of white and near-black squares laid
    ACROSS the road. Neither colour is in the road range, so the stripe
    CUTS the drivable mask in two and the bottom-anchored component keeps
    only the near half — measured on the real approach (stationary probes
    on the centreline, 0.3-0.8 m before the line): the mask's topmost row
    walks 6 -> 14 -> 27 -> 44 -> 73 -> 134 as the band fills the view, so
    the planner's horizon collapses to ~0.5 m and the arc lattice starts
    reading a phantom corridor. Measured plan on dead-straight road:
    +0.140, +0.279, +0.314, +0.279 (all LEFT = east). That is the
    eastward swing seen in the traces from 0.5 m out, ending 0.50 m off
    centre at wp1 — past the line SEGMENT rather than across it, so the
    lap does not register: +5, teleport, and another full lap (+70 s).

    Rejected alternatives, all measured on 102 straight ground-truth
    stations (bad-station count, lower is better; v36 = 24):
      * morphological closing of the road mask — 5x5 30, h41 44, v15 64
        (the squares are 30-60 px: too big for a kernel that does not
        also merge neighbouring track sections)
      * folding LOW-SATURATION pixels (S<40 cleanly separates the band:
        road S=108, grass S=130, band S=2-9) into the mask wherever a
        vertical close reaches them — fixes the finish zone but eats the
        white EDGE lines and biases the whole west straight by +0.209.
        This is v38's failure a third time.
      * repairing AFTER the component pick (v39) — by then the far half
        has already been discarded, so there is nothing to reconnect.

    What works (16): bridge on the RAW mask, before labelling, and only
    when BOTH banks are present. The near bank's edges are fitted over
    its topmost healthy rows and extrapolated (a straight road edge is a
    straight image line); a far bank counts only if it carries road where
    that extrapolation predicts. A bend, where the road really does end,
    has no far bank there and is left untouched.
    """
    h, w = drv.shape
    near = _bottom_comp(drv)
    rows = np.nonzero(near.any(axis=1))[0]
    if len(rows) == 0 or int(rows.min()) <= 2:
        return drv                      # already reaches the horizon
    rs, c0s, c1s = [], [], []
    for r in range(int(rows.min()), min(h, int(rows.min()) + 3 * anchor)):
        cols = np.nonzero(near[r])[0]
        if len(cols) < 40 or cols[0] <= 1 or cols[-1] >= w - 2:
            continue
        ext = cols[-1] - cols[0]
        if ext < 60 or len(cols) < 0.85 * ext:
            continue                    # ragged band boundary: no anchor
        rs.append(r)
        c0s.append(cols[0])
        c1s.append(cols[-1])
        if len(rs) >= anchor:
            break
    if len(rs) < 6:
        return drv
    rt = rs[0]                          # topmost HEALTHY row
    rsa = np.array(rs, dtype=float)
    f0 = np.polyfit(rsa, np.array(c0s, dtype=float), 1)
    f1 = np.polyfit(rsa, np.array(c1s, dtype=float), 1)
    if max(float(np.abs(np.polyval(f0, rsa) - c0s).max()),
           float(np.abs(np.polyval(f1, rsa) - c1s).max())) > 6.0:
        return drv                      # edges not straight: do not invent
    far = None
    for r in range(rt - 1, max(-1, rt - 1 - gap_max), -1):
        e0 = int(round(np.polyval(f0, r)))
        e1 = int(round(np.polyval(f1, r)))
        if e1 - e0 < 12:
            break
        seg = drv[r, max(0, e0 - tol):min(w, e1 + tol + 1)]
        if int(np.count_nonzero(seg)) >= min_overlap * (e1 - e0):
            far = r
            break
    if far is None:
        return drv                      # no far bank: nothing to bridge
    out = drv.copy()
    for r in range(far, rt):
        e0 = int(round(np.polyval(f0, r)))
        e1 = int(round(np.polyval(f1, r)))
        if e1 - e0 < 12:
            continue
        out[r, max(0, e0):min(w, e1 + 1)] = 255
    return out


def plan(mask, cones=(), v_hint=0.6, prev_delta=None, crop_row=LOOK_TOP_ROW):
    """→ (delta_best, safe_len_m, dbg dict).

    (prev_delta accepted but UNUSED: v33h hysteresis stuck to wrong arcs
    near cones/corners — mean 147.8, finish-miss loops 3/5. Rejected.)"""
    runs = road_runs(mask)
    h, w = mask.shape
    results = []
    for delta in DELTAS:
        k = math.tan(delta) / WHEELBASE
        safe = 0.0
        blocked = False
        end_err = 0.0
        room = None                # metres of road beside the cone pass
        s = S_STEP
        while s <= S_MAX + 1e-9:
            if abs(k) > 1e-6:
                x = math.sin(k * s) / k
                y = (1 - math.cos(k * s)) / k
            else:
                x, y = s, 0.0
            if x < 0.20:
                s += S_STEP
                continue
            for cD, cy, cr in cones:
                if math.hypot(x - cD, y - cy) < cr:
                    blocked = True
                    break
            if blocked:
                break
            row = int(round(H0 + KC / x)) - crop_row
            if not (0 <= row < h):
                break                      # beyond view: horizon cap
            run = runs.get(row)
            if run is None:
                break                      # no road at this depth: dead end
            c0, c1, clipped = run
            col = CX - y * FX / x
            m_px = MARGIN_M * FX / x
            if col < c0 - m_px or col > c1 + m_px:
                if not clipped:
                    break                  # genuinely outside the road
                # clipped row: unknown on the clipped side — tolerate
            safe = s
            end_err = abs(col - (c0 + c1) / 2) / max(1.0, (c1 - c0) / 2)
            # At the depth where this arc draws level with a cone, how much
            # road is left between it and the nearer edge? Threading the
            # narrow gap scores low, so does hugging the far edge after
            # swinging too wide — both are the failures being targeted.
            for cD, cy, cr in cones:
                if abs(x - cD) < 0.10:
                    r = min(col - c0, c1 - col) * x / FX
                    room = r if room is None else min(room, r)
            s += S_STEP
        bonus = (0.0 if room is None
                 else PASS_ROOM_W * max(0.0, min(room, PASS_ROOM_CAP)))
        # safe rides along untouched: the driver gates on it and must see
        # the arc's true length, not the length plus a preference.
        # v62: 가중합. safe 를 튜플 2번째에 따로 실어 동점이면 여전히 긴 원호가
        # 이기게 하고, 드라이버의 a_safe 게이트(index 4)는 가중합이 섞이지 않은
        # 원값을 계속 본다 (v53 과 같은 처리).
        score = safe + bonus - W_END * end_err - W_STEER * abs(delta)
        results.append((score, safe, -abs(delta), delta, safe))
    results.sort(reverse=True)
    best = results[0]
    # The driver gates on a_safe >= 0.24, so hand back the arc's true
    # length (index 4) rather than the score that carries the bonus.
    return best[3], best[4], {'top': [(round(r[3], 3), round(r[4], 2))
                                      for r in results[:5]]}
