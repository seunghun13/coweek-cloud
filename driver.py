#!/usr/bin/env python3
"""coweek_driver v7 — v4 reconstruction + post-pass hold (team 코윅윅 #28).

v4 (best: 109.6 s) perception restored: own-lane connected component with a
far lookahead band. Single targeted addition: after a cone leaves the view
during a pass, HOLD the passing line for a few ticks before recentering —
v3-v6 all clipped cone1 with the rear wheel by snapping back the moment the
cone dropped below the camera.

Invariants: stateless per frame (only frame-scale steering memory),
both commands published every tick at 20 Hz, /speed 0 until green.
"""
import math
import time

import cv2
import numpy as np
import rclpy

import arc_planner
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import Float64

# ── control ──
RATE_HZ = 20.0
MAX_STEER = 0.349
STEER_GAIN = 2.2
V_MAX = 0.9
V_MIN = 0.35                # v63 이후로는 감속 곡선에 안 쓰인다 (다른 곳에서 참조)
CONE_SLOW = 0.45
# ── v63: 감속을 물리로 ──
# 기존 곡선 speed = V_MAX-(V_MAX-V_MIN)*(2k-k^2), k=|steer|/MAX_STEER 는 물리가
# 아니라 임의의 오목 다항식이었다. k=0 에서 도함수가 2 라 조향 5.7도만 줘도
# 0.900 -> 0.630 으로 30 % 를 깎는다. 실제로 필요한 감속은 횡가속 한계다.
#     kappa = |tan(steer)|/WHEELBASE,  v = min(V_MAX, sqrt(A_LAT/kappa))
# 두 정책 비교 (계산):
#     steer  5.7deg  R 1.794  현재 0.630 -> 0.900  (143 %)
#           11.5deg  R 0.888  현재 0.450 -> 0.900  (200 %)
#           20.0deg  R 0.495  현재 0.350 -> 0.703  (201 %)
# A_LAT 근거: turn_radius.py 실측으로 풀락 0.80 m/s 에서 R 이 0.4906->0.4942 로
# 사실상 불변이었다 = 횡가속 1.295 m/s^2 까지 미끄러짐이 없다. 1.0 은 그 77 % 다.
# 콘 상한(CONE_SLOW, carea 0.6, 라이다 프리브레이크)과 체커 0.45 는 그대로 두므로
# 콘 근처에서는 이 값이 쓰이지 않는다 — 그게 이 변경의 설계다.
WHEELBASE_M = 0.18
A_LAT = 1.0                # m/s^2
HOLD_DECAY = 0.90
CAM_TILT = -0.2618
POST_PASS_TICKS = 8        # ~0.4 s: keep the passing line after the cone
                           # drops out of view (rear-wheel clearance)

# ── camera freshness (v60) ──
# self.frame carried no timestamp: it was assigned once and re-read forever,
# so a dead camera meant driving at full speed on a frozen picture. Harmless
# in the sim, but on the real kit on 8/25 it is the failure that ends a run
# with no symptom in the log.
#
# Measured on this box (probe_tmp/main_framegap.py, 40 s, 607 frames):
#   camera 15.15 Hz; frame gap min 63.2 / median 65.1 / p99 71.3 / MAX 72.1 ms
#   — and that was under loadavg 3.99, not idle.
# 0.35 s is 4.9x the worst observed gap, so normal driving never trips it
# (0/606 gaps exceeded it). Steering is deliberately left alone: that matches
# the platform's own /speed watchdog contract (on expiry the speed goes to 0
# and the steering angle is held).
STALE_SLOW = 0.35          # s without a new frame → creep
STALE_SLOW_V = 0.35        # m/s
STALE_STOP = 1.00          # s without a new frame → stop

# ── road corridor (v4 logic; road HSV ≈ (107,110,78)) ──
ROAD_LO = (95, 50, 40)
ROAD_HI = (125, 170, 130)
ROAD_MIN_FRAC = 0.03       # v64: 이 아래면 고정 HSV 창이 무너진 것으로 보고
                           # 씨앗 임계로 폴백. 정상 470프레임 최소 0.1521 의 20 %.
                           # 0.08 로 처음 재봤을 때 예선 15런 중 1런이 cone3 히트
                           # 연쇄로 페널티 15 가 났다(v63 은 15/15 가 정확히 5.0).
                           # 라벨 프레임에서는 0/470 오발동이었지만 주행 중에는
                           # 차가 잔디에 걸친 순간이 있고, 그때 씨앗이 잔디를 잡으면
                           # 오히려 해가 된다. 치명적 붕괴(면적 0.000~0.008)만
                           # 잡도록 조인다 — H-25/Sx0.3/H-15 는 여전히 전부 걸린다
LOOK_TOP = 0.45
LOOK_BOT = 0.75
ROAD_MIN_PX = 300

YELLOW_LO = (15, 70, 70)
YELLOW_HI = (40, 255, 255)
LINE_BLEND = 0.35
LINE_MIN_PX = 40

# ── cones (AMET 2026: GREEN #05FA05 — measured in-camera H=60 S=233-248
# V=179-251; grass stays below S170 so the S floor separates them. The lit
# green lamp of light1 also lands in this range — the on-path |cx| gate and
# commit guard keep it from hijacking dodge as we pass it once at start) ──
CONE_LO = (50, 170, 120)
CONE_HI = (70, 255, 255)
CONE_ROI_TOP = 0.3
CONE_NEAR_AREA = 150       # tracked (telemetry/speed) from this size
CONE_ENGAGE_AREA = 400     # dodge mode only for cones this big…
SHARP_CONE_VETO = 5000     # green px² above which a cone, not a
                           # dead-end, explains an empty far band.
                           # Measured: carea >= 5000 never happens
                           # with the nearest cone beyond 1.2 m
                           # (0/3574 ticks; max 2626 beyond 1.0 m)
CONE_ONPATH_CX = 0.5       # …AND roughly in our path (v8 telemetry: dodge
                           # owned 3x more ticks than road — roadside cones
                           # were steering the car)
CONE_PANIC_AREA = 2500
CONE_SHIFT = 0.35          # bounded aim-point shift for a pass (v21) —
CONE_SHIFT_RAMP = 1200     # the 0.9 repulse bias tuned for the 0.90 m
                           # practice road overshot AMET's half+0.12=0.47 m
                           # allowance by 1-4 cm (all 18 measured offtracks
                           # were cone-adjacent grazes, paired with hits)

LINE_FEED_K = 0.35         # line-slope feedforward (hairpin turn direction)
LINE_FEED_MAX = 0.5

# ── gated centering (v29) — measured: zero lateral stiffness on straights,
# the car rides whatever offset the last corner/pass left (constant -0.16 m
# for 4 m on the south straight = single-lane riding, user-observed).
# v17's UNGATED version was a disaster (112 s: pulled onto center-line
# cones, fought corner cuts) — hence the gates: no engaged cone, no pass
# hold, straight confirmed, no sharp/sharplock. ──
LAT_K = 0.25
LAT_MAX = 0.20
LAT_FEED_GATE = 0.12       # |line-slope feed| below this = straight

# ── BEV multi-station estimator (v31, Phase A) — flat-ground pinhole
# calibrated offline against sim GT: D(row)=BEV_K/(row-BEV_H0), validated
# d error 0.5-3.5 cm at ±0.15 m offsets, phi ±0.6° on straights.
# Stations whose road run clips the image edge are rejected (clipping
# biases the visible-run center → phantom slope/curvature). ──
BEV_H0 = 112.8
BEV_KC = 47.9
BEV_FX = 240.0
BEV_CX = 240.0
BEV_STATIONS = (0.25, 0.35, 0.42, 0.50, 0.60, 0.70)
BEV_KD = 0.7               # aim units per meter of lateral error

# (v32 apex-bias removed after 5-lap 96.3: it fires only once feed rises
# MID-corner — the outer-lane entry is locked in before that — and the
# amplified steering dragged coupling speed down 12 s/lap. Fourth failed
# half-measure on the corner-line class; the structural fix is the arc
# lattice (Phase C), where inside cutting falls out of arc feasibility.)


# ── start/finish band (v45) ──────────────────────────────────────────────
# The band is white and near-black squares painted ACROSS the road. Neither
# colour is in the road range, so the drivable mask is CUT in two and the
# bottom-anchored component throws the far half away: measured on
# stationary centreline probes, the mask's top row walks 6 -> 14 -> 27 ->
# 44 -> 73 -> 134 as the band fills the view, the planner's horizon
# collapses to ~0.5 m, and on dead-straight road the lattice answers with
# +0.140/+0.279/+0.314 (all LEFT). That is the swerve that crosses BESIDE
# the line segment: the official script (evaluation.json) requires the
# front point to cross inner[0]->outer[0], i.e. |offset| <= ~0.35 m, while
# the off-track allowance is 0.472 m — a 12 cm band where the car is legal
# but the lap never registers. Measured misses sat at 0.39-0.52 m.
#
# Folding low-saturation pixels in wholesale is v38's failure (the white
# EDGE lines get eaten and the whole west straight biases by +0.209). The
# band is instead identified by its CHECKER STRUCTURE, which no kerb has:
#   real band  white runs [27,30,56,53] [33,35,34,32] [45,49,50,48], dark 57-186
#   corner kerb white runs [331] [321] [426,5] [341],               dark 0-1
# Gated that way it fires on 15 of 414 labelled frames and ZERO outside the
# finish zone.
BAND_SAT = 40              # road S=108, grass S=130, band squares S=2..9
BAND_ROW_FRAC = 0.80       # centre-half low-sat fraction marking a band row
BAND_W_MIN = 0.70          # plausible band width in metres (road 0.70 …
BAND_W_MAX = 1.10          # … plus the white edge lines, measured 0.81-0.88)
BAND_MIN_RUNS = 3
BAND_SQ_M = 0.1125         # square size (measured 29-31 px at D=0.90)
BAND_MIN_DARK = 20         # dark squares must exist: a kerb has none

# ── lidar ──
LID_FRONT_HALF = math.radians(35)
LID_AVOID = 0.40
LID_CRAWL = 0.22
LID_RANGE_MAX = 16.0
# AMET 2026 pre-brake: the perimeter wall (0.4 m tall, lidar-visible) sits
# ~1.1-1.4 m past every hairpin entry, so front range directly encodes
# "corner soon" — the recurring baseline offtracks were straight-exit
# overspeed at (11.4,1.3)/(8.1,3.2). Cones (0.38 m tall) trip this too:
# free early braking before the camera dodge engages.
LID_PREBRAKE1 = 1.50       # front min below this → cap V
LID_PREBRAKE1_V = 0.55
LID_PREBRAKE2 = 0.90
LID_PREBRAKE2_V = 0.40

# ── green light (AMET 2026: cones are the SAME green as the lamp, so a
# green blob alone is no longer proof of GO — a random-final cone in the
# start view would false-start us +10 s. Anchor on the lit RED lamp
# instead: track its pixel position while waiting, and accept only a green
# blob that appears at that spot (green lamp sits just below the red one).
# Fallback (red never seen): size-bounded green, as before) ──
GREEN_LO = (45, 150, 150)
GREEN_HI = (75, 255, 255)
GREEN_ROI_BOTTOM = 0.75
GREEN_MIN_AREA = 40
GREEN_FRAMES = 3
RED_LO1 = (0, 150, 150)
RED_HI1 = (4, 255, 255)
RED_LO2 = (172, 150, 150)
RED_HI2 = (180, 255, 255)
GO_NEAR_R = 110            # px radius around the red lamp for a valid GO —
                           # orientation-agnostic (v29b): the venue light is
                           # a phone whose mounting we don't control; sim/
                           # webapp are portrait (green ~40 px below red)
                           # but a landscape mount would put green BESIDE
                           # red, outside the old vertical-only box.
GO_NEAR_MIN = 5


def sanitize(r, rmax=LID_RANGE_MAX):
    r = np.asarray(r, dtype=np.float32)
    r[~np.isfinite(r) | (r <= 0.01)] = rmax
    return r


class CoweekDriver(Node):
    def __init__(self):
        super().__init__('coweek_driver')
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)
        self.pub_tilt = self.create_publisher(Float64, '/camera/tilt', 10)
        self.create_subscription(CompressedImage, '/camera/image_raw/compressed',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan_filtered',
                                 self.on_scan, qos_profile_sensor_data)
        self.frame = None
        self.frame_t = None         # v60: monotonic stamp of self.frame
        self.scan = None
        self.green_streak = 0
        self.green_latched = False
        self.lamp_xy = None         # lit red lamp position while waiting
        self.last_steer = 0.0
        self.pass_hold = 0          # ticks remaining of post-pass line hold
        self.pass_side = 0.0        # latched pass direction for the active
                                    # pass (+1 right / -1 left / 0 idle)
        self.band_px = 0
        self.cg_last = []
        self.band_tick = -1     # tick band_px was measured on.
                                # road_aim() is the only writer
                                # and it does not run every
                                # tick, so the value goes stale
                                # (v47: cone4's empty-band 0
                                # survived 17 m and armed a
                                # full lock on straight road)
        self.cone_list = []         # engaged-cone bboxes (per frame)
        self.arc_safe = 1.0         # last arc-planner safe horizon (m)
        self.last_arc = None        # last chosen arc delta (hysteresis)
        self.prev_small = None      # downsampled last frame (teleport jump)
        self.sharp_lock = False     # hairpin: hold full lock while the far
        self.sharp_steer = 0.0      # band is dead (calibrated: <8k engage,
        self.ticks = 0              # >15k release; lap p5 = 9.2k)
        try:
            self.telem = open('/tmp/coweek_telem.csv', 'a', buffering=1)
            self.telem.write('tick,green,aim,mode,steer,speed,cone_area,band,ncone,cy,cd,pside,phold,safe\n')
        except OSError:
            self.telem = None
        self.create_timer(1.0 / RATE_HZ, self.tick)
        self.get_logger().info('coweek_driver v7 up — waiting for green')

    def on_image(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            # v34c teleport reset: a penalty teleport swaps the whole
            # scene in one frame — the strongest stateless signature we
            # have. Stale frame-scale memory (hold with the pre-teleport
            # dodge steer, sharp lock, pass latch) caused a measured
            # ~0.5 m drift on the fresh straight → finish-miss cascade.
            small = cv2.resize(img, (60, 45))
            if self.prev_small is not None:
                jump = float(np.mean(cv2.absdiff(small, self.prev_small)))
                if jump > 45.0:
                    self.pass_hold = 0
                    self.pass_side = 0.0
                    self.sharp_lock = False
                    self.last_steer = 0.0
                    self.last_arc = None
            self.prev_small = small
            self.frame = img
            self.frame_t = time.monotonic()

    def on_scan(self, msg):
        self.scan = msg

    def detect_go(self, hsv):
        """Red-lamp-anchored GO detection → matched green area (0 = no GO)."""
        roi = hsv[: int(hsv.shape[0] * GREEN_ROI_BOTTOM)]
        red = cv2.bitwise_or(cv2.inRange(roi, RED_LO1, RED_HI1),
                             cv2.inRange(roi, RED_LO2, RED_HI2))
        rcnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rbest = max(rcnts, key=cv2.contourArea, default=None)
        if rbest is not None and cv2.contourArea(rbest) >= 20:
            m = cv2.moments(rbest)
            self.lamp_xy = (m['m10'] / max(m['m00'], 1e-6),
                            m['m01'] / max(m['m00'], 1e-6))
        green = cv2.inRange(roi, GREEN_LO, GREEN_HI)
        gcnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = 0
        for c in gcnts:
            area = cv2.contourArea(c)
            if area < GREEN_MIN_AREA:
                continue
            m = cv2.moments(c)
            gx = m['m10'] / max(m['m00'], 1e-6)
            gy = m['m01'] / max(m['m00'], 1e-6)
            if self.lamp_xy is not None:
                lx, ly = self.lamp_xy
                if GO_NEAR_MIN < math.hypot(gx - lx, gy - ly) < GO_NEAR_R:
                    best = max(best, int(area))
            elif area <= 1500:      # no red ever seen: size-bounded fallback
                best = max(best, int(area))
        return best

    def road_aim(self, hsv, line_pull=True):
        """v4 own-lane component aim → (aim, trusted)."""
        h, w = hsv.shape[:2]
        low = hsv[int(h * LOOK_TOP):]
        road = cv2.inRange(low, ROAD_LO, ROAD_HI)
        yline = cv2.inRange(low, YELLOW_LO, YELLOW_HI)
        drivable = cv2.bitwise_or(road, yline)
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8))
        n_lbl, labels = cv2.connectedComponents(drivable)
        if n_lbl > 1:
            bottom = labels[-8:, :]
            counts = np.bincount(bottom.ravel(), minlength=n_lbl)
            counts[0] = 0
            own = int(counts.argmax())
            if counts[own] > 0:
                drivable = (labels == own).astype(np.uint8) * 255
        band_bot = int((LOOK_BOT - LOOK_TOP) / (1 - LOOK_TOP) * drivable.shape[0])
        sharp = False
        ys, xs = np.nonzero(drivable[:band_bot])
        self.band_px = len(xs)      # telemetry: hairpin-trigger calibration
        self.band_tick = self.ticks
        if len(xs) < ROAD_MIN_PX:
            # far band empty = the road visually dead-ends → hairpin-grade
            # corner right ahead (the (4.95,2.0) -118° signature)
            sharp = True
            ys, xs = np.nonzero(drivable)
            if len(xs) < ROAD_MIN_PX:
                return 0.0, False, sharp
        aim = (float(xs.mean()) - w / 2) / (w / 2)
        # whole-blob aim (near rows included): at a dead end the near road
        # curling under the car keeps the true turn direction (measured:
        # whole +0.45 right=correct vs far-remnant -0.8 left=wrong)
        wxs = np.nonzero(drivable)[1]
        self.whole_aim = ((float(wxs.mean()) - w / 2) / (w / 2)
                          if len(wxs) else 0.0)
        if sharp and abs(aim) < 0.25:
            aim = math.copysign(0.45, self.whole_aim) if abs(self.whole_aim) > 0.03 \
                else self.freer_side() * -0.45
        if line_pull and not sharp:
            lmask = cv2.bitwise_and(yline, drivable)
            lys, lxs = np.nonzero(lmask)
            if len(lxs) >= LINE_MIN_PX:
                line_off = (float(lxs.mean()) - w / 2) / (w / 2)
                aim = (1 - LINE_BLEND) * aim + LINE_BLEND * line_off
        return aim, True, sharp

    def cone_pick(self, hsv):
        """Strongest ON-PATH cone → (cx, area, max_area, engaged).

        (v18 note kept: the practice world's yellow line leaked into the
        old orange cone range; AMET's green cones don't collide with
        yellow, but the subtraction stays as cheap insurance.)"""
        h, w = hsv.shape[:2]
        roi = hsv[int(h * CONE_ROI_TOP):]
        mask = cv2.inRange(roi, CONE_LO, CONE_HI)
        mask &= cv2.bitwise_not(cv2.inRange(roi, YELLOW_LO, YELLOW_HI))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cone_list = []
        best, max_area = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            max_area = max(max_area, area)
            if area < CONE_NEAR_AREA:
                continue
            m = cv2.moments(c)
            cx = (m['m10'] / max(m['m00'], 1e-6) - w / 2) / (w / 2)
            on_path = abs(cx) <= CONE_ONPATH_CX
            if not on_path and area < SHARP_CONE_VETO:
                continue                       # roadside cone: not our problem
            # v49: a contour this large is a cone within about a metre,
            # whatever its bearing. Over 470 labelled frames the |cx| gate
            # rejected 142 contours; the 18 with area >= 5000 were ALL real
            # cones at D = 0.194-0.663 m, |y| = 0.132-0.381 m, and none was
            # printed decoration. The gate still stands for everything
            # smaller — v11 measured an off-road green object at cx = -0.91
            # reversing the plan, and that object was not this big.
            # planner obstacles register from NEAR (150 px ≈ 2.5 m) —
            # at ENGAGE (400) the cut line is already committed past the
            # clearance point (cone4 hit 5/5 with a 0.19 m miss distance)
            self.cone_list.append(cv2.boundingRect(c))
            if not on_path:
                continue          # planner obstacle only: the engaged-cone
                                  # pick below, which drives dodge and
                                  # pass_side, keeps its original gate so
                                  # this change tests exactly one thing
            if area < CONE_ENGAGE_AREA:
                continue
            if best is None or area > best[1]:
                best = (cx, area, cv2.boundingRect(c))
        if best is None:
            return 0.0, 0.0, max_area, False, None
        return best[0], best[1], max_area, True, best[2]

    def cones_ground(self, h):
        """Engaged-cone bboxes → [(D, y, r_inflate)] in vehicle meters."""
        out = []
        self.cg_last = out          # telemetry reads this; see the header
        for bx, by, bw, bh in self.cone_list:
            row = int(h * CONE_ROI_TOP) + by + bh
            if row <= BEV_H0 + 5:
                continue
            D = BEV_KC / (row - BEV_H0)
            if not (0.2 < D < 1.6):
                continue
            y = -((bx + bw / 2) - BEV_CX) * D / BEV_FX
            out.append((D, y, 0.25))
        return out

    def pass_gaps(self, hsv, bbox):
        """OWN-LANE road width left/right of the cone → (left_px, right_px).

        v27: measure on the bottom-anchored connected component, not the
        raw color mask — on the serpentine a neighboring road segment
        shows at the same image rows and a raw count adopts it as "gap"
        (v26: cone3 5/5, cone4 4/5 wrong-side passes)."""
        h, w = hsv.shape[:2]
        low = hsv[int(h * LOOK_TOP):]
        road = cv2.inRange(low, ROAD_LO, ROAD_HI)
        yline = cv2.inRange(low, YELLOW_LO, YELLOW_HI)
        drv = cv2.bitwise_or(road, yline)
        n_lbl, labels = cv2.connectedComponents(drv)
        if n_lbl > 1:
            cnt = np.bincount(labels[-8:, :].ravel(), minlength=n_lbl)
            cnt[0] = 0
            own = int(cnt.argmax())
            if cnt[own] > 0:
                drv = (labels == own).astype(np.uint8) * 255
        bx, by, bw, bh = bbox
        y0 = int(h * CONE_ROI_TOP) + by + bh // 2 - int(h * LOOK_TOP)
        y1 = int(h * CONE_ROI_TOP) + by + bh - int(h * LOOK_TOP)
        band = drv[max(0, y0):max(0, min(drv.shape[0], y1 + 4))]
        if band.size == 0:
            return 0, 0
        left = int(np.count_nonzero(band[:, :max(0, bx)]))
        right = int(np.count_nonzero(band[:, min(w, bx + bw):]))
        return left, right

    def detect_checker(self, hsv):
        """Checker start/finish band ahead → its depth in m, else None.
        (v39: returns the DEPTH, not a bool — the arc planner needs the
        band's rows to repair them; see arc_planner.bridge_checker.)
        The line
        print is the one visual cue that exists at EVERY layout's finish
        approach — an offtrack teleport near it swallows the crossing
        (+70 s cascade, the worst stochastic failure). Purely reactive:
        alternating unsaturated black/white squares on ≥2 sampled rows."""
        h, w = hsv.shape[:2]
        c0, c1 = int(w * 0.25), int(w * 0.75)   # center half: the white
        # EDGE lines live outside it, the orange dashes are saturated —
        # only the checker leaves ≥3 separated white runs mid-road.
        # The band is thin (~0.2 m = one sampled depth at a time) and its
        # dark squares are ROAD-colored, not black (V≈79): so the
        # signature is white/road alternation on ANY single row.
        for Dm in (0.45, 0.55, 0.65, 0.75, 0.9, 1.05, 1.25):
            r = int(round(BEV_H0 + BEV_KC / Dm))
            if not (0 <= r < h):
                continue
            row = hsv[r, c0:c1]
            white = ((row[:, 1] < 60) & (row[:, 2] > 150)).astype(np.int8)
            if int(white.sum()) < 24:
                continue
            # runs of white; the checker's squares are UNIFORM width
            # (measured 29-31 px) — corner edge-line fragments are not
            # (FP signatures: [7,33,23], [2,1,3,6,6])
            runs, ln = [], 0
            for v in white:
                if v:
                    ln += 1
                elif ln:
                    runs.append(ln)
                    ln = 0
            if ln:
                runs.append(ln)
            runs = [x for x in runs if x >= 6]
            if len(runs) >= 4 and max(runs) <= 1.7 * min(runs):
                return Dm
        return None

    def _seed_road(self, low):
        """Road mask from the asphalt directly in front of the car (V channel).

        Colour-free on purpose: the road is DARKER than the grass, and that
        ordering survives a hue or saturation shift that destroys a fixed HSV
        window. Measured on 69 frames — hue/saturation perturbations leave
        this mask bit-identical (IoU 1.000) and V x0.5 / x1.8 still give
        0.962 / 0.986. Against the fixed window on unperturbed frames it
        agrees at IoU 0.966 mean, so switching to it is not a jolt.

        The seed patch is the bottom 20 % x middle 30 %, i.e. the piece of
        ground the car is about to drive over. Two known ways to poison it:
        the car sitting on grass (the referee re-centres after an offtrack,
        so exposure is short) and a cone filling the patch — the Otsu clamp
        limits how far either can drag the threshold.
        """
        v = low[:, :, 2]
        h, w = v.shape
        patch = v[int(h * 0.80):, int(w * 0.35):int(w * 0.65)]
        thr = float(np.median(patch)) + max(26.0, 2.2 * float(np.std(patch)))
        t_otsu, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = min(thr, float(t_otsu) + 12.0)
        return (v <= thr).astype(np.uint8) * 255

    def own_lane(self, hsv, crop=LOOK_TOP):
        """Bottom-anchored own-lane component on the given crop.
        (Re-added for v31 — the v28 revert removed this helper while the
        new bev_lateral still called it: AttributeError on the first
        gated tick → crash-restart loop → 0/5 timeouts. Post-integration
        offline smoke tests are now mandatory; py_compile can't see it.)"""
        low = hsv[int(hsv.shape[0] * crop):]
        road = cv2.inRange(low, ROAD_LO, ROAD_HI)
        # v64: the fixed HSV window is a sim number. CLAUDE.md already flags
        # every HSV constant as grade C ("현장 재측정 필수") because the venue's
        # printed surfaces differ in brightness and saturation. Measured how
        # badly it fails (probe_tmp/hsv_robust.py, 69 frames, IoU vs unperturbed):
        #     H-15  IoU 0.010, area 0.008     H-25  IoU 0.000, area 0.000
        #     Sx0.3 IoU 0.000, area 0.000     Vx1.8 IoU 0.160, area 0.126
        # Those are not degraded masks, they are EMPTY ones -- the road stops
        # existing and the car drives blind. The seed threshold below holds
        # 0.96-1.00 through all of them.
        # Fallback, not replacement: normal frames keep the validated window.
        # Measured over 470 labelled frames the fixed mask never drops below
        # area 0.1521 (p1 0.3154, median 0.6990), so 0.08 fires 0/470 in the
        # sim -- behaviour is unchanged here and this only earns its keep on
        # the real kit.
        if int(np.count_nonzero(road)) < ROAD_MIN_FRAC * road.size:
            road = self._seed_road(low)
        yline = cv2.inRange(low, YELLOW_LO, YELLOW_HI)
        drv = cv2.bitwise_or(road, yline)
        # v45: the band is road, it is just not road-COLOURED. Fold it in
        # before anything else looks at connectivity.
        drv = self.fold_band(drv, hsv, self.band_rows(hsv))
        # v40: reconnect the road across the start/finish checker band
        # BEFORE the component pick — after it the far half is already
        # gone (that was v39's dead end).
        drv = arc_planner.band_bridge(drv)
        n_lbl, labels = cv2.connectedComponents(drv)
        if n_lbl > 1:
            cnt = np.bincount(labels[-8:, :].ravel(), minlength=n_lbl)
            cnt[0] = 0
            own = int(cnt.argmax())
            if cnt[own] > 0:
                drv = (labels == own).astype(np.uint8) * 255
        return drv


    def band_rows(self, hsv):
        """Nearest real start/finish band → (r0, r1, D, y_centre) or None.

        A fixed centre-of-image window was WRONG: with the car 0.25 m off
        centre the band leaves the frame on one side and only covers
        65-75 % of the centre half, so the detector went blind exactly in
        the case that fails (bench: lat -0.25 crossed 0/9 while every
        other offset went 9/9). Each row is judged by the METRIC width of
        its longest low-saturation run instead, which is invariant to
        where the band sits in the image.

        y_centre is the band mid-point in vehicle metres (+ = left); on
        the labelled capture set it tracks the true offset to 1.3 cm mean
        / 3.2 cm max, and bev_lateral returns None on exactly the frames
        where this still reads."""
        h, w = hsv.shape[:2]
        crop0 = int(h * LOOK_TOP)
        low = hsv[crop0:]
        ls = low[:, :, 1] < BAND_SAT
        cnt = ls.sum(axis=1)
        # Cheap pre-filter: a band row carries DARK squares (measured
        # 127-200 px) while the near rows that flood the low-saturation
        # mask — white edge line, concrete shoulder — carry 0-1. Without
        # it the row scan has to be capped and never reaches the band when
        # the car is off centre (that cap is why lat -0.25 read as "no
        # band" on every frame and crossed 0/9 on the bench).
        dcnt = (ls & (low[:, :, 2] <= 150)).sum(axis=1)
        cand = np.nonzero((cnt >= 60) & (dcnt >= BAND_MIN_DARK))[0]
        if len(cand) == 0:
            return None
        for r in cand[::-1]:                   # nearest first
            cols = np.nonzero(ls[r])[0]
            best, s, prev = (cols[0], cols[0]), cols[0], cols[0]
            for c in cols[1:]:
                if c - prev > 8:
                    if prev - s > best[1] - best[0]:
                        best = (s, prev)
                    s = c
                prev = c
            if prev - s > best[1] - best[0]:
                best = (s, prev)
            a, b = int(best[0]), int(best[1])
            Dm = BEV_KC / max(1.0, (r + crop0) - BEV_H0)
            wm = (b - a) * Dm / BEV_FX
            clipped = (a <= 1 or b >= w - 2)
            if clipped:
                # the band runs out of frame when the car is off centre —
                # its VISIBLE width is short and the midpoint is biased,
                # so only require enough of it to be sure and mark the
                # lateral read untrusted
                #
                # v56: 0.40 m is unreachable close in. At depth Dm the
                # whole frame spans w*Dm/BEV_FX metres — 0.38 m at
                # Dm=0.19 — so no run there can pass, and that is exactly
                # where the band sits when the car is at the line. At
                # wp385, 60 of 61 candidates died on this one test, the
                # band was never found, fold_band never ran, and the
                # corridor read 0.5-0.8 m east of true: the planner then
                # picked WRONG-WAY on BOTH sides (+0.209 / -0.209) and
                # the car slid off the road across the finish segment.
                # Spanning the frame is the same "enough to be sure" test
                # in the form that close range allows; the rejected rows
                # covered 83-84 % of it; 0.80 is set from that measurement.
                frame_m = w * Dm / BEV_FX
                if wm < 0.40 and wm < 0.80 * frame_m:
                    continue
            elif not (BAND_W_MIN <= wm <= BAND_W_MAX):
                continue
            px = low[r, a:b + 1]
            white = (px[:, 1] < 60) & (px[:, 2] > 150)
            if int(((px[:, 1] < 60) & (px[:, 2] <= 150)).sum()) < BAND_MIN_DARK:
                continue                       # solid kerb, not a checker
            exp_px = BAND_SQ_M * BEV_FX / Dm
            runs, ln = [], 0
            for v in white:
                if v:
                    ln += 1
                elif ln:
                    runs.append(ln)
                    ln = 0
            if ln:
                runs.append(ln)
            big = [x for x in runs if x >= 0.45 * exp_px]
            # v56: BAND_MIN_RUNS is unreachable close in. At D=0.195 one
            # square projects to 139 px and only 403 px of band is
            # visible — 2.9 squares, so about 1.5 white runs exist and
            # asking for 3 rejects by construction. That is exactly the
            # range at which the car sits when it is AT the line, and
            # losing the band there let the corridor read 0.5-0.8 m off
            # and steered the car across the finish segment. Scale the
            # demand to the squares actually in view; far away, where 5+
            # are visible, this still asks for the full BAND_MIN_RUNS.
            # Relax ONLY in the regime that could not satisfy the width
            # floor either — the close-range rows where the geometry, not
            # the band, is what fails the test. Applied unconditionally
            # this fired on 3 frames outside the finish zone (wp228 at
            # D=0.56, wp264 at D=0.97) and broke the no-false-positive
            # property v45 established.
            need = BAND_MIN_RUNS
            if wm < 0.40:
                visible_sq = wm / max(1e-6, BAND_SQ_M)
                need = max(1, min(BAND_MIN_RUNS, int(visible_sq / 2)))
            if len(big) < need or max(big) > 3.0 * exp_px:
                continue
            r0 = r1 = int(r)
            while r0 > 0 and cnt[r0 - 1] >= 0.5 * cnt[r]:
                r0 -= 1
            while r1 + 1 < ls.shape[0] and cnt[r1 + 1] >= 0.5 * cnt[r]:
                r1 += 1
            y = (None if clipped
                 else -(((a + b) / 2) - BEV_CX) * Dm / BEV_FX)
            return (r0, r1, Dm, y)
        return None

    def fold_band(self, drv, hsv, info):
        """Paint the band's own pixels into the drivable mask."""
        if info is None:
            return drv
        r0, r1, _, _ = info
        h = drv.shape[0]
        crop0 = int(hsv.shape[0] * LOOK_TOP)
        ls = (hsv[crop0:][:, :, 1] < BAND_SAT).astype(np.uint8) * 255
        band = np.zeros_like(drv)
        lo, hi = max(0, r0 - 2), min(h - 1, r1 + 2)
        band[lo:hi + 1] = ls[lo:hi + 1]
        return cv2.bitwise_or(drv, band)

    def bev_lateral(self, hsv):
        """Metric lateral error d via multi-station BEV fit → (d, phi,
        kappa) or None. Positive d = road center is to our LEFT."""
        h, w = hsv.shape[:2]
        drv = self.own_lane(hsv)
        crop0 = int(h * LOOK_TOP)
        pts = []
        for Dm in BEV_STATIONS:
            r = int(round(BEV_H0 + BEV_KC / Dm)) - crop0
            if not (8 <= r < drv.shape[0]):
                continue   # top rows of the crop catch horizon junk
                           # (distant track strips) — smoke test: a bogus
                           # far point bent the fit to d=0.18/phi=-48°
                           # on a perfectly centered frame
            band = drv[max(0, r - 2):r + 3]
            cols = np.nonzero(band.any(axis=0))[0]
            if len(cols) < 20:
                continue
            runs, s = [], cols[0]
            for a, b in zip(cols, cols[1:]):
                if b - a > 5:
                    runs.append((s, a))
                    s = b
            runs.append((s, cols[-1]))
            c0, c1 = max(runs, key=lambda t: t[1] - t[0])
            if c0 <= 1 or c1 >= w - 2:
                continue           # clipped: center biased → reject
            y_m = -((c0 + c1) / 2 - BEV_CX) * Dm / BEV_FX
            if abs(y_m) > 0.6:
                continue   # farther off than the road is wide: not us
            pts.append((Dm, y_m, c1 - c0))
        if len(pts) < 2:
            return None
        Ds = np.array([p[0] for p in pts])
        Ys = np.array([p[1] for p in pts])
        Wt = np.sqrt([p[2] for p in pts])
        try:
            if len(pts) >= 3:
                A = np.vstack([np.ones_like(Ds), Ds, Ds ** 2]).T * Wt[:, None]
                coef, *_ = np.linalg.lstsq(A, Ys * Wt, rcond=None)
                a0, a1, a2 = coef
            else:               # 2 stations: linear (d, phi; kappa 0) —
                A = np.vstack([np.ones_like(Ds), Ds]).T * Wt[:, None]
                coef, *_ = np.linalg.lstsq(A, Ys * Wt, rcond=None)
                a0, a1 = coef   # v31b shipped without this fallback and
                a2 = 0.0        # gated itself off on every frame
        except np.linalg.LinAlgError:
            return None
        d = a0 + a1 * 0.25 + a2 * 0.0625
        phi = math.atan(a1 + 0.5 * a2)
        kap = 2 * a2 / (1 + (a1 + a2) ** 2) ** 1.5
        if abs(d) > 0.35 or abs(phi) > 0.87:
            return None    # implausible on-road state: distrust the fit
        return d, phi, kap

    def lat_center(self, hsv):
        """Near-band lateral offset from the center line (v17 sensor,
        v29 gating). Positive = line is to our right = we sit left."""
        h, w = hsv.shape[:2]
        nb = hsv[int(h * 0.65):]
        lmask = cv2.inRange(nb, YELLOW_LO, YELLOW_HI)
        ys, xs = np.nonzero(lmask)
        if len(xs) < LINE_MIN_PX:
            return 0.0
        off = (float(xs.mean()) - w / 2) / (w / 2)
        return max(-LAT_MAX, min(LAT_MAX, LAT_K * off))

    def line_feed(self, hsv):
        """Line-slope feedforward: which way does the centerline bend ahead?
        The blob centroid is direction-blind at the hairpin (measured aim
        ≈0.01 fifteen cm before the apex) — the dash slope is not."""
        h, w = hsv.shape[:2]
        nb = hsv[int(h * 0.55):]
        lmask = cv2.inRange(nb, YELLOW_LO, YELLOW_HI)
        ys, xs = np.nonzero(lmask)
        if len(xs) < LINE_MIN_PX or ys.ptp() < 12:
            return 0.0
        slope = float(np.polyfit(ys.astype(float), xs.astype(float), 1)[0])
        # x per row-downward; ahead is row-upward → bend right = -slope > 0
        feed = -slope * nb.shape[0] / (w / 2) * LINE_FEED_K
        return max(-LINE_FEED_MAX, min(LINE_FEED_MAX, feed))

    def freer_side(self):
        if self.scan is None:
            return 1.0
        s = self.scan
        r = sanitize(s.ranges)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        lm = np.median(r[(ang > 0.1) & (ang < 0.9)])
        rm = np.median(r[(ang < -0.1) & (ang > -0.9)])
        return 1.0 if lm >= rm else -1.0

    def lidar_adjust(self):
        if self.scan is None:
            return 0.0, False, LID_RANGE_MAX
        s = self.scan
        r = sanitize(s.ranges)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        front = np.abs(ang) <= LID_FRONT_HALF
        if not front.any():
            return 0.0, False, LID_RANGE_MAX
        fmin = float(r[front].min())
        if fmin > LID_AVOID:
            return 0.0, False, fmin
        left = float(np.median(r[(ang > 0) & front]))
        right = float(np.median(r[(ang < 0) & front]))
        return (0.6 if left > right else -0.6), fmin < LID_CRAWL, fmin

    def tick(self):
        self.ticks += 1
        if self.frame is None and self.ticks % 100 == 0:
            self.get_logger().warning('no camera frames — check topics/RMW')
        speed, steer, aim, mode, carea = 0.0, 0.0, 0.0, 'wait', 0.0
        if self.frame is not None:
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            if not self.green_latched:
                area = self.detect_go(hsv)
                self.green_streak = self.green_streak + 1 if area > 0 else 0
                if self.green_streak >= GREEN_FRAMES:
                    self.green_latched = True
                    self.get_logger().info(f'GREEN (area={area}) — GO')
            if self.green_latched:
                checker_d = self.detect_checker(hsv)
                ccx, ceng, carea, dodging, cbox = self.cone_pick(hsv)
                # commit guard: defer the dodge while mid-turn. Removing
                # it (v22) did NOT unlock the corner-exit cones (cone5
                # still 3/3 hit) and broke five corner lines instead —
                # even a bounded shift mid-corner exceeds the 0.47 m
                # allowance. Kept.
                if dodging and abs(self.last_steer) > 0.25:
                    dodging = False
                lbias, crawl, fmin = self.lidar_adjust()
                feed = self.line_feed(hsv)
                sharp = False
                if not dodging and self.pass_hold == 0:
                    self.pass_side = 0.0    # pass over: unlatch (teleport-
                                            # safe: cone vanishes → ≤0.4 s)
                # v33 arc lattice: one feasibility search replaces the
                # road/dodge mode split — inside cutting and early turn-in
                # fall out of "which physical arc stays legal longest".
                # Old branches remain as the fallback chain. Hold (cone
                # just below the camera) keeps priority: the planner
                # forgets an invisible cone instantly and would recenter
                # onto it (the v4-v7 rear-clip lesson).
                arc_delta = None
                if dodging or self.pass_hold == 0:
                    amask = self.own_lane(hsv)
                    if checker_d is not None:
                        # v39: the checker band erases the mask's far
                        # rows → the safe-first tiebreak hands the lap to
                        # a road-EDGE arc exactly at the finish line.
                        arc_planner.bridge_checker(amask, checker_d)
                    a_delta, a_safe, _ = arc_planner.plan(
                        amask, self.cones_ground(hsv.shape[0]))
                    if a_safe >= 0.24:
                        arc_delta = a_delta
                        self.arc_safe = a_safe
                self.last_arc = arc_delta   # None clears the hysteresis
                                            # (teleport/fallback-safe)
                if arc_delta is not None:
                    if dodging:
                        self.pass_hold = POST_PASS_TICKS
                    mode = 'arc'
                    # v33e: v29's PROVEN gated centering added to the arc
                    # steering on straights only. The planner's tiebreak
                    # can't recover offsets there (all arcs max-safe; the
                    # clipped-endpoint heuristic that wins corners is
                    # center-blind on straights) — random-run finish-zone
                    # excursions timed out 4/5. Scoring-side fixes (v33c/d)
                    # broke corner cutting instead; the lat gate (no cone,
                    # |feed| small) shields corners by construction.
                    lat = 0.0
                    if (not dodging and self.pass_hold == 0
                            and carea < CONE_ENGAGE_AREA
                            and abs(feed) < LAT_FEED_GATE
                            and abs(arc_delta) < 0.10):
                        # planner-agreement gate (v33f): |feed| alone let
                        # lat fire on short inter-corner segments where the
                        # curving dash centroid is NOT "center" — new 4/5
                        # graze at (6.5,4.0). Long straights: delta≈0.
                        lat = self.lat_center(hsv)
                        if lat == 0.0:
                            # v36: the finish approach has NO orange
                            # dashes (the checker band replaces them) —
                            # lat_center goes blind exactly where the car
                            # must be centered to CROSS THE LINE SEGMENT
                            # (measured misses passed 0.48 m east, BESIDE
                            # the line → no crossing → +5 → loop; 6/10
                            # random runs). The validated BEV d-estimator
                            # reads the road EDGES instead.
                            bev = self.bev_lateral(hsv)
                            if bev is not None:
                                lat = max(-LAT_MAX, min(LAT_MAX,
                                                        -BEV_KD * bev[0]))
                        if lat != 0.0:
                            mode = 'arc_c'
                    aim = arc_delta
                    steer = max(-MAX_STEER, min(MAX_STEER,
                                arc_delta - lat * MAX_STEER * STEER_GAIN))
                elif dodging:
                    self.pass_hold = POST_PASS_TICKS
                    off, ok, sharp = self.road_aim(hsv, line_pull=False)
                    road_c = off if ok else 0.0
                    # v21: pass = corridor aim + BOUNDED shift to the wide
                    # side, chosen from the cone's position RELATIVE TO THE
                    # ROAD CENTER in view. v24: the side is LATCHED for the
                    # whole pass — re-deciding per tick flips the moment the
                    # pass half-succeeds (the cone crosses to the other side
                    # of the road centroid in view), steering back INTO the
                    # cone: telemetry showed L-dodge → R-recover → L → hit,
                    # cone2 5/5 at 0.4 m/s. Frame-scale memory only.
                    if self.pass_side == 0.0:
                        # v26: side = which side of the cone shows more ROAD
                        # in view. The rel(cone-vs-centroid) rule inverts
                        # once the car is already deep on the narrow side
                        # (west-return trace: exiting the U wide put us west
                        # of cone6, rel said "keep west" into the 8 cm gap
                        # → offtrack+hit). Pixel gaps measure the actual
                        # clearances from the current viewpoint.
                        gl, gr = self.pass_gaps(hsv, cbox)
                        if max(gl, gr) > 1.3 * max(1, min(gl, gr)):
                            self.pass_side = -1.0 if gl > gr else 1.0
                        else:
                            rel = ccx - road_c
                            if abs(rel) > 0.04:
                                self.pass_side = -math.copysign(1.0, rel)
                            else:
                                self.pass_side = -self.freer_side()
                    shift = self.pass_side * CONE_SHIFT * min(1.0, ceng / CONE_SHIFT_RAMP)
                    mode = 'dodge'
                    aim = road_c + shift + lbias + 0.5 * feed
                    steer = max(-MAX_STEER, min(MAX_STEER,
                                -aim * MAX_STEER * STEER_GAIN))
                elif self.pass_hold > 0:
                    # cone just dropped below the camera: keep the passing
                    # line so the rear wheel clears it before recentering.
                    # v34b: a HIT-teleport strands the pre-teleport dodge
                    # steer in this hold — measured ~0.5 m lateral drift
                    # onto the fresh straight, feeding the finish-miss
                    # cascade. If a fresh plan flatly contradicts the held
                    # steer, cut the hold to one more tick; a genuine pass
                    # (mild recenter disagreement) keeps its protection.
                    self.pass_hold -= 1
                    mode = 'hold'
                    # v51: decay rather than freeze. The DURATION is load-
                    # bearing -- v48 halved it and cone3 rear clips came
                    # back 5/15 -- but the magnitude is where the measured
                    # harm is. At cone6 the frozen value was -0.279, a
                    # steer toward the cone's own side picked while the
                    # road was still curving, and holding it flat drove
                    # the car 0.115 m further west against an excursion
                    # that clears the limit by 0.019 m. HOLD_DECAY over
                    # 8 ticks leaves 43 %: a correct pass line survives,
                    # a wrong one stops compounding.
                    steer = self.last_steer * HOLD_DECAY
                    # (v34b hold-conflict cut REVERTED: it also cut
                    # legitimate cone3 passes — rear clips returned,
                    # 84.4. Teleport staleness is handled upstream by
                    # the frame-jump reset instead.)
                else:
                    off, ok, sharp = self.road_aim(hsv, line_pull=True)
                    if ok:
                        mode = 'sharp' if sharp else 'road'
                        gain = STEER_GAIN * (1.4 if sharp else 1.0)
                        lat = 0.0
                        if (not sharp and not self.sharp_lock
                                and carea < CONE_ENGAGE_AREA
                                and abs(feed) < LAT_FEED_GATE):
                            lat = self.lat_center(hsv)   # v29 source kept:
                            # BEV-d centering isolated at 85.07 vs the
                            # 81.5-83.6 same-config noise band — no win.
                            # bev_lateral stays as Phase B/C infrastructure.
                            if lat != 0.0:
                                mode = 'center'
                        aim = off + lbias + feed + lat
                        steer = max(-MAX_STEER, min(MAX_STEER,
                                    -aim * MAX_STEER * gain))
                    else:
                        mode = 'decay'
                        steer = self.last_steer * HOLD_DECAY
                # sharp lock: at a visual dead-end (hairpin) the leftover
                # blob is garbage (measured 645 px steering a wrong full
                # lock) — freeze the lock chosen at engagement until the
                # road genuinely reappears. Teleport-safe: band recovers
                # instantly after a reset, releasing the lock.
                # v33: an accepted arc bypasses and clears the lock — the
                # arc path never calls road_aim, so band_px froze and the
                # lock became permanent (sanity run: 1717 sharplock ticks,
                # full-lock circling off-road, penalty 345).
                if arc_delta is not None:
                    self.sharp_lock = False
                elif carea >= SHARP_CONE_VETO:
                    # v46: a cone within ~1 m fills the frame and empties
                    # the far band on its own, so an empty band is no
                    # longer evidence of a hairpin. Practice map, 10 runs:
                    # ALL 115 sharplock ticks were cone-induced and none
                    # sat on a real hairpin — 86 at cone4 on dead-straight
                    # track (curvature 0.000 from s=9.6 to 11.1), 22 at
                    # cone1 on an R=2.06 m sweeper. Worse, the lock chose
                    # FULL LEFT into cone4, which sits on the left, while
                    # the arc planner had been picking right correctly.
                    # Clearing (not just refusing to arm) matters: a lock
                    # latched one tick before the cone crossed the
                    # threshold would otherwise ride straight through.
                    self.sharp_lock = False
                elif (not self.sharp_lock
                      and self.band_tick == self.ticks
                      and self.band_px < 8000):
                    self.sharp_lock = True
                    # direction from the WHOLE blob (near rows keep the true
                    # curl; the engage-instant steer may still be the entry
                    # flick pointing the wrong way — v13 regression)
                    wa = getattr(self, 'whole_aim', 0.0)
                    base = -wa if abs(wa) > 0.03 else self.freer_side()
                    self.sharp_steer = math.copysign(MAX_STEER, base)
                if self.sharp_lock:
                    # A lock held on a reading nobody took this tick is the
                    # v33 failure (band_px frozen, full-lock circling,
                    # penalty 345). Release on stale as well as on recovery.
                    if (self.band_tick != self.ticks
                            or self.band_px > 15000):
                        self.sharp_lock = False
                    else:
                        steer = self.sharp_steer
                        mode = 'sharplock'
                self.last_steer = steer
                # v63: 조향 다항식 -> 횡가속 한계. 아래 상한들은 전부 그대로다.
                kap = abs(math.tan(steer)) / WHEELBASE_M
                speed = (V_MAX if kap < 1e-6
                         else min(V_MAX, math.sqrt(A_LAT / kap)))
                if self.sharp_lock:
                    speed = min(speed, 0.30)
                if abs(aim) > 0.55:
                    speed = min(speed, 0.4)
                if sharp:
                    speed = min(speed, 0.35)   # hairpin: crawl through
                if (mode in ('dodge', 'hold') or lbias != 0.0
                        or (mode == 'arc' and dodging)):
                    speed = min(speed, CONE_SLOW)
                elif carea >= CONE_ENGAGE_AREA:
                    speed = min(speed, 0.6)   # pre-slow on cone approach —
                                              # 0.9 leaves no dodge distance
                                              # (v13: cone2 hit 4/5)
                if mode == 'arc' and self.arc_safe < 0.60:
                    speed = min(speed, 0.50)  # short legal horizon = corner
                                              # right ahead (stateless
                                              # preview braking)
                if checker_d is not None:
                    speed = min(speed, 0.45)  # finish band ahead: cross it
                                              # slow and clean — a graze-
                                              # teleport here swallows the
                                              # line crossing (+70 s)
                if fmin < LID_PREBRAKE2:
                    speed = min(speed, LID_PREBRAKE2_V)
                elif fmin < LID_PREBRAKE1:
                    speed = min(speed, LID_PREBRAKE1_V)
                if crawl:
                    speed = 0.15
        # v60: a frame this old is no longer evidence about the road ahead.
        # Applied last, after every other cap, so it can only ever slow the
        # car down. Steering is left as-is (see STALE_SLOW above).
        if self.frame_t is not None:
            age = time.monotonic() - self.frame_t
            if age > STALE_STOP:
                speed = 0.0
            elif age > STALE_SLOW:
                speed = min(speed, STALE_SLOW_V)
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_steer.publish(Float64(data=float(steer)))
        self.pub_tilt.publish(Float64(data=CAM_TILT))
        if self.telem and self.ticks % 2 == 0:  # 10 Hz for calibration
            cg = getattr(self, 'cg_last', []) or []
            near = min(cg, key=lambda c: c[0]) if cg else None
            self.telem.write(f'{self.ticks},{int(self.green_latched)},'
                             f'{aim:.3f},{mode},{steer:.3f},{speed:.2f},'
                             f'{carea:.0f},{self.band_px},'
                             f'{len(self.cone_list)},'
                             f'{near[1] if near else 9.99:.3f},'
                             f'{near[0] if near else 9.99:.3f},'
                             f'{self.pass_side:+.0f},{self.pass_hold},'
                             f'{self.arc_safe:.2f}\n')


def main():
    rclpy.init()
    node = CoweekDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_speed.publish(Float64(data=0.0))
        node.pub_steer.publish(Float64(data=0.0))
        node.destroy_node()


if __name__ == '__main__':
    main()
