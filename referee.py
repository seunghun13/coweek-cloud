#!/usr/bin/env python3
"""Faithful Python port of the official 대회연습 evaluation script (v1).

Replicates the browser referee (evaluation.js + world script) so we can run
headless scored laps: light sequence (red → 2-5 s random → green), false
start (+10 s), object hit (+5 s, 1 cm xy / 4 cm z, latch+rearm 1 s, restore,
teleport 30 cm past obstacle), off-track (+5 s, half+0.12 m, teleport 30 cm
ahead), start-line crossing + half-track advance to finish, 180 s timeout.

Usage:  python3 referee.py [--runs N] [--jitter-cones M]
"""
import argparse
import json
import math
import random
import sys
import time as wall
import urllib.error
import urllib.request

BASE = 'http://localhost/sim/api'
FRONT = 0.12
TIME_LIMIT = 180.0
TICK_WALL = 0.07  # ~14 Hz polling


# 409 states the sim clears by itself, given a moment. "already running"
# is NOT one of them — retrying that would paper over a real conflict.
RETRY_409 = ('yellow transition in progress', 'world not ready')


def api(path, payload=None, timeout=90, tries=8):
    """One call, retrying only the 409s that are transient by construction.

    A 10-run batch once died on run 2 because setting the light to red
    landed mid yellow-transition: the HTTPError escaped main(), the
    referee never reached /evaluation/stop, and run.sh kept respawning
    the driver — an orphan that drove the car and held loadavg at 3.6.
    A crash here is far more expensive than a wait.
    """
    for attempt in range(tries):
        req = urllib.request.Request(BASE + path)
        if payload is not None:
            req.data = json.dumps(payload).encode()
            req.add_header('Content-Type', 'application/json')
            req.method = 'POST'
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code != 409 or attempt == tries - 1:
                raise
            try:
                err = json.loads(e.read()).get('error', '')
            except Exception:
                err = ''
            if not any(k in err for k in RETRY_409):
                raise
            print(f'  api {path}: 409 "{err}" — retry {attempt + 1}/{tries}',
                  flush=True)
            wall.sleep(0.5)
    try:
        return json.loads(body)
    except Exception:
        return {}


class Track:
    def __init__(self):
        r = api('/route')
        self.wp = r['waypoints']
        self.inner = r['inner']
        self.outer = r['outer']
        self.N = len(self.wp) - 1
        self.seg = [math.hypot(self.wp[i + 1][0] - self.wp[i][0],
                               self.wp[i + 1][1] - self.wp[i][1])
                    for i in range(len(self.wp) - 1)]
        self.length = sum(self.seg)

    def arc_of(self, x, y):
        """Arc length along the route of the point nearest (x, y)."""
        i = min(range(self.N),
                key=lambda k: (self.wp[k][0] - x) ** 2
                + (self.wp[k][1] - y) ** 2)
        d = 0.0
        for k in range(i):
            d += math.hypot(self.wp[k + 1][0] - self.wp[k][0],
                            self.wp[k + 1][1] - self.wp[k][1])
        return d

    def at_arc(self, s):
        """(x, y, left-normal, idx) at arc distance s along the centerline."""
        s %= self.length
        d, i = 0.0, 0
        while i < self.N and d + self.seg[i] < s:
            d += self.seg[i]
            i += 1
        w, q = self.wp[i], self.wp[(i + 1) % self.N]
        tl = math.hypot(q[0] - w[0], q[1] - w[1]) or 1.0
        f = (s - d) / tl
        return (w[0] + (q[0] - w[0]) * f, w[1] + (q[1] - w[1]) * f,
                (-(q[1] - w[1]) / tl, (q[0] - w[0]) / tl), i)

    def local_radius(self, i, arm=5):
        a = self.wp[max(0, i - arm)]
        b = self.wp[i]
        c = self.wp[min(self.N - 1, i + arm)]
        cross = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if abs(cross) < 1e-9:
            return 1e9
        ab = math.hypot(b[0] - a[0], b[1] - a[1])
        bc = math.hypot(c[0] - b[0], c[1] - b[1])
        ca = math.hypot(a[0] - c[0], a[1] - c[1])
        return ab * bc * ca / (2 * abs(cross))


PRACTICE_OFF = 0.179       # measured mean of the practice layout


def random_cone_layout(track, names, lat=None, min_gap=1.5, start_clear=2.0,
                       finish_clear=1.40, pinned_s=(),
                       min_radius=0.6, tries=200, edge=True):
    """Fresh random placement (finals model). Per the practice-venue
    photo (user 8/19): cones sit FLUSH AGAINST the white edge line —
    center at ±(half-0.10), aligned with the ROAD TANGENT (a map-aligned
    square-based cone reads diagonal to the camera on slants). Uniform
    arc position, straight-ish spots, min arc gap, start area clear.
    edge=False keeps the harder lane-center variant (±half/2) as a
    stress distribution."""
    placed, out = list(pinned_s), {}
    # pinned cones keep their world pose; their arc positions still block
    # the min_gap check so a random cone cannot land on top of one
    for n in names:
        for _ in range(tries):
            s = random.uniform(0.0, track.length)
            if s < start_clear or (track.length - s) < finish_clear:
                continue        # finish_clear = 1.40 m: MEASURED from the
                                # official practice map, where cone6 sits
                                # s=29.103 of 30.505, i.e. 1.402 m before
                                # the line. A cone that close to the finish
                                # is real and must be handled; we simply do
                                # not invent anything CLOSER (user call,
                                # 2026-08-19). The old symmetric 2.0 was
                                # LOOSER than the real map.
            if any(min(abs(s - p), track.length - abs(s - p)) < min_gap
                   for p in placed):
                continue
            x, y, nrm, i = track.at_arc(s)
            if track.local_radius(i) < min_radius:
                continue
            inn, outr = track.inner[i], track.outer[i]
            half = math.hypot(inn[0] - outr[0], inn[1] - outr[1]) / 2
            off = random.choice([-1.0, 1.0]) * (
                PRACTICE_OFF if edge else half / 2)
            # PRACTICE_OFF: the practice map's own offset, MEASURED from
            # state.json (all six cones sit 0.176-0.184 m off the centre,
            # mean 0.179). The earlier model used half-0.10 = 0.252, which
            # is 7 cm further out than the venue actually sets them and
            # squeezes the narrow gap from 0.174 m to 0.100 m — a harder
            # track than the real one. User call 2026-08-19: randomise the
            # POSITION along the track, keep the offset as the map has it.
            # Reference (measured from a frame): asphalt edge 0.344,
            # white kerb line 0.352-0.439, grass beyond 0.44.
            yaw = math.atan2(track.wp[(i + 1) % track.N][1] - track.wp[i][1],
                             track.wp[(i + 1) % track.N][0] - track.wp[i][0])
            placed.append(s)
            out[n] = (x + nrm[0] * off, y + nrm[1] * off, round(s, 2),
                      round(off, 3), round(yaw, 3))
            break
    return out


class Referee:
    def __init__(self, track, log):
        self.tr = track
        self.log = log
        self.prev = None
        self.last_idx = None
        self.adv = 0
        self.base = None
        self.start_t = None
        self.penalty = 0.0
        self.off_latch = False
        self.hit = {}
        self.red_t = None
        self.green_sent = False
        self.green_delay = None
        self.car_start = None
        self.result = None
        self.finished = False
        self.events = []

    # ── geometry helpers (1:1 ports) ──
    def nearest_idx(self, x, y):
        w, bi, bd = self.tr.wp, 0, 1e18
        for i in range(len(w) - 1):
            d = (w[i][0] - x) ** 2 + (w[i][1] - y) ** 2
            if d < bd:
                bd, bi = d, i
        return bi

    def nearest_idx_near(self, x, y, from_idx, back, fwd):
        w, N = self.tr.wp, self.tr.N
        bi, bd = from_idx, 1e18
        for k in range(-back, fwd + 1):
            i = ((from_idx + k) % N + N) % N
            d = (w[i][0] - x) ** 2 + (w[i][1] - y) ** 2
            if d < bd:
                bd, bi = d, i
        return bi

    def fwd_index(self, from_idx, dist):
        w, N = self.tr.wp, self.tr.N
        d, i = 0.0, from_idx
        while d < dist:
            j = (i + 1) % N
            d += math.hypot(w[i][0] - w[j][0], w[i][1] - w[j][1])
            i = j
            if i == from_idx:
                break
        return i

    def front_point(self, pose):
        return (pose['x'] + math.cos(pose['yaw']) * FRONT,
                pose['y'] + math.sin(pose['yaw']) * FRONT)

    def crossed_start_line(self, p):
        if self.prev is None:
            return False
        a, b = self.tr.inner[0], self.tr.outer[0]
        px, py = self.prev
        d = (p[0] - px) * (b[1] - a[1]) - (p[1] - py) * (b[0] - a[0])
        if abs(d) < 1e-12:
            return False
        t = ((a[0] - px) * (b[1] - a[1]) - (a[1] - py) * (b[0] - a[0])) / d
        u = ((a[0] - px) * (p[1] - py) - (a[1] - py) * (p[0] - px)) / d
        return 0 <= t <= 1 and 0 <= u <= 1

    def track_advance(self, pose):
        N = self.tr.N
        i = (self.nearest_idx(pose['x'], pose['y']) if self.last_idx is None
             else self.nearest_idx_near(pose['x'], pose['y'], self.last_idx, 10, 15))
        if self.last_idx is not None:
            d = i - self.last_idx
            if d > N / 2:
                d -= N
            elif d < -N / 2:
                d += N
            self.adv += d
        self.last_idx = i

    def off_track(self, pose, margin):
        i = self.nearest_idx(pose['x'], pose['y'])
        w = self.tr.wp[i]
        inn, out = self.tr.inner[i], self.tr.outer[i]
        half = math.hypot(inn[0] - out[0], inn[1] - out[1]) / 2
        return math.hypot(pose['x'] - w[0], pose['y'] - w[1]) > half + margin

    def move_to_center(self, pose):
        i = (self.nearest_idx(pose['x'], pose['y']) if self.last_idx is None
             else self.nearest_idx_near(pose['x'], pose['y'], self.last_idx, 10, 15))
        i = self.fwd_index(i, 0.3)
        w = self.tr.wp
        q = w[(i + 1) % len(w)]
        api('/pose', {'x': w[i][0], 'y': w[i][1],
                      'yaw': math.atan2(q[1] - w[i][1], q[0] - w[i][0])})

    # ── objects ──
    def snapshot_base(self, objects):
        self.base = {n: dict(p) for n, p in objects.items()}

    def displaced(self, objects, name, tol):
        b, p = self.base.get(name), objects.get(name)
        if not b or not p:
            return False
        return (math.hypot(p['x'] - b['x'], p['y'] - b['y']) > tol
                or abs(p.get('z', 0) - b.get('z', 0)) > 0.04)

    def restore(self, name):
        b = self.base[name]
        api(f'/models/{name}/pose', {'x': b['x'], 'y': b['y'], 'yaw': b.get('yaw', 0)})

    def drive_light(self, t, light):
        if self.green_sent:
            return
        if self.red_t is None:
            if light == 'red':
                self.red_t = t
                self.green_delay = 2.0 + random.random() * 3.0
            return
        if t - self.red_t >= self.green_delay:
            self.green_sent = True
            api('/traffic_lights/light1', {'state': 'green'})
            self.events.append(('green_on', t))

    # ── the per-tick evaluate(), 1:1 with the official script ──
    def evaluate(self, st):
        t = st['time']
        pose = st['vehicle']
        objects = st.get('objects', {})
        lights = {l['name']: l['state'] for l in st.get('lights', [])}
        light = lights.get('light1')

        if self.base is None:
            self.snapshot_base(objects)

        if self.start_t is None:
            for n in list(self.base.keys()):
                h = self.hit.get(n)
                if self.displaced(objects, n, 0.01):
                    if not h or t >= h['rearmT']:
                        self.restore(n)
                        self.hit[n] = {'latched': False, 'rearmT': t + 1.0}
            if self.car_start is None:
                self.car_start = (pose['x'], pose['y'])
            self.drive_light(t, light)
            go_green = self.green_sent and light == 'green'
            moved = math.hypot(pose['x'] - self.car_start[0],
                               pose['y'] - self.car_start[1]) > 0.03
            if go_green or moved:
                self.start_t = t
                self.prev = None
                self.events.append(('clock_start', t))
                if moved and light != 'green':
                    self.penalty += 10
                    self.events.append(('false_start_+10', t))
                    api('/overlay', {'text': '+10s penalty (false start)', 'ttl': 4})
            return

        self.drive_light(t, light)

        f = self.front_point(pose)
        if self.prev and math.hypot(f[0] - self.prev[0], f[1] - self.prev[1]) > 0.5:
            self.prev = None
        crossed = self.crossed_start_line(f)
        self.prev = f
        if crossed and self.adv > self.tr.N / 2:
            self.finished = True
            self.result = t - self.start_t + self.penalty
            return

        self.track_advance(pose)

        if self.off_track(pose, 0.12):
            if not self.off_latch:
                self.off_latch = True
                self.penalty += 5
                self.events.append((f'offtrack_+5@({pose["x"]:.2f},{pose["y"]:.2f})', t))
                api('/overlay', {'text': '+5s penalty (off track)', 'ttl': 4})
                self.move_to_center(pose)
                self.prev = None
        else:
            self.off_latch = False

        for n in list(self.base.keys()):
            moved = self.displaced(objects, n, 0.01)
            h = self.hit.get(n)
            if h and h.get('latched'):
                if t < h['rearmT']:
                    continue
                if not moved:
                    h['latched'] = False
                    continue
                self.restore(n)
                h['rearmT'] = t + 1.0
                continue
            if moved:
                self.hit[n] = {'latched': True, 'rearmT': t + 1.0}
                self.penalty += 5
                self.events.append((f'hit_{n}_+5', t))
                api('/overlay', {'text': f'+5s penalty (hit {n})', 'ttl': 4})
                self.restore(n)
                b = self.base[n]
                car_idx = (self.last_idx if self.last_idx is not None
                           else self.nearest_idx(pose['x'], pose['y']))
                obs_idx = self.nearest_idx_near(b['x'], b['y'], car_idx, 5, 20)
                bi = self.fwd_index(obs_idx, 0.3)
                w = self.tr.wp
                q = w[(bi + 1) % self.tr.N]
                api('/pose', {'x': w[bi][0], 'y': w[bi][1],
                              'yaw': math.atan2(q[1] - w[bi][1], q[0] - w[bi][0])})
                self.prev = None
        self.result = t - self.start_t + self.penalty  # running value


def one_run(track, jitter=0.0, randomize=False, rand_lat=0.20, log=print,
            start_clear=2.0, finish_clear=1.40, pin=()):
    api('/evaluation/stop', {})
    api('/reset', {})
    wall.sleep(2.0)

    # initialize(): red + park 10 cm behind wp0 (light goes yellow 3 s first)
    api('/traffic_lights/light1', {'state': 'red'})
    wp = track.wp
    yaw = math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0])
    api('/pose', {'x': wp[0][0] - math.cos(yaw) * 0.10,
                  'y': wp[0][1] - math.sin(yaw) * 0.10, 'yaw': yaw})
    wall.sleep(3.5)  # let yellow → red settle BEFORE base snapshot

    jit = {}
    if randomize:
        names = []
        for _ in range(10):     # /state can race the reset settling and
            st = api('/state')  # return no objects → runs 2-5 silently
            names = sorted(n for n in st.get('objects', {})  # reused run
                           if n.startswith('cone'))          # 1's layout
            if names:
                break
            wall.sleep(1.0)
        # PIN (user call 2026-08-20): the LAST cone before the finish is
        # the single biggest bottleneck — corner exit at s=29.10, 1.40 m
        # before the line. Randomising it away makes a batch look good for
        # the wrong reason, so it stays where the practice map puts it
        # until goal 1 (a penalty-free lap, repeatably) is met. Then all
        # six randomise.
        st0 = api('/state')
        pin_s, free = [], []
        for n in names:
            if n in pin:
                p = st0.get('objects', {}).get(n)
                if p:
                    pin_s.append(track.arc_of(p['x'], p['y']))
            else:
                free.append(n)
        layout = random_cone_layout(track, free,
                                    start_clear=start_clear,
                                    finish_clear=finish_clear,
                                    pinned_s=pin_s)
        if len(layout) < len(free):
            print(f'WARN: placed only {len(layout)}/{len(free)} free cones '
                  f'(+{len(pin_s)} pinned)',
                  flush=True)
        for n, (x, y, s, off, yaw) in layout.items():
            jit[n] = (s, off)
            api(f'/models/{n}/pose', {'x': x, 'y': y, 'yaw': yaw})
        wall.sleep(1.0)
    elif jitter > 0:
        st = api('/state')
        for n, p in st.get('objects', {}).items():
            if not n.startswith('cone'):
                continue
            dx = random.uniform(-jitter, jitter)
            dy = random.uniform(-jitter, jitter)
            jit[n] = (round(dx, 3), round(dy, 3))
            api(f'/models/{n}/pose', {
                'x': p['x'] + dx, 'y': p['y'] + dy, 'yaw': p.get('yaw', 0)})
        wall.sleep(1.0)

    ref = Referee(track, log)
    api('/evaluation/run', {})          # spawns: source run.sh
    t0 = None
    trace = []
    rtfs = []                # sim real-time factor, sampled every poll
    while True:
        st = api('/state')
        t = st['time']
        if isinstance(st.get('rtf'), (int, float)):
            rtfs.append(float(st['rtf']))
        if t0 is None:
            t0 = t
        ref.evaluate(st)
        v = st['vehicle']
        trace.append((round(t - t0, 2), round(v['x'], 3), round(v['y'], 3)))
        if ref.finished:
            api('/evaluation/stop', {})
            api('/overlay', {'text': f'FINISH  {ref.result:.2f}s '
                                     f'(lap {ref.result - ref.penalty:.2f} '
                                     f'+ pen {ref.penalty:.0f})', 'ttl': 10})
            return {'outcome': 'finished', 'score': round(ref.result, 3),
                    'penalty': ref.penalty, 'events': ref.events, 'jit': jit,
                    'lap_time': round(ref.result - ref.penalty, 3),
                    'rtf_min': round(min(rtfs), 3) if rtfs else None,
                    'rtf_mean': round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
                    'trace': trace}
        if t - t0 > TIME_LIMIT:
            api('/evaluation/stop', {})
            return {'outcome': 'timeout', 'score': None, 'penalty': ref.penalty,
                    'events': ref.events, 'adv': ref.adv, 'jit': jit,
                    'rtf_min': round(min(rtfs), 3) if rtfs else None,
                    'rtf_mean': round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
                    'trace': trace}
        wall.sleep(TICK_WALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--jitter-cones', type=float, default=0.0,
                    help='random ± m applied to each cone before the run')
    ap.add_argument('--random-cones', action='store_true',
                    help='finals model: fresh random placement on the road '
                         '(uniform arc pos, straight-ish, ±lat of centerline)')
    ap.add_argument('--start-clear', type=float, default=2.0,
                    help='arc metres kept clear of cones AFTER the start '
                         'line')
    ap.add_argument('--finish-clear', type=float, default=1.40,
                    help='arc metres kept clear of cones BEFORE the finish '
                         'line. 1.40 = where the official practice map puts '
                         'cone6 (s=29.103 of 30.505). Cones that close are '
                         'real; we just never place one closer.')
    ap.add_argument('--pin', type=str, default='cone6',
                    help="cones NOT randomised, comma separated. Default "
                         "cone6: the last cone before the finish is the "
                         "current bottleneck, and randomising it away "
                         "flatters the result. Pass --pin '' for the real "
                         "finals model once goal 1 is met.")
    ap.add_argument('--random-lat', type=float, default=0.20,
                    help='max |lateral offset| for --random-cones')
    ap.add_argument('--seed', type=int, default=None,
                    help='seed the cone layouts and green delays so two '
                         'batches see the IDENTICAL sequence of runs. '
                         'Default None = nothing is seeded, i.e. byte '
                         'identical to before this flag existed. Use it to '
                         'PAIR an A/B: the same v64 build scored 130/170/200 '
                         'penalty across three unseeded finals batches, so '
                         'an unpaired 15-run arm has very little power.')
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    track = Track()
    results = []
    try:
        run_batch(args, track, results)
    finally:
        # run.sh respawns the driver whenever it exits, so a referee that
        # dies without this leaves an orphan driving the car and eating a
        # third of a core — silently slowing every measurement that
        # follows (measured: loadavg 1.6 -> 3.6 on 8/20).
        try:
            api('/evaluation/stop', {})
        except Exception as e:
            print(f'WARN: could not stop the evaluation: {e}', flush=True)
    summarise(results)


def run_batch(args, track, results):
    for k in range(args.runs):
        r = one_run(track, jitter=args.jitter_cones,
                    randomize=args.random_cones, rand_lat=args.random_lat,
                    start_clear=args.start_clear,
                    finish_clear=args.finish_clear,
                    pin=tuple(n for n in args.pin.split(',') if n))
        results.append(r)
        with open('/tmp/ref_traces.jsonl', 'a' if k else 'w') as f:
            f.write(json.dumps({'run': k + 1, 'events': r['events'],
                                'outcome': r['outcome'],
                                'score': r.get('score'),
                                'penalty': r.get('penalty'),
                                # jit = {name: (arc_s, lateral_off)}
                                # for the cones this run MOVED. Pinned
                                # cones are absent (they keep their
                                # world pose). Without this the trace
                                # plot drew every run against the
                                # practice-map cone positions.
                                'jit': r.get('jit', {}),
                                'rtf_min': r.get('rtf_min'),
                                'rtf_mean': r.get('rtf_mean'),
                                'trace': r.get('trace', [])}) + '\n')
        ev = ' '.join(f'{e[0]}@{e[1]:.1f}' for e in r['events'])
        jt = ' '.join(f'{n}:({d[0]},{d[1]})' for n, d in r.get('jit', {}).items())
        rtf = ''
        if r.get('rtf_mean') is not None:
            rtf = f" rtf={r['rtf_mean']:.3f}/min{r['rtf_min']:.3f}"
            # Flag on the MEAN, not the min: every run dips at the start
            # (reset, spawn, the referee's own pose writes), so a min-based
            # flag fired on all 15 runs of a batch whose mean rtf was
            # 0.998-1.000 — an environment that was perfectly fine.
            if r['rtf_mean'] < 0.95:
                rtf += ' <<DEGRADED'
        print(f"RUN {k+1}: {r['outcome']} score={r.get('score')} "
              f"penalty={r['penalty']}{rtf} {ev} JIT[{jt}]", flush=True)


def summarise(results):
    done = [r['score'] for r in results if r['outcome'] == 'finished']
    if done:
        print(f"SUMMARY: {len(done)}/{len(results)} finished, "
              f"best={min(done)}, mean={sum(done)/len(done):.2f}")
    else:
        print(f"SUMMARY: 0/{len(results)} finished")


if __name__ == '__main__':
    main()
