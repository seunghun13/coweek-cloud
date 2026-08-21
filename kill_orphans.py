#!/usr/bin/env python3
"""Kill leftover run.sh / driver.py processes, by PID.

Pattern-killing is unsafe here: `pkill -f run.sh` also matches the shell
wrapper running this very command, and a self-match either kills the
caller or silently reports work it did not do. Read /proc directly,
exclude our own process tree, and kill the PIDs that remain.

run.sh restarts the driver whenever it exits, so run.sh must go first —
otherwise the driver is respawned 0.5 s later.
"""
import os
import signal
import time

ME = os.getpid()
MY_TREE = {ME}


def cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        return ''


def ppid(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            return int(f.read().split(') ', 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return 0


def in_my_tree(pid):
    seen = 0
    while pid > 1 and seen < 40:
        if pid in MY_TREE:
            return True
        pid = ppid(pid)
        seen += 1
    return False


def find():
    hits = []
    for e in os.listdir('/proc'):
        if not e.isdigit():
            continue
        pid = int(e)
        c = cmdline(pid)
        if not c:
            continue
        is_run = 'run.sh' in c and 'kill_orphans' not in c
        is_drv = 'coweek/driver.py' in c
        if (is_run or is_drv) and not in_my_tree(pid):
            hits.append((pid, 'run.sh' if is_run else 'driver', c.strip()))
    return hits


for label, order in (('run.sh', 'run.sh'), ('driver', 'driver')):
    for pid, kind, c in find():
        if kind != order:
            continue
        print(f'  kill {kind:7s} pid={pid}  {c[:90]}')
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            print(f'    (already gone: {e})')
    time.sleep(1.0)

time.sleep(1.5)
left = find()
if left:
    print('still alive, sending SIGKILL:')
    for pid, kind, c in left:
        print(f'  kill -9 {kind} pid={pid}')
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(1.0)
    left = find()

print('remaining:', left if left else 'none — clean')
print('loadavg:', open('/proc/loadavg').read().strip())
