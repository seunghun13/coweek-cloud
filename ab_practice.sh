#!/bin/bash
# Same-conditions A/B of the current build against a git ref.
#
# History of what this script has had to defend against, all of it real:
#   - batch() did `cd $WS`, moving the cwd for the rest of the script, so
#     the checkout ran in a different repository, failed, and BOTH halves
#     measured the same build. Every cd now happens in a subshell.
#   - `git checkout <ref> -- file` stages the ref, so restoring the file
#     alone left the index holding the old build and the next commit
#     would have shipped it. The index is restored and verified.
#   - it swapped driver.py only. v50 changed arc_planner.py, which would
#     have been left at the working version in BOTH halves — a silent
#     no-op A/B. Every file in FILES is now swapped and md5-checked.
#   - a batch that dies leaves run.sh respawning the driver, and the next
#     batch runs with two drivers fighting: laps go 60 s -> 86 s with no
#     warning. Orphans are swept before and after each batch.
#
# usage: ab_practice.sh [runs] [ref] [extra referee args...]
#   ab_practice.sh 15 racer-v49
#   ab_practice.sh 15 racer-v49 --random-cones
set -euo pipefail
WS=/home/physicar/physicar_ws
CW=$WS/coweek
RUNS=${1:-10}
REF=${2:-racer-v45}
shift 2 2>/dev/null || true
EXTRA=("$@")

# Everything that defines the driving build. Adding a file here is what
# keeps a future A/B honest when the change lands somewhere new.
FILES=(driver.py arc_planner.py)

sweep() { python3 "$CW/kill_orphans.py" | tail -2; }

sig() {   # combined md5 of the build, order-stable
  ( cd "$CW" && md5sum "${FILES[@]}" | md5sum | cut -d' ' -f1 )
}

show() { ( cd "$CW" && md5sum "${FILES[@]}" | sed 's/^/    /' ); }

batch() {   # batch <tag> <expected-sig>
  local tag=$1 want=$2 got
  got=$(sig)
  if [ "$got" != "$want" ]; then
    echo "!! ABORT: $tag expected build sig $want but found $got"
    show
    exit 1
  fi
  echo "=== $tag: $RUNS runs ${EXTRA[*]-} — sig $got — load $(cut -d' ' -f1-3 /proc/loadavg) ==="
  show
  sweep
  ( cd "$WS" && python3 -u coweek/referee.py --runs "$RUNS" ${EXTRA[@]+"${EXTRA[@]}"} \
      > "/tmp/ab_${tag}.txt" 2>&1 ) || echo "!! $tag: referee exited nonzero"
  cp /tmp/ref_traces.jsonl "/tmp/ab_${tag}_traces.jsonl" 2>/dev/null || true
  cp /tmp/coweek_telem.csv "/tmp/ab_${tag}_telem.csv" 2>/dev/null || true
  sweep
  grep -E '^SUMMARY' "/tmp/ab_${tag}.txt" || echo "!! $tag produced no SUMMARY"
}

CUR_SIG=$(sig)
for f in "${FILES[@]}"; do cp "$CW/$f" "/tmp/abcur_$f"; done
( cd "$CW" && git checkout "$REF" -- "${FILES[@]}" )
REF_SIG=$(sig)
for f in "${FILES[@]}"; do cp "$CW/$f" "/tmp/abref_$f"; done
for f in "${FILES[@]}"; do cp "/tmp/abcur_$f" "$CW/$f"; done
( cd "$CW" && git reset -q HEAD -- "${FILES[@]}" )

if [ "$CUR_SIG" = "$REF_SIG" ]; then
  echo "!! ABORT: the working build is already identical to $REF across"
  echo "   ${FILES[*]} — there is nothing to compare."
  exit 1
fi
echo "current sig = $CUR_SIG"
echo "$REF sig     = $REF_SIG"

batch cur "$CUR_SIG"

echo
echo "--- switching ${FILES[*]} to $REF ---"
for f in "${FILES[@]}"; do cp "/tmp/abref_$f" "$CW/$f"; done
batch ref "$REF_SIG"

echo
echo "--- restoring the working build ---"
for f in "${FILES[@]}"; do cp "/tmp/abcur_$f" "$CW/$f"; done
( cd "$CW" && git reset -q HEAD -- "${FILES[@]}" )
[ "$(sig)" = "$CUR_SIG" ] && echo "restore verified ($CUR_SIG)" \
  || { echo "!! RESTORE FAILED"; show; exit 1; }
( cd "$CW" && git diff --cached --quiet -- "${FILES[@]}" ) \
  && echo "index clean" || { echo "!! INDEX STILL DIRTY"; exit 1; }
sweep
echo "=== A/B COMPLETE ==="
