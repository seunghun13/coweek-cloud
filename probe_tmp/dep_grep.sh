#!/bin/bash
cd /home/physicar/physicar_ws
PAT='rclpy|sensor_msgs|LaserScan|Odometry|image_raw|scan_filtered|/scan|/imu|/odom|ros2|cv_bridge|DDS|CycloneDDS'
for f in wick.py wick_v2.py wick_v3.py coweek/driver.py; do
  echo "--- $f : $(grep -Ecin "$PAT" "$f") matching lines"
  grep -Ein "$PAT" "$f" | head -6
done
echo
echo "=== every /sim/api path used in wick_v3 ==="
grep -n "sim_get(\|sim_post(" wick_v3.py
echo
echo "=== every non-sim HTTP path in wick_v3 ==="
grep -n "self.base +\|base + '/" wick_v3.py
echo
echo "=== file IO in wick_v3 ==="
grep -n "imread\|imwrite\|open(\|makedirs\|glob" wick_v3.py
echo
echo "=== env vars in wick_v3 ==="
grep -n "environ\|getenv" wick_v3.py || echo "(none)"
echo
echo "=== run.sh (friend) ==="
cat /home/physicar/physicar_ws/run.sh 2>/dev/null | head -20
