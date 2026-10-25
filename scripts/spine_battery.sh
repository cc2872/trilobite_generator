#!/bin/bash
# one process per animal (a 4 GB build is OOM-killed when they share a process)
cd "$(dirname "$0")/.."
for a in $(python3 scripts/spine_battery.py log/battery --list); do
  python3 scripts/spine_battery.py log/battery "$a" --one 2>&1 | grep -v "^done"
done
python3 scripts/spine_battery.py log/battery --summary
echo BATTERY-DONE
