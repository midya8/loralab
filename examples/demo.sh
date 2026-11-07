#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== evaluation ==="
python -m loralab evaluate

echo
echo "=== gates (achievable set: must pass) ==="
python -m loralab gates --min-base-drop 0.3 --min-adapted-accuracy 0.95 \
    --max-rank2-param-share 0.7 --max-merge-diff 1e-12

echo
echo "=== gates (impossible param-share cap: must fail) ==="
if python -m loralab gates --max-rank2-param-share 0.5; then
    echo "ERROR: impossible gate set passed"
    exit 1
else
    echo "impossible gate set failed as expected"
fi
