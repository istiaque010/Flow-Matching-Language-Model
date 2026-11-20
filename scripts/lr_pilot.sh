#!/usr/bin/env bash
# Phase 7 pilot: peak learning rate, chosen on VALIDATION flow MSE only (unconditional, fixed noise and t grid).
# 1,500 steps each (a quarter of the main run), seed 100.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
for lr in 3e-4 1e-3 3e-3; do
  .venv/Scripts/python -m flow.train --run lr$lr --lr $lr --steps 1500 --evals 3 --seed 100 > results/metrics/logs/lr$lr.log 2>&1
  tail -1 results/metrics/logs/lr$lr.log
done
