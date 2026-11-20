#!/usr/bin/env bash
# Phase 7 + 14: the main flow model (6,000 steps = 49.2M tokens) and the ablations (2,000 steps each), seed 0 unless
# stated; the reference (main settings at 2,000 steps) is trained twice to measure seed variation. Skips finished runs.
set -e
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run(){ # name, extra args...
  local name=$1; shift
  if [ -f checkpoints/$name/last.pt ]; then echo "$name exists"; return; fi
  .venv/Scripts/python -m flow.train --run $name "$@" > results/metrics/logs/$name.log 2>&1
  echo "$name done $(date +%H:%M) $(grep 'step' results/metrics/logs/$name.log | tail -1)"
}
run main
run ref_s0     --steps 2000 --evals 4 --seed 0
run ref_s1     --steps 2000 --evals 4 --seed 1
run abl_notime --steps 2000 --evals 4 --seed 0 --time_cond none
run abl_auxce  --steps 2000 --evals 4 --seed 0 --aux_ce 0.05
run abl_pca128 --steps 2000 --evals 4 --seed 0 --space pca128
run abl_tpow2  --steps 2000 --evals 4 --seed 0 --t_power 2
run abl_len32  --steps 2000 --evals 4 --seed 0 --seq_len 32 --batch_size 256   # same tokens per step
echo "all training done $(date +%H:%M)"
