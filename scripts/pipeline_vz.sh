#!/usr/bin/env bash
# Vertex-class head conditioned on the flow's z (consistent plane snapping, restores the vertex-z tails): head-only
# training warm-started from the adopted M3 models, standard evaluation, and the tail isolation test on each.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_vz; mkdir -p $L; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
V=${VZ_MODE:-}; for M in m3_1A_all m3_1A_z; do
  run "train $M head" $PY -u scripts/run_m3.py reports/${M}_vz$V --init reports/$M/model.pt --epochs 3 --lr 2e-4 --steps 64 --vtx-cond-z ${V#_} --only-vtx --m1 none > $L/train_$M.log 2>&1
  cp reports/$M/bayes_last.pt reports/${M}_vz$V/   # the Bayesian last layers belong to the unchanged heads and flows
  run "isolate $M" $PY scripts/vtx_tail_isolate.py --model reports/${M}_vz$V --out reports/vtx_tail/isolate_${M}_vz$V.json > $L/isolate_$M.log 2>&1
done
echo "$(stamp) PIPELINE VZ DONE" >> $L/pipeline.log
