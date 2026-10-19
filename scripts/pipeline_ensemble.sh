#!/usr/bin/env bash
# Ensemble of the deployed all-events model A: members s=1..N trained like A (M2 -> M3 -> z-conditioned vertex head,
# Bayesian last layers) with a different initialisation and shuffling seed; the split and the transforms are the same.
# Every stage is skipped when its output exists, and a stage is not started unless it can finish before the nightly
# reboot (00:05); a persistent user timer (sim2reco-ensemble.timer) relaunches the chain each day until done.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_ensemble; mkdir -p $L; stamp() { date "+%F %T"; }; N=${ENS_MEMBERS:-3}
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
fits() { local need=$1; local now=$(date +%s); local cutoff=$(date -d "today 23:50" +%s); [ $((now + need)) -le $cutoff ] || { echo "$(stamp) deferred: $2 needs ${need}s, past 23:50" >> $L/pipeline.log; exit 0; }; }
pgrep -f "scripts/run_m[23].py|surrogate_to_ntuple.py" >/dev/null && { echo "$(stamp) GPU busy, deferred" >> $L/pipeline.log; exit 0; }
for s in $(seq 1 $N); do
  M2=reports/m2_1A_all_s$s; M3=reports/m3_1A_all_s$s
  [ -f $M2/metrics.json ] || { fits 9000 "M2 s$s"; run "M2 s$s" $PY -u scripts/run_m2.py $M2 --epochs 30 --steps 64 --population all --seed $s --m1 none > $L/m2_s$s.log 2>&1; }
  [ -f $M3/metrics_tier2.json ] || { fits 6600 "M3 s$s"; run "M3 s$s" $PY -u scripts/run_m3.py $M3 --init $M2/model.pt --epochs 16 --lr 2e-4 --steps 64 --seed $s --m1 none > $L/m3_s$s.log 2>&1; }
  [ -f ${M3}_vz_far/model.pt ] || { fits 2400 "head s$s"; run "head s$s" env VZ_MODE=_far VZ_MODELS="m3_1A_all_s$s" scripts/pipeline_vz.sh; }
  [ -f $M3/history_vz.json ] || { cp $M3/model.pt ${M3}_prevz.pt; cp ${M3}_vz_far/model.pt $M3/model.pt; cp ${M3}_vz_far/history.json $M3/history_vz.json; echo "$(stamp) adopted the z-conditioned head in $M3" >> $L/pipeline.log; }
  [ -f $M3/bayes_last.pt ] || { fits 1200 "bayes s$s"; run "bayes s$s" $PY scripts/fit_bayes_last.py $M3 > $L/bayes_s$s.log 2>&1; }
done
echo "$(stamp) PIPELINE ENSEMBLE DONE ($N members)" >> $L/pipeline.log
