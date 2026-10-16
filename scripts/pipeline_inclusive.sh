#!/usr/bin/env bash
# Inclusive generator comparison over the full training volume: one NuWro sample of all flavours (nu_mu, nu_mu-bar,
# nu_e, nu_e-bar), CC and NC, on the tracker nuclei + Pb + Fe, built to the open dataset's exposure and composition
# (scripts/nuwro_inclusive.py), the matching open-dataset (GENIE) sample, both through the all-events model A, and the
# reco-fiducial comparison. Run it as a systemd unit with a memory cap (MemoryMax), never in an interactive session.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_inclusive; D=data/inclusive; F=data/nuwro_flavours; A=reports/m3_1A_all; mkdir -p $L $F; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
i=0; TRUTH="data/nuwro_2026-10/nuwro_me_fhc_tracker.truth.parquet data/fullvolume/nuwro_pbfe.truth.parquet data/nuwro_nc_2026-10/nuwro_me_fhc_tracker_nc.truth.parquet data/fullvolume/nuwro_pbfe_nc.truth.parquet"
RUNS="data/nuwro_2026-10/nuwro_run.json data/nuwro_pbfe_2026-10/nuwro_run.json data/nuwro_nc_2026-10/nuwro_run.json data/nuwro_pbfe_nc_2026-10/nuwro_run.json"
for g in tracker targets_pbfe; do for f in numubar nue nuebar; do for c in cc nc; do
  s=nuwro_me_fhc_${g}_${f}_${c}_2026-10; i=$((i + 1))
  [ -f $F/$s.truth.parquet ] || run "convert $s" $PY scripts/nuwro_to_truth.py $F/$s/jobs $F/$s.truth.parquet --volume full --pool $c --run-offset $((940000 + 1000 * i)) > $L/convert_$s.log 2>&1
  TRUTH="$TRUTH $F/$s.truth.parquet"; RUNS="$RUNS $F/$s/nuwro_run.json"
done; done; done
run "NuWro inclusive build" $PY scripts/nuwro_inclusive.py $D --truth $TRUTH --runs $RUNS > $L/build.log 2>&1
run "GENIE inclusive build" $PY scripts/nuwro_inclusive.py $D --genie-only > $L/genie.log 2>&1
run "NuWro surrogate" $PY scripts/surrogate_to_ntuple.py $A $D/nuwro_inclusive.truth.parquet $D/surrogate_Aall_nuwro_inclusive.root --epi-draws 8 > $L/nuwro_sur.log 2>&1
run "GENIE surrogate" $PY scripts/surrogate_to_ntuple.py $A $D/genie_inclusive.truth.parquet $D/surrogate_Aall_genie_inclusive.root --epi-draws 8 > $L/genie_sur.log 2>&1
run "compare" $PY scripts/fullvolume_compare.py --genie-truth $D/genie_inclusive.truth.parquet --genie-sur $D/surrogate_Aall_genie_inclusive.root --nuwro-truth $D/nuwro_inclusive.truth.parquet --nuwro-sur $D/surrogate_Aall_nuwro_inclusive.root --out reports/inclusive --macro-prefix nIncOFV > $L/compare.log 2>&1
echo "$(stamp) PIPELINE INCLUSIVE DONE" >> $L/pipeline.log
