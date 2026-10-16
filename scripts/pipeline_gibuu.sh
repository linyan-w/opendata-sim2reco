#!/usr/bin/env bash
# Inclusive GiBUU sample over the full training volume (all flavours, CC and NC, tracker nuclei + Pb + Fe), built to
# the open dataset's composition at the exposure fraction of the NuWro inclusive sample (so the GENIE pair and its
# surrogate output of scripts/pipeline_inclusive.sh are reused), through the all-events model A, and the comparison.
# Run it as a systemd unit with a memory cap (MemoryMax), never in an interactive session.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_gibuu; G=data/gibuu; D=data/inclusive_gibuu; GEN=data/inclusive; A=reports/m3_1A_all; mkdir -p $L $D; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
i=0; TRUTH=""; RUNS=""
for g in tracker targets_pbfe; do for f in numu numubar nue nuebar; do for c in cc nc; do
  s=gibuu_me_fhc_${g}_${f}_${c}_2026-10; i=$((i + 1))
  [ -f $G/$s.truth.parquet ] || run "convert $s" $PY scripts/gibuu_to_truth.py $G/$s $G/$s.truth.parquet --volume full --pool $c --run-offset $((960000 + 1000 * i)) > $L/convert_$s.log 2>&1
  TRUTH="$TRUTH $G/$s.truth.parquet"; RUNS="$RUNS $G/$s/gibuu_run.json"
done; done; done
FRAC=$($PY -c "import json; print(json.load(open('$GEN/nuwro_inclusive.json'))['exposure_fraction'])")
run "GiBUU inclusive build" $PY scripts/nuwro_inclusive.py $D --name gibuu_inclusive --max-fraction $FRAC --truth $TRUTH --runs $RUNS > $L/build.log 2>&1
run "GiBUU surrogate" $PY scripts/surrogate_to_ntuple.py $A $D/gibuu_inclusive.truth.parquet $D/surrogate_Aall_gibuu_inclusive.root --epi-draws 8 > $L/gibuu_sur.log 2>&1
run "compare" $PY scripts/fullvolume_compare.py --genie-truth $GEN/genie_inclusive.truth.parquet --genie-sur $GEN/surrogate_Aall_genie_inclusive.root --nuwro-truth $D/gibuu_inclusive.truth.parquet --nuwro-sur $D/surrogate_Aall_gibuu_inclusive.root --out reports/inclusive_gibuu --macro-prefix nIncGB --gen-label GiBUU > $L/compare.log 2>&1
echo "$(stamp) PIPELINE GIBUU DONE" >> $L/pipeline.log
