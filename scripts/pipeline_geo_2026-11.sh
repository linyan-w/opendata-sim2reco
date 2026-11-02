#!/usr/bin/env bash
# The GDML-vertexed NuWro and GiBUU samples (data/geo_2026-11, HANDOFF.md; scripts/build_geo_truth.py) through the
# four-member ensemble, in slices of 2 M input events (memory), merged into one two-tree file per generator
# (scripts/merge_ntuples.py). Every slice and merge is skipped when its output exists, so the script resumes after the
# nightly reboot; slices are removed once the merged file holds every input event. Upload from the session.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_ensemble; A=reports/m3_1A_all; M="reports/m3_1A_all_s1 reports/m3_1A_all_s2 reports/m3_1A_all_s3"; D=data/geo_2026-11; SL=2000000
stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/infer.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/infer.log; exit 1; }; echo "$(stamp) done" >> $L/infer.log; }
entries() { $PY -c "import uproot; print(uproot.open('$1')['Truth'].num_entries)"; }
seed0=100
for g in nuwro gibuu; do
  T=$D/${g}_geo_2026-11.truth.parquet; OUT=$D/surrogate_Eall_${g}_geo_2026-11.root; N=$($PY -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$T').metadata.num_rows)")
  if [ ! -f $OUT ]; then
    PARTS=""
    for ((s=0, i=0; s<N; s+=SL, i++)); do
      P=$D/slices/surrogate_Eall_${g}_geo_2026-11_$(printf %02d $i).root; PARTS="$PARTS $P"; [ -f $P ] && continue
      run "ensemble $g slice $i [$s, $((s+SL)))" $PY scripts/surrogate_to_ntuple.py $A $T $P.new --range $s $((s+SL)) --seed $((seed0 + i)) --ensemble $M > $L/infer_${g}_geo_$i.log 2>&1 && mv $P.new $P
    done
    run "merge $g" $PY scripts/merge_ntuples.py $OUT.new $PARTS > $L/merge_${g}_geo.log 2>&1 && mv $OUT.new $OUT
  fi
  [ "$(entries $OUT)" = "$N" ] && rm -f $D/slices/surrogate_Eall_${g}_geo_2026-11_*.root || { echo "$(stamp) $OUT: Truth entries differ from $N input events" >> $L/infer.log; exit 1; }
  seed0=$((seed0 + 100))
done
echo "$(stamp) GEO 2026-11 ENSEMBLE DONE" >> $L/infer.log
