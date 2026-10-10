#!/usr/bin/env bash
# Out-of-fiducial-volume validation: NuWro (tracker nuclei + Pb + Fe campaigns) and GENIE with full-training-volume
# vertices (true 4000 <= z <= 8700 mm, tracker and outside it), through model A.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_fullvolume; O=data/fullvolume; mkdir -p $L $O; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1"; shift; "$@" || { echo "$(stamp) FAILED"; exit 1; }; echo "$(stamp) done"; }
run "NuWro Pb/Fe convert" $PY scripts/nuwro_to_truth.py data/nuwro_pbfe_2026-10/jobs $O/nuwro_pbfe.truth.parquet --volume full --run-offset 910000 > $L/convert_pbfe.log 2>&1
run "NuWro revertex"  $PY scripts/revertex_truth.py $O/nuwro_fullvol.truth.parquet data/nuwro_2026-10/nuwro_me_fhc_tracker.truth.parquet $O/nuwro_pbfe.truth.parquet --volume full > $L/revertex.log 2>&1
run "GENIE full ref"  $PY scripts/genie_tracker_truth.py $O/genie_fullvol.truth.parquet --volume full --n 2000000 > $L/genie_ref.log 2>&1
run "NuWro surrogate" $PY scripts/surrogate_to_ntuple.py reports/m3_1A_z $O/nuwro_fullvol.truth.parquet $O/surrogate_Az_nuwro_fullvol.root --epi-draws 8 > $L/nuwro_sur.log 2>&1
run "GENIE surrogate" $PY scripts/surrogate_to_ntuple.py reports/m3_1A_z $O/genie_fullvol.truth.parquet $O/surrogate_Az_genie_fullvol.root --epi-draws 8 > $L/genie_sur.log 2>&1
echo "$(stamp) OUTFV INPUTS DONE"
