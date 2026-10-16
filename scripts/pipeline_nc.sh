#!/usr/bin/env bash
# NC generator comparison through the all-events model A: NuWro NC (tracker nuclei; Pb + Fe) against the open
# dataset's NC nu_mu events (GENIE) with their own reconstruction. Fiducial tracker comparison, then the full
# training volume (tracker nuclei + Pb + Fe, revertexed) as in pipeline_fullvolume.sh. NC vertices and nucleus shares
# come from the open dataset's NC nu_mu events. Then the CC full-volume samples of pipeline_fullvolume.sh through the
# all-events model, for the all-events full-volume comparison.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_nc; O=data/fullvolume; A=reports/m3_1A_all; mkdir -p $L $O; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
NW=data/nuwro_nc_2026-10; GT=data/genie_tracker
run "NuWro NC convert"     $PY scripts/nuwro_to_truth.py $NW/jobs $NW/nuwro_me_fhc_tracker_nc.truth.parquet --run-offset 920000 --pool nc > $L/convert.log 2>&1
[ -f $GT/genie_me_fhc_tracker_nc.truth.parquet ] || run "GENIE NC tracker ref" $PY scripts/genie_tracker_truth.py $GT/genie_me_fhc_tracker_nc.truth.parquet --population ncnumu > $L/genie_ref.log 2>&1
run "NuWro NC surrogate"   $PY scripts/surrogate_to_ntuple.py $A $NW/nuwro_me_fhc_tracker_nc.truth.parquet $NW/surrogate_Aall_nuwro_me_fhc_tracker_nc.root --epi-draws 8 > $L/nuwro_sur.log 2>&1
[ -f $GT/surrogate_Aall_genie_me_fhc_tracker_nc.root ] || run "GENIE NC surrogate"   $PY scripts/surrogate_to_ntuple.py $A $GT/genie_me_fhc_tracker_nc.truth.parquet $GT/surrogate_Aall_genie_me_fhc_tracker_nc.root --epi-draws 8 > $L/genie_sur.log 2>&1
run "NC compare"           $PY scripts/nuwro_compare.py --genie-truth $GT/genie_me_fhc_tracker_nc.truth.parquet --genie-sur $GT/surrogate_Aall_genie_me_fhc_tracker_nc.root --nuwro-truth $NW/nuwro_me_fhc_tracker_nc.truth.parquet --nuwro-sur $NW/surrogate_Aall_nuwro_me_fhc_tracker_nc.root --out reports/nuwro_compare_nc --nc --macro-prefix nNC > $L/compare.log 2>&1
run "NuWro NC Pb/Fe convert" $PY scripts/nuwro_to_truth.py data/nuwro_pbfe_nc_2026-10/jobs $O/nuwro_pbfe_nc.truth.parquet --volume full --run-offset 930000 --pool nc > $L/convert_pbfe.log 2>&1
run "NuWro NC revertex"    $PY scripts/revertex_truth.py $O/nuwro_fullvol_nc.truth.parquet $NW/nuwro_me_fhc_tracker_nc.truth.parquet $O/nuwro_pbfe_nc.truth.parquet --volume full --pool nc > $L/revertex.log 2>&1
[ -f $O/genie_fullvol_nc.truth.parquet ] || run "GENIE NC full ref"    $PY scripts/genie_tracker_truth.py $O/genie_fullvol_nc.truth.parquet --volume full --population ncnumu > $L/genie_full_ref.log 2>&1
run "NuWro NC full surrogate" $PY scripts/surrogate_to_ntuple.py $A $O/nuwro_fullvol_nc.truth.parquet $O/surrogate_Aall_nuwro_fullvol_nc.root --epi-draws 8 > $L/nuwro_full_sur.log 2>&1
run "GENIE NC full surrogate" $PY scripts/surrogate_to_ntuple.py $A $O/genie_fullvol_nc.truth.parquet $O/surrogate_Aall_genie_fullvol_nc.root --epi-draws 8 > $L/genie_full_sur.log 2>&1
run "NC full compare"      $PY scripts/fullvolume_compare.py --genie-truth $O/genie_fullvol_nc.truth.parquet --genie-sur $O/surrogate_Aall_genie_fullvol_nc.root --nuwro-truth $O/nuwro_fullvol_nc.truth.parquet --nuwro-sur $O/surrogate_Aall_nuwro_fullvol_nc.root --out reports/outfv_nc --macro-prefix nNCOFV > $L/full_compare.log 2>&1
run "NuWro CC full surrogate (all-events A)" $PY scripts/surrogate_to_ntuple.py $A $O/nuwro_fullvol.truth.parquet $O/surrogate_Aall_nuwro_fullvol.root --epi-draws 8 > $L/nuwro_cc_full_sur.log 2>&1
run "GENIE CC full surrogate (all-events A)" $PY scripts/surrogate_to_ntuple.py $A $O/genie_fullvol.truth.parquet $O/surrogate_Aall_genie_fullvol.root --epi-draws 8 > $L/genie_cc_full_sur.log 2>&1
run "CC full compare (all-events A)" $PY scripts/fullvolume_compare.py --genie-truth $O/genie_fullvol.truth.parquet --genie-sur $O/surrogate_Aall_genie_fullvol.root --nuwro-truth $O/nuwro_fullvol.truth.parquet --nuwro-sur $O/surrogate_Aall_nuwro_fullvol.root --out reports/outfv_all --macro-prefix nAllOFV > $L/cc_full_compare.log 2>&1
echo "$(stamp) PIPELINE NC DONE" >> $L/pipeline.log
