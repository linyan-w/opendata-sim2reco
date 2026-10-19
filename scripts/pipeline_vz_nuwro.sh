#!/usr/bin/env bash
# After scripts/pipeline_vz_adopt.sh: the inclusive NuWro sample through the adopted all-events model, its copy to the
# share, the inclusive comparison and the technote tables. The non-inclusive samples (tracker, full volume, NC tracker)
# are not regenerated: their reports document the model before the vertex-head fix. Run as a systemd unit when the GPU is free.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_vz_adopt; mkdir -p $L; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/pipeline_nuwro.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline_nuwro.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline_nuwro.log; }
try() { echo "$(stamp) start $1" >> $L/pipeline_nuwro.log; shift; "$@" && echo "$(stamp) done" >> $L/pipeline_nuwro.log || echo "$(stamp) FAILED (continuing)" >> $L/pipeline_nuwro.log; }
share() { try "share $(basename $1)" bash -c "ssh -o BatchMode=yes $GPVM 'rm -f $SHARE/$(basename $1)' && scp -q $1 $GPVM:$SHARE/"; }
SHARE=/pnfs/dune/persistent/users/lwan/Share/NDP; GPVM=lwan@dunegpvm08.fnal.gov
Z=reports/m3_1A_z; A=reports/m3_1A_all; I=data/inclusive; O=data/fullvolume; NW=data/nuwro_2026-10; NC=data/nuwro_nc_2026-10; GT=data/genie_tracker
run "inclusive NuWro surrogate" $PY scripts/surrogate_to_ntuple.py $A $I/nuwro_inclusive.truth.parquet $I/surrogate_Aall_nuwro_inclusive.root --epi-draws 8 > $L/inc_nuwro.log 2>&1
share $I/surrogate_Aall_nuwro_inclusive.root
run "inclusive compare NuWro" $PY scripts/fullvolume_compare.py --genie-truth $I/genie_inclusive.truth.parquet --genie-sur $I/surrogate_Aall_genie_inclusive.root --nuwro-truth $I/nuwro_inclusive.truth.parquet --nuwro-sur $I/surrogate_Aall_nuwro_inclusive.root --out reports/inclusive --macro-prefix nIncOFV > $L/inc_cmp_nuwro.log 2>&1
run "inclusive tables" $PY scripts/technote_inclusive.py > $L/inc_tables.log 2>&1
echo "$(stamp) PIPELINE VZ NUWRO DONE" >> $L/pipeline_nuwro.log
