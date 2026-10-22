#!/usr/bin/env bash
# The inclusive NuWro sample through the four-member ensemble (universes), and its comparison. Upload from the session.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_ensemble; A=reports/m3_1A_all; M="reports/m3_1A_all_s1 reports/m3_1A_all_s2 reports/m3_1A_all_s3"; I=data/inclusive; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/infer.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/infer.log; exit 1; }; echo "$(stamp) done" >> $L/infer.log; }
run "ensemble NuWro" $PY scripts/surrogate_to_ntuple.py $A $I/nuwro_inclusive.truth.parquet $I/surrogate_Eall_nuwro_inclusive.root --ensemble $M > $L/infer_nuwro.log 2>&1
run "compare NuWro (ensemble)" $PY scripts/fullvolume_compare.py --genie-truth $I/genie_inclusive.truth.parquet --genie-sur $I/surrogate_Eall_genie_inclusive.root --nuwro-truth $I/nuwro_inclusive.truth.parquet --nuwro-sur $I/surrogate_Eall_nuwro_inclusive.root --out reports/inclusive_ens --macro-prefix nEnsOFV > $L/infer_cmp_nuwro.log 2>&1
echo "$(stamp) ENSEMBLE NUWRO DONE" >> $L/infer.log
