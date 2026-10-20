#!/usr/bin/env bash
# The three inclusive samples through the four-member ensemble of the all-events model (nominal = mixture, epistemic
# columns = between-member spread); the comparisons are rerun on the ensemble files. Uploads are done from the session.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_ensemble; A=reports/m3_1A_all; M="reports/m3_1A_all_s1 reports/m3_1A_all_s2 reports/m3_1A_all_s3"; I=data/inclusive; G=data/inclusive_gibuu; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/infer.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/infer.log; exit 1; }; echo "$(stamp) done" >> $L/infer.log; }
run "ensemble GENIE" $PY scripts/surrogate_to_ntuple.py $A $I/genie_inclusive.truth.parquet $I/surrogate_Eall_genie_inclusive.root --ensemble $M > $L/infer_genie.log 2>&1
run "ensemble GiBUU" $PY scripts/surrogate_to_ntuple.py $A $G/gibuu_inclusive.truth.parquet $G/surrogate_Eall_gibuu_inclusive.root --ensemble $M > $L/infer_gibuu.log 2>&1
run "ensemble NuWro" $PY scripts/surrogate_to_ntuple.py $A $I/nuwro_inclusive.truth.parquet $I/surrogate_Eall_nuwro_inclusive.root --ensemble $M > $L/infer_nuwro.log 2>&1
run "compare GiBUU (ensemble)" $PY scripts/fullvolume_compare.py --genie-truth $I/genie_inclusive.truth.parquet --genie-sur $I/surrogate_Eall_genie_inclusive.root --nuwro-truth $G/gibuu_inclusive.truth.parquet --nuwro-sur $G/surrogate_Eall_gibuu_inclusive.root --out reports/inclusive_gibuu_ens --macro-prefix nEnsGB --gen-label GiBUU > $L/infer_cmp_gibuu.log 2>&1
run "compare NuWro (ensemble)" $PY scripts/fullvolume_compare.py --genie-truth $I/genie_inclusive.truth.parquet --genie-sur $I/surrogate_Eall_genie_inclusive.root --nuwro-truth $I/nuwro_inclusive.truth.parquet --nuwro-sur $I/surrogate_Eall_nuwro_inclusive.root --out reports/inclusive_ens --macro-prefix nEnsOFV > $L/infer_cmp_nuwro.log 2>&1
echo "$(stamp) ENSEMBLE INFERENCE DONE" >> $L/infer.log
