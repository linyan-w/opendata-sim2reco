#!/usr/bin/env bash
# All-events models (every interaction in the training volume: CC and NC, all flavours; 10 MeV + neutrons, zero flags,
# length-bucketed batches): M2 -> M3 for A (full) and B (2p2h blind), Bayesian last layers; evaluation on the full test
# split and on its CC nu_mu / NC / other subsets; 2p2h holdout (A and B on 2p2h, B on the CC nu_mu non-2p2h control,
# comparison and bootstrap uncertainties); calibration of A; NuWro and the paired GENIE sample through A.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_all; mkdir -p $L; stamp() { date +%H:%M:%S; }
run() { echo "$(stamp) start $1"; shift; "$@" || { echo "$(stamp) FAILED"; exit 1; }; echo "$(stamp) done"; }
A=reports/m3_1A_all; B=reports/m3_1A_all_no2p2h
link() { mkdir -p reports/$2 && ln -sfn ../$1/model.pt reports/$2/model.pt; }
run "M2 A"  $PY -u scripts/run_m2.py reports/m2_1A_all --epochs 30 --steps 64 --population all --m1 none > $L/m2_A.log 2>&1
run "M3 A"  $PY -u scripts/run_m3.py $A --init reports/m2_1A_all/model.pt --epochs 16 --lr 2e-4 --steps 64 --m1 none > $L/m3_A.log 2>&1
run "bayes fit A" $PY scripts/fit_bayes_last.py $A > $L/bayes_fit_A.log 2>&1
for S in ccnumu nc other; do link m3_1A_all m3_1A_all_$S
  run "A on $S" $PY -u scripts/run_m3.py reports/m3_1A_all_$S --eval-only --steps 64 --subset $S --m1 none > $L/eval_A_$S.log 2>&1; done
run "NuWro A"  $PY scripts/surrogate_to_ntuple.py $A data/nuwro_2026-10/nuwro_me_fhc_tracker.truth.parquet data/nuwro_2026-10/surrogate_Aall_nuwro_me_fhc_tracker.root --epi-draws 8 > $L/nuwro_A.log 2>&1
run "GENIE A"  $PY scripts/surrogate_to_ntuple.py $A data/genie_tracker/genie_me_fhc_tracker.truth.parquet data/genie_tracker/surrogate_Aall_genie_me_fhc_tracker.root --epi-draws 8 > $L/genie_A.log 2>&1
run "NuWro cmp" $PY scripts/nuwro_compare.py --genie-sur data/genie_tracker/surrogate_Aall_genie_me_fhc_tracker.root --nuwro-sur data/nuwro_2026-10/surrogate_Aall_nuwro_me_fhc_tracker.root --out reports/nuwro_compare_all > $L/nuwro_cmp.log 2>&1
run "M2 B"  $PY -u scripts/run_m2.py reports/m2_1A_all_no2p2h --epochs 30 --steps 64 --population all --exclude-inttype 8 --m1 none > $L/m2_B.log 2>&1
run "M3 B"  $PY -u scripts/run_m3.py $B --init reports/m2_1A_all_no2p2h/model.pt --epochs 16 --lr 2e-4 --steps 64 --exclude-inttype 8 --m1 none > $L/m3_B.log 2>&1
run "bayes fit B" $PY scripts/fit_bayes_last.py $B > $L/bayes_fit_B.log 2>&1
link m3_1A_all m3_1A_all_on2p2h; link m3_1A_all_no2p2h m3_1A_all_no2p2h_on2p2h; link m3_1A_all_no2p2h m3_1A_all_no2p2h_control
run "A on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_all_on2p2h --eval-only --steps 64 --only-inttype 8 --m1 none > $L/eval_A_on2p2h.log 2>&1
run "B on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_all_no2p2h_on2p2h --eval-only --steps 64 --only-inttype 8 --m1 none > $L/eval_B_on2p2h.log 2>&1
run "B control"   $PY -u scripts/run_m3.py reports/m3_1A_all_no2p2h_control --eval-only --steps 64 --only-inttype 1 2 3 4 6 --subset ccnumu --m1 none > $L/eval_B_control.log 2>&1
run "holdout cmp" $PY scripts/holdout_compare.py reports/holdout_2p2h_1A_all --a reports/m3_1A_all_on2p2h --b reports/m3_1A_all_no2p2h_on2p2h --b-control reports/m3_1A_all_no2p2h_control --a-full $A --population all > $L/holdout.log 2>&1
run "holdout unc" $PY scripts/holdout_uncertainty.py reports/holdout_2p2h_1A_all --a $A --b $B > $L/holdout_unc.log 2>&1
run "cal A test"  $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_all --model $A --tag _A --draws 8 --n-control 300000 > $L/cal_A.log 2>&1
echo "$(stamp) PIPELINE ALL DONE"
