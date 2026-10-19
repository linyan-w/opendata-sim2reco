#!/usr/bin/env bash
# Technote products of the all-events models evaluated on CC nu_mu events, replacing the CC-only models' directories:
# calibration directories with the CC nu_mu control (reports/bayes_1A_all_cc, _trainctl), binned figures, floor transfer,
# positive control, pull subsets, multiplicity-symmetry figure, technote numbers of the CC nu_mu subset evaluation.
# Needs scripts/pipeline_paper_all.sh done (calibration JSONs with the _ccctl tag).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_paper_all; A=reports/m3_1A_all; B=reports/m3_1A_all_no2p2h; E=reports/m3_1A_all_ccnumu; C=reports/bayes_1A_all_cc; mkdir -p $C/figures $C/tables ${C}_trainctl; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
for m in A B; do cp reports/bayes_1A_all/bayes_uncertainty_${m}_ccctl.json $C/bayes_uncertainty_$m.json; cp reports/bayes_1A_all_trainctl/bayes_uncertainty_${m}_ccctl.json ${C}_trainctl/bayes_uncertainty_$m.json; cp reports/bayes_1A_all/figures/bayes_flags_${m}_ccctl.png $C/figures/bayes_flags_$m.png; done
sed -i 's/CC~$\\nu_\\mu$ events with/events (all interactions) with/' reports/m3_1A_all/tables/data.tex
run "binned figures"  $PY scripts/bayes_binned_figures.py $C --draws 8 > $L/binned.log 2>&1
run "bins table"      $PY scripts/bayes_bins_table.py $C > $L/bins.log 2>&1
run "floor"           $PY scripts/bayes_floor_transfer.py $C ${C}_trainctl > $L/floor.log 2>&1
run "positive control" $PY scripts/bayes_positive_control.py $C --model $B --tag _B > $L/pc.log 2>&1
run "pull subsets"    bash -c "$PY scripts/bayes_pulls_subset.py $C ${C}_trainctl --exclude blobs --tag NB && $PY scripts/bayes_pulls_subset.py $C ${C}_trainctl --exclude blobs nonvtx100 --tag CI" > $L/pull_subsets.log 2>&1
run "symmetry"        $PY scripts/confusion_symmetry.py $E > $L/symmetry.log 2>&1
run "closure tables"  $PY scripts/paper_closure_macros.py > $L/closure_macros.log 2>&1
run "numbers"         $PY scripts/technote_numbers.py $E $E/numbers.tex --m1 none --holdout reports/holdout_2p2h_1A_all --bayes $C --bayes-train ${C}_trainctl --floor $C/floor_transfer.json > $L/numbers.log 2>&1
run "paper macros"    $PY scripts/paper_macros_all.py > $L/paper_macros.log 2>&1
echo "$(stamp) PIPELINE TECHNOTE ALL DONE" >> $L/pipeline.log
