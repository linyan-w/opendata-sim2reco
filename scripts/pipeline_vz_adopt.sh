#!/usr/bin/env bash
# Adopt the z-conditioned vertex-plane head (vtx_cond_z=far, scripts/pipeline_vz.sh) in the four adopted models and
# regenerate what depends on it. Order: the A models are adopted at once and the inclusive GENIE and GiBUU files are
# regenerated and put on the share first; then the head stage of the 2p2h-blind models;
# then the evaluation chains of the CC models (paper, technote) and of the all-events models (technote). The NuWro
# files, their GENIE partners (full volume, tracker, NC) and the generator comparisons are left to
# scripts/pipeline_vz_nuwro.sh (run when the GPU is free); they feed the technote only.
# Run as a systemd unit with a memory cap. Previous model weights are kept in reports/<model>_prevz/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_vz_adopt; mkdir -p $L; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
try() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" && echo "$(stamp) done" >> $L/pipeline.log || echo "$(stamp) FAILED (continuing)" >> $L/pipeline.log; }  # the copies to the share may fail on an expired ticket
link() { mkdir -p reports/$2 && ln -sfn ../$1/model.pt reports/$2/model.pt; }
adopt() { mkdir -p reports/$1_prevz; [ -f reports/$1_prevz/model.pt ] || cp reports/$1/model.pt reports/$1_prevz/model.pt; cp reports/$1_vz_far/model.pt reports/$1/model.pt; cp reports/$1_vz_far/history.json reports/$1/history_vz.json; echo "$(stamp) adopted vtx_cond_z=far weights in reports/$1" >> $L/pipeline.log; }
share() { try "share $(basename $1)" bash -c "ssh -o BatchMode=yes $GPVM 'rm -f $SHARE/$(basename $1)' && scp -q $1 $GPVM:$SHARE/"; }
SHARE=/pnfs/dune/persistent/users/lwan/Share/NDP; GPVM=lwan@dunegpvm08.fnal.gov
Z=reports/m3_1A_z; ZB=reports/m3_1A_z_no2p2h; A=reports/m3_1A_all; AB=reports/m3_1A_all_no2p2h
I=data/inclusive; G=data/inclusive_gibuu; O=data/fullvolume; NW=data/nuwro_2026-10; NC=data/nuwro_nc_2026-10; GT=data/genie_tracker
# ---- 1. adopt the A models (head stage done by pipeline_vz.sh) and regenerate the inclusive GENIE and GiBUU files
adopt m3_1A_all; adopt m3_1A_z
[ -f $I/surrogate_Aall_genie_inclusive.root ] || run "inclusive GENIE surrogate" $PY scripts/surrogate_to_ntuple.py $A $I/genie_inclusive.truth.parquet $I/surrogate_Aall_genie_inclusive.root --epi-draws 8 > $L/inc_genie.log 2>&1
# uploads are done from the session (the unit has no Kerberos ticket)
[ -f $G/surrogate_Aall_gibuu_inclusive.root ] || run "inclusive GiBUU surrogate" $PY scripts/surrogate_to_ntuple.py $A $G/gibuu_inclusive.truth.parquet $G/surrogate_Aall_gibuu_inclusive.root --epi-draws 8 > $L/inc_gibuu.log 2>&1
[ -f reports/inclusive_gibuu/summary.json ] && [ reports/inclusive_gibuu/summary.json -nt $G/surrogate_Aall_gibuu_inclusive.root ] || run "inclusive compare GiBUU" $PY scripts/fullvolume_compare.py --genie-truth $I/genie_inclusive.truth.parquet --genie-sur $I/surrogate_Aall_genie_inclusive.root --nuwro-truth $G/gibuu_inclusive.truth.parquet --nuwro-sur $G/surrogate_Aall_gibuu_inclusive.root --out reports/inclusive_gibuu --macro-prefix nIncGB --gen-label GiBUU > $L/inc_cmp_gibuu.log 2>&1
[ -f reports/vtx_tail/summary.json ] && [ reports/vtx_tail/summary.json -nt $I/surrogate_Aall_genie_inclusive.root ] || run "vertex tail diag" $PY scripts/vtx_tail_diag.py > $L/vtx_tail.log 2>&1
echo "$(stamp) INCLUSIVE GENIE/GIBUU DONE" >> $L/pipeline.log
# ---- 2. head stage for the 2p2h-blind models, adopt
[ -f ${AB}_vz_far/model.pt ] || run "head stage all-events B" env VZ_MODE=_far VZ_MODELS="m3_1A_all_no2p2h" scripts/pipeline_vz.sh
[ -f ${ZB}_vz_far/model.pt ] || run "head stage CC B" env VZ_MODE=_far VZ_MODELS="m3_1A_z_no2p2h" scripts/pipeline_vz.sh
adopt m3_1A_all_no2p2h; adopt m3_1A_z_no2p2h
# ---- 3. CC models: evaluations, holdout, calibration, closure subsets, paper figures, numbers
for M in m3_1A_z m3_1A_z_no2p2h; do link $M ${M}_on2p2h; done; link m3_1A_z_no2p2h m3_1A_z_no2p2h_control; link m3_1A_z m3_1A_z_paper
run "eval A full"   $PY -u scripts/run_m3.py $Z --eval-only --steps 64 --cache $Z/samples_test.npz > $L/eval_A.log 2>&1
run "eval B full"   $PY -u scripts/run_m3.py $ZB --eval-only --steps 64 --cache $ZB/samples_test.npz > $L/eval_B.log 2>&1
run "A on 2p2h"     $PY -u scripts/run_m3.py ${Z}_on2p2h --eval-only --steps 64 --only-inttype 8 --from-cache $Z/samples_test.npz > $L/eval_A_on2p2h.log 2>&1
run "B on 2p2h"     $PY -u scripts/run_m3.py ${ZB}_on2p2h --eval-only --steps 64 --only-inttype 8 --from-cache $ZB/samples_test.npz > $L/eval_B_on2p2h.log 2>&1
run "B control"     $PY -u scripts/run_m3.py ${ZB}_control --eval-only --steps 64 --only-inttype 1 2 3 4 --from-cache $ZB/samples_test.npz > $L/eval_B_control.log 2>&1
run "holdout cmp"   $PY scripts/holdout_compare.py reports/holdout_2p2h_1A_z --a ${Z}_on2p2h --b ${ZB}_on2p2h --b-control ${ZB}_control --a-full $Z > $L/holdout.log 2>&1
run "holdout unc"   $PY scripts/holdout_uncertainty.py reports/holdout_2p2h_1A_z --a $Z --b $ZB > $L/holdout_unc.log 2>&1
run "cal B test"    $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z --model $ZB --tag _B --draws 8 --n-control 300000 > $L/cal_B.log 2>&1
run "cal A test"    $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z --model $Z --tag _A --draws 8 --n-control 300000 > $L/cal_A.log 2>&1
run "cal B train"   $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z_trainctl --model $ZB --tag _B --draws 8 --n-control 300000 --control-split 0 > $L/cal_B_train.log 2>&1
run "cal A train"   $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_z_trainctl --model $Z --tag _A --draws 8 --n-control 300000 --control-split 0 > $L/cal_A_train.log 2>&1
run "cal figures"   $PY scripts/bayes_binned_figures.py reports/bayes_1A_z --draws 8 > $L/cal_figures.log 2>&1
run "floor"         $PY scripts/bayes_floor_transfer.py reports/bayes_1A_z reports/bayes_1A_z_trainctl > $L/floor.log 2>&1
run "pos control"   $PY scripts/bayes_positive_control.py reports/bayes_1A_z --model $ZB --tag _B > $L/pc.log 2>&1
run "pull subsets"  bash -c "$PY scripts/bayes_pulls_subset.py reports/bayes_1A_z reports/bayes_1A_z_trainctl --exclude blobs --tag NB && $PY scripts/bayes_pulls_subset.py reports/bayes_1A_z reports/bayes_1A_z_trainctl --exclude blobs nonvtx100 --tag CI" > $L/pull_subsets.log 2>&1
run "closure subset" $PY scripts/closure_subset.py --a $Z --b $ZB --out $Z > $L/closure_subset.log 2>&1
run "paper eval"    env SIM2RECO_REAL_LABEL="open dataset" $PY -u scripts/run_m3.py ${Z}_paper --eval-only --steps 64 --from-cache $Z/samples_test.npz > $L/eval_A_paper.log 2>&1
run "symmetry"      $PY scripts/confusion_symmetry.py $Z > $L/symmetry.log 2>&1
run "numbers"       bash -c "$PY scripts/technote_numbers.py $Z $Z/numbers.tex --m1 reports/m1_1A/metrics.json --holdout reports/holdout_2p2h_1A_z --bayes reports/bayes_1A_z --bayes-train reports/bayes_1A_z_trainctl --floor reports/bayes_1A_z/floor_transfer.json $( [ -f reports/variants_1A/summary.json ] && echo --variants reports/variants_1A/summary.json ) $( [ -f reports/holdout_2p2h_1A/summary.json ] && echo --holdout-old reports/holdout_2p2h_1A )" > $L/numbers.log 2>&1
run "paper figures" scripts/pipeline_paper_figs.sh > $L/paper_figs.log 2>&1
run "paper pulls"   bash -c "$PY scripts/paper_fig_pulls.py reports/bayes_1A_z reports/paper_figs/pulls_epistemic.png && $PY scripts/paper_fig_pulls.py reports/bayes_1A_z reports/paper_figs/pulls_epistemic_2p2h.png --models A B --split heldout" > $L/paper_pulls.log 2>&1
echo "$(stamp) CC EVALUATIONS DONE" >> $L/pipeline.log
# ---- 4. all-events models: evaluations by subset, holdout, calibration, closure subsets, technote tables
run "all A full"      $PY -u scripts/run_m3.py $A --eval-only --steps 64 --m1 none --cache $A/samples_test.npz > $L/all_eval_A.log 2>&1
run "all B full"      $PY -u scripts/run_m3.py $AB --eval-only --steps 64 --m1 none --cache $AB/samples_test.npz > $L/all_eval_B.log 2>&1
for S in ccnumu nc other; do link m3_1A_all m3_1A_all_$S; run "A on $S" $PY -u scripts/run_m3.py reports/m3_1A_all_$S --eval-only --steps 64 --subset $S --m1 none --from-cache $A/samples_test.npz > $L/all_eval_A_$S.log 2>&1; done
link m3_1A_all m3_1A_all_on2p2h; link m3_1A_all_no2p2h m3_1A_all_no2p2h_on2p2h; link m3_1A_all_no2p2h m3_1A_all_no2p2h_control
run "all A on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_all_on2p2h --eval-only --steps 64 --only-inttype 8 --m1 none --from-cache $A/samples_test.npz > $L/all_eval_A_on2p2h.log 2>&1
run "all B on 2p2h"   $PY -u scripts/run_m3.py reports/m3_1A_all_no2p2h_on2p2h --eval-only --steps 64 --only-inttype 8 --m1 none --from-cache $AB/samples_test.npz > $L/all_eval_B_on2p2h.log 2>&1
run "all B control"   $PY -u scripts/run_m3.py reports/m3_1A_all_no2p2h_control --eval-only --steps 64 --only-inttype 1 2 3 4 6 --subset ccnumu --m1 none --from-cache $AB/samples_test.npz > $L/all_eval_B_control.log 2>&1
run "all holdout cmp" $PY scripts/holdout_compare.py reports/holdout_2p2h_1A_all --a reports/m3_1A_all_on2p2h --b reports/m3_1A_all_no2p2h_on2p2h --b-control reports/m3_1A_all_no2p2h_control --a-full $A --population all > $L/all_holdout.log 2>&1
run "all holdout unc" $PY scripts/holdout_uncertainty.py reports/holdout_2p2h_1A_all --a $A --b $AB > $L/all_holdout_unc.log 2>&1
run "all cal A"       $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_all --model $A --tag _A --draws 8 --n-control 300000 > $L/all_cal_A.log 2>&1
run "all cal A ccctl" $PY scripts/bayes_uncertainty_eval.py reports/bayes_1A_all --model $A --tag _A_ccctl --draws 8 --n-control 300000 --control-subset ccnumu > $L/all_cal_A_ccctl.log 2>&1
run "all closure"     $PY scripts/closure_subset_all.py --a $A --b $AB --out reports/all_events/closure.json > $L/all_closure.log 2>&1
run "all technote"    $PY scripts/technote_all.py > $L/all_technote.log 2>&1
echo "$(stamp) PIPELINE VZ ADOPT DONE" >> $L/pipeline.log
