#!/usr/bin/env bash
# Paper and technote products from the all-events models (A = reports/m3_1A_all, B = reports/m3_1A_all_no2p2h) evaluated on
# CC nu_mu events, replacing the CC-only models: calibrations (test and training controls, CC nu_mu control subset),
# positive control, pull subsets, paper evaluation with the open-dataset label, response/hadron/pull figures, macros.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_paper_all; A=reports/m3_1A_all; B=reports/m3_1A_all_no2p2h; CAL=reports/bayes_1A_all; mkdir -p $L reports/paper_figs; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
[ -f $CAL/bayes_uncertainty_B_ccctl.json ] || run "cal B test (CC control)" $PY scripts/bayes_uncertainty_eval.py $CAL --model $B --tag _B_ccctl --draws 8 --n-control 300000 --control-subset ccnumu > $L/cal_B.log 2>&1
[ -f ${CAL}_trainctl/bayes_uncertainty_A_ccctl.json ] || run "cal A train (CC control)" $PY scripts/bayes_uncertainty_eval.py ${CAL}_trainctl --model $A --tag _A_ccctl --draws 8 --n-control 300000 --control-split 0 --control-subset ccnumu > $L/cal_A_train.log 2>&1
[ -f ${CAL}_trainctl/bayes_uncertainty_B_ccctl.json ] || run "cal B train (CC control)" $PY scripts/bayes_uncertainty_eval.py ${CAL}_trainctl --model $B --tag _B_ccctl --draws 8 --n-control 300000 --control-split 0 --control-subset ccnumu > $L/cal_B_train.log 2>&1
mkdir -p reports/m3_1A_all_paper && ln -sfn ../m3_1A_all/model.pt reports/m3_1A_all_paper/model.pt
run "paper eval (CC nu_mu, open-dataset label)" env SIM2RECO_REAL_LABEL="open dataset" $PY -u scripts/run_m3.py reports/m3_1A_all_paper --eval-only --steps 64 --subset ccnumu --m1 none --from-cache $A/samples_test.npz > $L/eval_paper.log 2>&1
run "paper figures: full" $PY scripts/paper_fig_response.py reports/paper_figs/full --models A=$A --subset ccnumu > $L/paperfig_full.log 2>&1
run "paper figures: 2p2h" $PY scripts/paper_fig_response.py reports/paper_figs/twop --models A=$A B=$B --only-inttype 8 --subset ccnumu > $L/paperfig_twop.log 2>&1
run "paper pulls" bash -c "$PY scripts/paper_fig_pulls.py $CAL reports/paper_figs/pulls_epistemic.png --tag _ccctl && $PY scripts/paper_fig_pulls.py $CAL reports/paper_figs/pulls_epistemic_2p2h.png --models A B --split heldout --tag _ccctl" > $L/paper_pulls.log 2>&1
run "closure macros" $PY scripts/paper_closure_macros.py > $L/closure_macros.log 2>&1
echo "$(stamp) PIPELINE PAPER ALL DONE" >> $L/pipeline.log
