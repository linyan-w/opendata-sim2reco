#!/usr/bin/env bash
# Paper figures on the full test sample (all interactions) instead of CC nu_mu: response and hadron figures with the
# ensemble band, the 2p2h holdout figures (all 2p2h test events), and the ensemble spread study for the pulls figure.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_paper_all; A=reports/m3_1A_all; B=reports/m3_1A_all_no2p2h; ENS="--ensemble A=reports/m3_1A_all_s1 A=reports/m3_1A_all_s2 A=reports/m3_1A_all_s3"; stamp() { date "+%F %T"; }
run() { echo "$(stamp) start $1" >> $L/pipeline.log; shift; "$@" || { echo "$(stamp) FAILED" >> $L/pipeline.log; exit 1; }; echo "$(stamp) done" >> $L/pipeline.log; }
run "full-test figures" $PY scripts/paper_fig_response.py reports/paper_figs/full --models A=$A $ENS > $L/paperfig_full_all.log 2>&1
run "2p2h figures (all 2p2h)" $PY scripts/paper_fig_response.py reports/paper_figs/twop --models A=$A B=$B --only-inttype 8 $ENS > $L/paperfig_twop_all.log 2>&1
run "ensemble spread, full test" $PY scripts/ensemble_spread.py --members $A reports/m3_1A_all_s1 reports/m3_1A_all_s2 reports/m3_1A_all_s3 --subset "" --out reports/ensemble_full > $L/spread_full.log 2>&1
echo "$(stamp) PIPELINE PAPER FULLTEST DONE" >> $L/pipeline.log
