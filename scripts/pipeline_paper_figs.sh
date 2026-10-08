#!/usr/bin/env bash
# Paper response figures (muon, recoil, hadron) for the full test sample (model A) and the held-out 2p2h events (A, B).
cd "$(dirname "$0")/.."
PY=.venv/bin/python; L=reports/logs_zero; mkdir -p reports/paper_figs; stamp() { date +%H:%M:%S; }
echo "$(stamp) start full"; $PY scripts/paper_fig_response.py reports/paper_figs/full --models A=reports/m3_1A_z > $L/paperfig_full.log 2>&1 || { echo "$(stamp) FAILED full"; exit 1; }
echo "$(stamp) start twop"; $PY scripts/paper_fig_response.py reports/paper_figs/twop --models A=reports/m3_1A_z B=reports/m3_1A_z_no2p2h --only-inttype 8 > $L/paperfig_twop.log 2>&1 || { echo "$(stamp) FAILED twop"; exit 1; }
echo "$(stamp) PAPER FIGS DONE"
