#!/usr/bin/env python
"""Run the M1 baselines. Usage: scripts/run_m1.py data/slim/<stem> reports/m1 [--epochs N] [--cpu]"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim2reco.train.baselines import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem"); ap.add_argument("out_dir"); ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    run(a.stem, a.out_dir, device="cpu" if a.cpu else "cuda", epochs=a.epochs)
