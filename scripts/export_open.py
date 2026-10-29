#!/usr/bin/env python
"""Export the open-release tree (XSuM-MINERvA) from this repository: the package without the Bayesian last layer and the
baselines, the curation / training / inference / evaluation scripts with their new names and layout, the paper figure
scripts with their small inputs, the five model checkpoints, the docs and the tests with a small example file.
Rerunnable: the destination is emptied (its .git, if any, is kept) and rebuilt from the current tree.
Usage: scripts/export_open.py [DEST]   (default /home/lwan/AI/XSuM-MINERvA)"""
import pathlib, re, shutil, sys, json
SRC = pathlib.Path(__file__).resolve().parents[1]
DEST = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/lwan/AI/XSuM-MINERvA")

# ---------------------------------------------------------------- manifest: source -> destination (relative paths)
PACKAGE = {  # sim2reco/... -> xsum/...
    "constants.py": "constants.py", "__init__.py": "__init__.py",
    "io/__init__.py": "io/__init__.py", "io/branches.py": "io/branches.py", "io/slim.py": "io/slim.py", "io/writer.py": "io/writer.py", "io/deadtime.py": "io/deadtime.py",
    "prep/__init__.py": "prep/__init__.py", "prep/particles.py": "prep/particles.py", "prep/frames.py": "prep/frames.py", "prep/muon.py": "prep/muon.py",
    "prep/prongs.py": "prep/prongs.py", "prep/features.py": "prep/features.py", "prep/volumes.py": "prep/volumes.py",
    "data/__init__.py": "data/__init__.py", "data/dataset.py": "data/dataset.py", "data/compact.py": "data/compact.py", "data/prongs_tf.py": "data/prongs_tf.py",
    "models/__init__.py": "models/__init__.py", "models/encoder.py": "models/encoder.py", "models/flow.py": "models/flow.py", "models/prongflow.py": "models/prongflow.py", "models/surrogate.py": "models/surrogate.py",
    "train/__init__.py": "train/__init__.py", "train/samples.py": "train/samples.py", "train/m2.py": "train/event_stage.py", "train/m3.py": "train/prong_stage.py",
    "eval/__init__.py": "eval/__init__.py", "eval/metrics.py": "eval/metrics.py", "eval/plots.py": "eval/plots.py",
}
SCRIPTS = {  # scripts/... -> ...
    "slim_remote.py": "scripts/curate/slim_remote.py", "slim.py": "scripts/curate/slim.py", "dataset_summary.py": "scripts/curate/dataset_summary.py",
    "genie_tracker_truth.py": "scripts/curate/genie_truth.py", "nuwro_to_truth.py": "scripts/curate/nuwro_to_truth.py", "gibuu_to_truth.py": "scripts/curate/gibuu_to_truth.py",
    "deadtime_fetch.py": "scripts/curate/deadtime_fetch.py", "deadtime_study.py": "scripts/curate/deadtime_table.py",
    "run_m2.py": "scripts/train/train_event.py", "run_m3.py": "scripts/train/train_prong.py",
    "surrogate_to_ntuple.py": "scripts/infer/surrogate_to_ntuple.py", "conform_ntuple.py": "scripts/infer/conform_ntuple.py", "add_deadtime.py": "scripts/infer/add_deadtime.py",
    "closure_subset_all.py": "scripts/evaluate/closure_classifier.py", "ensemble_spread.py": "scripts/evaluate/ensemble_spread.py", "holdout_compare.py": "scripts/evaluate/holdout_compare.py",
    "render_schematic.sh": "scripts/evaluate/render_schematic.sh", "setup_env.sh": "scripts/setup_env.sh",
    "paper_fig_response.py": "paper/fig_response.py", "paper_fig_confusion.py": "paper/fig_confusion.py", "paper_fig_ensemble.py": "paper/fig_ensemble.py",
}
WEIGHTS = {"reports/m3_1A_all": "weights/full", "reports/m3_1A_all_s1": "weights/full_s1", "reports/m3_1A_all_s2": "weights/full_s2", "reports/m3_1A_all_s3": "weights/full_s3", "reports/m3_1A_all_no2p2h": "weights/holdout_2p2h"}
WEIGHT_FILES = ["model.pt", "metrics.json", "metrics_tier2.json", "history.json"]
PAPER_INPUTS = {"reports/all_events/closure.json": "paper/inputs/closure/closure.json",                 "reports/ensemble_full/spread.json": "paper/inputs/ensemble/spread.json", "reports/ensemble_full/pull_macros.tex": "paper/inputs/ensemble/pull_macros.tex",
                "reports/holdout_2p2h_1A_all/summary.json": "paper/inputs/holdout/summary.json",
                "PaperDraft/figures/model_schematic_paper.tex": "paper/schematic/model_schematic.tex", "PaperDraft/figures/model_schematic_paper.pdf": "paper/schematic/model_schematic.pdf"}
DOCS = {"docs/surrogate_files.md": "docs/surrogate_files.md", "docs/tuple_notes.md": "docs/tuple_notes.md",
        "docs/branches/MasterAnaDev_branches.txt": "docs/branches/MasterAnaDev_branches.txt", "docs/branches/Truth_branches.txt": "docs/branches/Truth_branches.txt",
        "docs/branches/Meta_branches.txt": "docs/branches/Meta_branches.txt", "docs/branches/MAD_tuple_MainDoc_upstream.csv": "docs/branches/MAD_tuple_MainDoc_upstream.csv",
        "configs/MediumEnergy_FHC_StandardMC_Playlist1A.txt": "configs/playlist1A_files.txt", "data/deadtime/deadtime_marginal.json": "data/deadtime_marginal.json"}
TESTS = ["conftest.py", "test_dataset.py", "test_m2.py", "test_m3.py", "test_muon_writer.py", "test_particles.py", "test_prongs.py"]

# ---------------------------------------------------------------- text rewrites applied to every copied text file
OLD_SCRIPT_PATHS = {f"scripts/{k}": v for k, v in SCRIPTS.items()}
OLD_SCRIPT_PATHS.update({"scripts/pipeline_ensemble.sh": "scripts/train/train_all.sh", "scripts/deadtime_study.py": "scripts/curate/deadtime_table.py"})
REWRITES = [
    ("from sim2reco.train import m2, m3", "from xsum.train import event_stage as m2, prong_stage as m3"),
    ("from sim2reco.train.m2 import", "from xsum.train.event_stage import"), ("from sim2reco.train.m3 import", "from xsum.train.prong_stage import"),
    ("from . import m2", "from . import event_stage as m2"), ("sim2reco.train.m3.evaluate_tier2", "xsum.train.prong_stage.evaluate_tier2"),
    ("sim2reco", "xsum"),
    ("reports/m3_1A_all_no2p2h", "weights/holdout_2p2h"), ("reports/m3_1A_all_s", "weights/full_s"), ("reports/m3_1A_all", "weights/full"),
    ("reports/ensemble_full/", "paper/inputs/ensemble/"), ("reports/ensemble/", "paper/inputs/ensemble/"), ("reports/all_events/", "paper/inputs/closure/"),
    ("reports/holdout_2p2h_1A_all", "paper/inputs/holdout"), ("reports/coverage_2p2h/macros.tex", "paper/inputs/coverage_macros.tex"), ("reports/paper_figs", "paper/figures"),
    ("data/deadtime/deadtime_marginal.json", "data/deadtime_marginal.json"),
    ('ap.add_argument("--m1", default="reports/m1_1A/metrics.json"); ', ""), ('ap.add_argument("--m1", default="reports/m1_1A/metrics.json")', ""),
    ('[--m1 reports/m1_1A/metrics.json] ', ""), ('m1 = json.load(open(a.m1)) if pathlib.Path(a.m1).exists() else None', "m1 = None"),
    ('ap.add_argument("--init", default="reports/m2_1A/model.pt")', 'ap.add_argument("--init", required=True, help="stage-one checkpoint (scripts/train/train_event.py)")'),
    ("[--init reports/m2/model.pt]", "--init STAGE1/model.pt"), ('init_from="reports/m2/model.pt"', 'init_from=None'),
    ("--models A=reports/m3_1A_z [B=reports/m3_1A_z_no2p2h]", "--models A=weights/full [B=weights/holdout_2p2h]"),
    ("--a reports/m3_1A_on2p2h --b reports/m3_1A_no2p2h_on2p2h", "--a EVAL_FULL_ON_2P2H --b EVAL_HOLDOUT_ON_2P2H"),
    ("[--b-control reports/m3_1A_no2p2h_control] [--a-full reports/m3_1A]", "[--b-control EVAL_HOLDOUT_CONTROL] [--a-full weights/full]"),
    ("Reads metrics.json / metrics_tier2.json written by run_m3.py", "Reads metrics.json / metrics_tier2.json written by scripts/train/train_prong.py --eval-only"),
    ("(reports/deadtime)", "(reports/deadtime, not part of the release)"),
    (' (technote, section "Ensemble of the deployed model")', ""), (' (technote section "Uncertainty awareness")', ""),
    (" The earlier\nBayesian-last-layer (posterior-draw) columns of the same names in `surrogate_Aall_*` files are superseded: their\nbinned uncertainty is zero by construction.", ""), ("Writes reports/<out>/tables/closure_ccinc.tex and closure_ccinc_macros.tex. ", ""),
]
REWRITES = [(k, v) for k, v in sorted(OLD_SCRIPT_PATHS.items(), key=lambda kv: -len(kv[0]))] + REWRITES

def rewrite(text, dest_rel):
    for k, v in REWRITES: text = text.replace(k, v)
    if dest_rel.startswith("scripts/") and dest_rel.count("/") == 2:  # one level deeper than before
        text = text.replace("pathlib.Path(__file__).resolve().parents[1]", "pathlib.Path(__file__).resolve().parents[2]")
    return text

def copy(src_rel, dest_rel, text=True):
    s, d = SRC / src_rel, DEST / dest_rel; d.parent.mkdir(parents=True, exist_ok=True)
    if not s.exists(): print("MISSING", src_rel); return
    if text and s.suffix in (".py", ".sh", ".md", ".txt", ".tex", ".toml", ".json"):
        d.write_text(rewrite(s.read_text(), dest_rel)); d.chmod(s.stat().st_mode)
    else: shutil.copy2(s, d)

# ---------------------------------------------------------------- build
if DEST.exists():
    for p in DEST.iterdir():
        if p.name == ".git": continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
DEST.mkdir(parents=True, exist_ok=True)
for k, v in PACKAGE.items(): copy(f"sim2reco/{k}", f"xsum/{v}")
for k, v in SCRIPTS.items(): copy(f"scripts/{k}", v)
for k, v in WEIGHTS.items():
    for f in WEIGHT_FILES: copy(f"{k}/{f}", f"{v}/{f}", text=False)
for k, v in PAPER_INPUTS.items(): copy(k, v)
for k, v in DOCS.items(): copy(k, v)
for t in TESTS: copy(f"tests/{t}", f"tests/{t}")
copy("pyproject.toml", "pyproject.toml")
import awkward as ak
stem = "MasterAnaDev_mc_AnaTuple_run00110000_Playlist"; (DEST / "tests/data").mkdir(parents=True, exist_ok=True)
for kind, n in (("reco", 5000), ("truth", 10000)):
    ak.to_parquet(ak.from_parquet(SRC / f"data/slim_1A/{stem}.{kind}.parquet")[:n], DEST / f"tests/data/{stem}.{kind}.parquet", compression="zstd")
    shutil.copy(SRC / f"data/slim_1A/{stem}.{kind}.typenames.json", DEST / f"tests/data/{stem}.{kind}.typenames.json")
for f in sorted((SRC / "open_release").glob("**/*")):  # hand-written files of the release kept in the private repo
    if f.is_file(): d = DEST / f.relative_to(SRC / "open_release"); d.parent.mkdir(parents=True, exist_ok=True); d.write_text(rewrite(f.read_text(), str(f.relative_to(SRC / "open_release")))); d.chmod(f.stat().st_mode)

# ---- the ntuple writer without the Bayesian-last-layer code path
p = DEST / "scripts/infer/surrogate_to_ntuple.py"; s = p.read_text()
a0 = s.index('    ap.add_argument("--epi-draws"'); a1 = s.index("\n", a0) + 1; s = s[:a0] + s[a1:]
b0 = s.index("    elif a.epi_draws > 0 and bl_path.exists():"); b1 = s.index('posterior draws (common noise)", flush=True)\n', b0) + len('posterior draws (common noise)", flush=True)\n'); s = s[:b0] + s[b1:]
c0 = s.index('    bl_path = pathlib.Path(a.model_dir) / "bayes_last.pt"\n'); s = s[:c0] + s[c0 + len('    bl_path = pathlib.Path(a.model_dir) / "bayes_last.pt"\n'):]
s = s.replace("# ---- epistemic uncertainties: analytic per-variable variance (model space) + K common-noise posterior draws (physical units)",
              "# ---- model uncertainty: one common-noise draw per ensemble member (the universes) and the between-member spread per event")
assert "bayes" not in s.lower(), [l for l in s.splitlines() if "bayes" in l.lower()]
p.write_text(s)

# ---- package: pyproject for xsum, trimmed requirements
p = DEST / "pyproject.toml"; s = p.read_text().replace('name = "xsum"', 'name = "xsum"').replace('description = "MINERvA open-data sim-to-reco surrogate"', 'description = "XSuM-MINERvA: a surrogate detector model for the MINERvA open dataset"')
p.write_text(s)
(DEST / "requirements.txt").write_text("""# Python >= 3.9. Create the environment with: bash scripts/setup_env.sh
torch>=2.8,<2.9
numpy>=1.26,<2.1
uproot>=5.6,<6
awkward>=2.8,<3
xrootd>=5.6
fsspec-xrootd>=0.5
pyarrow>=17
pandas>=2.0
matplotlib>=3.8
scipy>=1.13
scikit-learn>=1.5
tqdm
pytest>=8
""")
(DEST / ".gitignore").write_text("""# data and outputs never go in the repository
*.root
*.parquet
*.npz
*.pkl
*.log
data/slim*/
reports/
runs/
outputs/
__pycache__/
*.pyc
.venv/
*.egg-info/
# the small example file for the tests is kept
!tests/data/*.parquet
""")
# ---- tests: fixtures look in tests/data first
p = DEST / "tests/conftest.py"; s = p.read_text()
s = s.replace('_cands = sorted(ROOT_DIR.glob("data/slim*/MasterAnaDev_mc_AnaTuple_run*_Playlist.reco.typenames.json"))',
              '_cands = sorted(ROOT_DIR.glob("tests/data/MasterAnaDev_mc_AnaTuple_run*_Playlist.reco.typenames.json")) or sorted(ROOT_DIR.glob("data/slim*/MasterAnaDev_mc_AnaTuple_run*_Playlist.reco.typenames.json"))')
p.write_text(s)
print("exported to", DEST)
for root, dirs, files in sorted((str(pathlib.Path(r).relative_to(DEST)), d, f) for r, d, f in __import__("os").walk(DEST) if ".git" not in r):
    if files: print(f"  {root}/: {len(files)} files")
