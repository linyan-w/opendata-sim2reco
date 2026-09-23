# opendata-sim2reco

An AI/ML surrogate for the MINERvA detector: per-event mapping from GENIE final-state truth to MasterAnaDev
reconstructed variables, trained on the [MINERvA open data](https://minerva.fnal.gov/opendata/) MC, built to
extrapolate to final states the released MC does not cover (alternative generators, higher multiplicity, new
kinematic regions).

Status: M0 (data pipeline) and M1 (baselines) done; M2 (flow-matching model) next. Read [`docs/PROJECT.md`](docs/PROJECT.md) first; performance numbers are in `reports/performance/` (LaTeX, compiled with tectonic).

## Quick start

```
bash scripts/setup_env.sh                                  # Python >= 3.9 venv with torch (CUDA), uproot, ...
source .venv/bin/activate
python scripts/slim_remote.py configs/MediumEnergy_FHC_StandardMC_Playlist1M.txt data/slim --budget-gb 9.5
                                                           # stream from xrootd, 20 GB ROOT -> 300 MB Parquet per file
python scripts/slim.py data/slim some_local_file.root      # same for a local ROOT file
python scripts/dataset_summary.py data/slim/MasterAnaDev_mc_AnaTuple_run00113069_Playlist
python scripts/run_m1.py data/slim/MasterAnaDev_mc_AnaTuple_run00113069_Playlist reports/m1   # baselines, ~1 min on a GPU
pytest -q                                                  # round-trip tests (ROOT file or its slimmed Parquet)
(cd reports/performance && tectonic -X compile main.tex)   # performance report PDF
```

## Layout

- `sim2reco/io/` — branch lists, ROOT -> Parquet slimming, pruned-ntuple writer.
- `sim2reco/prep/` — input pipeline (particle selection, context), canonical prong table encode/decode, muon decode, beam/detector frames.
- `sim2reco/data/` — torch dataset (padded particle sets, context, Tier 0/1 targets) and subrun-level splits.
- `sim2reco/models/` — MDN baseline (flow matching to come); `sim2reco/train/baselines.py` — M1; `sim2reco/eval/` — metrics and plots.
- `reports/m1/` — M1 metrics, LaTeX tables, figures; `reports/performance/main.tex` — the performance report.
- `configs/data.yaml` — selection and split settings.
- `tests/` — pipeline and exact round-trip tests on the example ntuple.
- `docs/PROJECT.md` — project design: goal, inputs, conditioning, outputs, variable-length handling, extrapolation strategy, milestones, open questions.
- `docs/tuple_notes.md` — what is actually in the open-data MC AnaTuple, with measured numbers from one ME FHC file.
- `docs/branches/` — full branch lists of the `MasterAnaDev`, `Truth`, `Meta` trees, and a copy of the upstream `MAD_tuple_MainDoc.csv`.

## Data

Open-data AnaTuples are streamed from Fermilab's xrootd door and slimmed to Parquet under `data/slim/`
(not committed). No ROOT file needs to be stored locally; reading needs only `uproot`, `awkward` and the
`xrootd` Python bindings. Source: https://minerva.fnal.gov/getdata.

## Acknowledgment

The authors thank the MINERvA Collaboration for making their data, simulated data, and analysis tools available
to the community. Data release: https://doi.org/10.15484/3022562 (CC0).
