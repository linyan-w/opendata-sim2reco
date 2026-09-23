# opendata-sim2reco

An AI/ML surrogate for the MINERvA detector: per-event mapping from GENIE final-state truth to MasterAnaDev
reconstructed variables, trained on the [MINERvA open data](https://minerva.fnal.gov/opendata/) MC, built to
extrapolate to final states the released MC does not cover (alternative generators, higher multiplicity, new
kinematic regions).

Status: M0 (data pipeline) done; M1/M2 (baselines, first generative model) next. Read [`docs/PROJECT.md`](docs/PROJECT.md) first.

## Quick start

```
bash scripts/setup_env.sh                                  # Python >= 3.9 venv with torch (CUDA), uproot, ...
source .venv/bin/activate
python scripts/slim.py data/slim MasterAnaDev_mc_AnaTuple_run00113069_Playlist.root   # 20 GB ROOT -> 300 MB Parquet
python scripts/dataset_summary.py data/slim/MasterAnaDev_mc_AnaTuple_run00113069_Playlist
pytest -q                                                  # round-trip tests against the example ntuple
```

## Layout

- `sim2reco/io/` — branch lists, ROOT -> Parquet slimming, pruned-ntuple writer.
- `sim2reco/prep/` — input pipeline (particle selection, context), canonical prong table encode/decode, muon decode, beam/detector frames.
- `sim2reco/data/` — torch dataset (padded particle sets, context, Tier 0/1 targets) and subrun-level splits.
- `sim2reco/models`, `train`, `eval/` — empty until M1/M2.
- `configs/data.yaml` — selection and split settings.
- `tests/` — pipeline and exact round-trip tests on the example ntuple.
- `docs/PROJECT.md` — project design: goal, inputs, conditioning, outputs, variable-length handling, extrapolation strategy, milestones, open questions.
- `docs/tuple_notes.md` — what is actually in the open-data MC AnaTuple, with measured numbers from one ME FHC file.
- `docs/branches/` — full branch lists of the `MasterAnaDev`, `Truth`, `Meta` trees, and a copy of the upstream `MAD_tuple_MainDoc.csv`.

## Data

Open-data AnaTuples (`MasterAnaDev_{mc,data}_AnaTuple_run*_Playlist.root`) live next to this repo but are not
committed (see `.gitignore`); get them from https://minerva.fnal.gov/getdata. Reading only needs
`uproot` and `awkward`.

## Acknowledgment

The authors thank the MINERvA Collaboration for making their data, simulated data, and analysis tools available
to the community. Data release: https://doi.org/10.15484/3022562 (CC0).
