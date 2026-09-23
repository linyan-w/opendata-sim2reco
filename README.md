# minerva-sim2reco

An AI/ML surrogate for the MINERvA detector: per-event mapping from GENIE final-state truth to MasterAnaDev
reconstructed variables, trained on the [MINERvA open data](https://minerva.fnal.gov/opendata/) MC, built to
extrapolate to final states the released MC does not cover (alternative generators, higher multiplicity, new
kinematic regions).

Status: design phase. Read [`docs/PROJECT.md`](docs/PROJECT.md) first; open questions are listed in its §10.

## Layout

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
