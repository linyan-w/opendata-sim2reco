# Retired scripts

Kept for provenance (the technote and docs/PROJECT.md cite them); moved here on 2026-11-03. Their inputs are no longer on disk
unless noted, and they are not part of the open release. Import paths were adjusted so they still run from the repository root.

| group | scripts | superseded by |
|---|---|---|
| Generator placement before the geometry-placed samples: vertex pool, re-vertexing, the inclusive stratum construction tied to the GENIE sample, and their comparisons | `nuwro_to_truth.py`, `gibuu_to_truth.py`, `revertex_truth.py`, `nuwro_inclusive.py`, `nuwro_compare.py`, `fullvolume_compare.py`, `technote_inclusive.py`, `nuwro_surrogate_service.sh`, `pipeline_fullvolume.sh`, `pipeline_gibuu.sh`, `pipeline_inclusive.sh`, `pipeline_nc.sh`, `pipeline_ensemble_infer.sh`, `pipeline_ensemble_nuwro.sh`, `pipeline_vz_nuwro.sh` | the collaborator's GDML-vertexed samples (`data/geo_2026-11/HANDOFF.md`), `scripts/build_geo_truth.py`, `thin_geo_truth.py`, `pipeline_geo_2026-11.sh`; outputs of the old comparisons in `reports/inclusive*` |
| CC-only model era and input-definition variants (10 MeV threshold, neutron tokens, zero flags) | `pipeline_1A.sh`, `pipeline_adopt_ke10n.sh`, `pipeline_variants.sh`, `pipeline_variant_ke10n.sh`, `variants_compare.py`, `pipeline_zero*.sh`, `zero_compare.py`, `run_m1.py`, `closure_subset.py`, `paper_fig_conditionals.py`, `pipeline_paper_figs.sh`, `reference_categories.py` | the all-events models (`scripts/pipeline_all.sh`), `closure_subset_all.py`, `paper_fig_response.py`; numbers in `reports/history` |
| Adoption of the z-conditioned vertex-plane head and the vertex-tail diagnostics | `pipeline_vz_adopt.sh`, `pipeline_vz_adopt_fixB.sh`, `pipeline_vz_adopt_resume.sh`, `vtx_tail_diag.py`, `vtx_tail_isolate.py` | `scripts/pipeline_vz.sh` (the deployed stage); results in `reports/vtx_tail` |
| 2p2h coverage by process | `coverage_by_process.py` | `scripts/coverage_2p2h.py`, `paper_fig_coverage.py` |
