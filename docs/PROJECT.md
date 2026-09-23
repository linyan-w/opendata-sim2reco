# MINERvA sim→reco surrogate: project design

Status: design draft, 2026-09-23. No code yet by intent; the goal of this document is to make the idea
solid enough to start coding. Open questions are collected in §10 and are meant to become GitHub issues.

## 1. Goal

Learn, from the MINERvA open-data MC alone, a **per-event stochastic map from GENIE final-state truth to
MasterAnaDev reconstructed quantities**, and make it extrapolate sensibly beyond the phase space the released
MC covers.

Why this matters for open data specifically:

- The open-data release contains reconstructed MC with truth attached, but not the simulation chain
  (GENIE → Geant4 → overlay → reconstruction). An outside user can reweight the MC, but cannot generate
  reco-level predictions for a final state that GENIE 2.12.6 with the MINERvA tune did not produce.
- The natural application is therefore: take truth-level events from **another generator or model variation**
  (GENIE 3 tunes, NuWro, NEUT, ACHILLES, a different FSI model, more 2p2h, ...), run them through the
  surrogate, and compare to MINERvA data at reco level, or build a reco-level response matrix for a new signal
  definition. Reweighting cannot do this when the new model populates final states with different multiplicity,
  species, or kinematics.
- Secondary uses: fast MC for statistics-limited corners; sensitivity studies; a differentiable detector model.

What the surrogate is not: it does not fix data/MC disagreement. It is trained on MC and reproduces the MC
detector response, including its flaws. Its value is transporting that response to new truth inputs.

## 2. Problem statement

For one neutrino interaction, let

- `X` = the set of particles leaving the nucleus, each with species and 4-momentum (variable size),
- `c` = event-level context: true vertex position, struck target/material, beam configuration, and per-spill
  detector conditions,
- `R` = the reconstructed record, which is itself structured: a "was reconstructed at all" flag, a fixed block of
  event-level quantities (muon, vertex, calorimetric sums), and a **variable-size set of hadron prongs**.

We want to learn `p(R | X, c)`, sample from it, and characterize where the samples can be trusted.

Three properties of this problem drive the design:

1. **It is genuinely stochastic and multimodal.** In the file examined, muon momentum reco/true has a median of
   0.98 but a 16–84% band of [0.52, 1.13]; 36% of events are off by more than 20%. The reasons are discrete
   (MINOS-matched or not, range vs curvature, exiting muon). A regression to the mean is useless; we need a
   conditional generative model.
2. **Reco quality is an event property, not a particle property.** Reco multiplicity saturates near 2 while true
   charged-hadron multiplicity keeps growing (table in `tuple_notes.md`): tracks overlap, vertices get pulled,
   calorimetry mixes. A per-particle smearing table cannot represent this. Hence per-event.
3. **Everything visible on the reco side has to be explainable by what physically entered the detector.**
   This is the constraint that gives extrapolation a chance (§7).

## 3. Inputs: which truth particles, which features

**Particle set.** Use `mc_FSPart*` (equivalently `mc_er_*` with `status == 1`): GENIE's stable final state
**after** intranuclear FSI. These are exactly the particles that leave the nucleus and enter Geant4. The pre-FSI
hadronic system (`mc_er_status` 14 etc.) is invisible to the detector and must not be used. Concretely:

| Keep | Drop |
|---|---|
| mu±, e±, p, n, pi±, pi0, gamma, K±, K0/K0bar (K0L/K0S), Lambda, Sigma, other hyperons | outgoing neutrinos (PDG ±12, ±14, ±16): invisible |
| | GENIE pseudo-particle 2000000101 (binding energy carrier) |
| | nuclear remnants 100ZZZAAA0: recoil nuclei with keV–MeV KE, invisible |

Neutrons stay. They are the most common FS particle in this sample and they do produce visible energy (blobs,
extra prongs: 17% of events with zero true charged hadrons above threshold still have a second reco track).
pi0 and gamma stay for the same reason. Whether to decay pi0 → γγ on the input side is an open question (§10).

**Per-particle features.** For each particle: a learned embedding of a species class (about 12 classes), and
kinematic features. Two options: raw `(px, py, pz)` in detector coordinates, or `(log KE, cos θ, φ)` plus mass.
Recommendation: the latter, with the muon and hadrons treated by the same encoder (the muon is just a token with
its own species class). Reasons: KE is what determines range and calorimetric response, and log KE spans the
MeV-to-tens-of-GeV dynamic range evenly; θ w.r.t. the beam axis is the relevant angle for tracking in a
planar detector.

**Do not feed generator-internal labels.** `mc_intType` (QE/RES/DIS/2p2h), `mc_Q2`, `mc_w`, `mc_Bjorkenx/y`,
`mc_incomingE`, and `mc_targetNucleon` are not observable by the detector; the detector only sees `X`. A
surrogate that uses them would (a) learn shortcuts that break when a different generator labels things
differently, and (b) be impossible to apply to a generator that does not produce those labels. Excluding them
also gives us a clean extrapolation test: train on QE+RES, test on DIS (§7).

## 4. Conditioning: vertex, target, detector conditions

The question raised was whether the vertex is an input or a condition. Mathematically both `X` and `c` sit on
the right-hand side of `p(R | X, c)`; the model conditions on everything. The distinction that matters is
architectural and practical:

- **Is it something we want to vary independently at inference time?** For the vertex, yes: "same final state,
  different location" is a meaningful query (e.g. moving a hadronic system from tracker to a nuclear target
  region). That argues for treating it as an explicit, separately-controllable context vector.
- **Does it act on each particle or on the event as a whole?** The vertex acts on both. Globally it decides
  which material the event starts in and how much detector lies downstream. Per particle, it decides (together
  with the particle's direction) how much material the particle traverses before exiting, which drives
  containment and hence momentum-by-range vs. calorimetry.

Recommendation: put the vertex in the **global context vector** `c`, broadcast it to every particle token
(via a global token or FiLM), and additionally give each particle a few **derived geometric features** computed
from (vertex, direction): projected path length to the downstream ECAL/HCAL boundary and to the side, and whether
the straight-line ray exits the tracker sideways. These are cheap to compute from the geometry and they make the
"new vertex" extrapolation a change in a smooth scalar feature rather than something the network has to
rediscover from raw coordinates.

Proposed contents of `c`:

| Quantity | Branch | Notes |
|---|---|---|
| True vertex (x, y, z) | `mc_vtx[0:3]` | Detector coordinates in mm. Time is irrelevant. |
| Target material | `mc_targetZ`, `mc_targetA`; `truth_targetID`, `truth_vtx_module` | Largely a function of z, but not fully (tracker is a C/H mix, water target, passive layers). Keep both raw position and material class. |
| Beam / playlist | `mc_beamConfig`; neutrino vs antineutrino mode | Constant within a playlist; needed when training across playlists (ME FHC vs RHC). |
| Detector conditions | `phys_n_dead_discr_pair` (dead channels), `numi_pot` (spill intensity → overlay pileup), possibly `ev_gps_time_sec` binned into run periods | In ME MC these come from the **data overlay**, so they have realistic distributions and are available identically in data. At inference they are sampled from the empirical distribution of the run period one wants to emulate. |

One caveat: the more we put into `c`, the more the surrogate is tied to MINERvA-ME-specific conditions. Keep `c`
small and physically interpretable.

## 5. Outputs: what "reco variables" means concretely

MasterAnaDev does not produce a generic reco particle list. It produces one muon hypothesis, a set of hadron
prongs each scored under several particle hypotheses, and a stack of calorimetric sums. We should model that
structure rather than invent a particle list that does not exist in the tuple. Tiered plan:

**Tier 0: selection.** Was there a MasterAnaDev entry for this truth event at all? (32% overall in this file;
~a factor higher inside the fiducial volume.) Plus the quality flags analyses cut on: `minos_trk_is_ok`,
`nuHelicity`, containment. This is a set of Bernoulli/categorical outputs conditioned on `(X, c)` and is a
required part of the surrogate: efficiency is the first thing any reco-level prediction needs.

**Tier 1: fixed-size event block (~15 continuous numbers).**
Muon: `MasterAnaDev_muon_P`, `muon_theta`, `muon_phi`, `muon_qp` sign, MINOS momentum method.
Vertex: `MasterAnaDev_vtx[0:3]`.
Calorimetry: `MasterAnaDev_recoil_E`, `MasterAnaDev_hadron_recoil`, `recoil_energy_nonmuon_nonvtx100mm`,
`nonvtx_iso_blobs_energy`, `vtx_blobs_energy`.
Counts: `multiplicity` (or `MasterAnaDev_hadron_number`), `n_nonvtx_iso_blobs`.
Derived quantities (`MasterAnaDev_E`, `_Q2`, `_W`) are functions of these and should be computed, not modelled.

**Tier 2: variable-size hadron prong set.** `N = multiplicity − 1` prongs, each with: momentum (`pion_P`
or `proton_P_fromdEdx` as appropriate), θ, φ, proton score(s), pion score(s), is-contained/is-exiting, has
end-Michel. We should build one canonical prong table from the `proton_*`, `sec_protons_*`, `pion_*[10]` and
`hadron_*[10]` arrays (they describe the same prongs under different hypotheses; the `hadron_tm_*` truth-match
fields tell us which true particle each prong came from and are gold for validation).

**Tier 3 (later):** Michel electrons, pi0/EM blobs (`gamma1/2_*`), per-subdetector recoil splits.

**Parameterize as response, not absolute value, where a truth reference exists.** Model
`log(P_reco/P_true)` for the muon, `θ_reco − θ_true`, `vtx_reco − vtx_true`, and `recoil / Σ visible KE`
rather than the raw reco values. Response distributions are far smoother and more local in the inputs than
absolute values, which is what we need for extrapolation. For prongs, a matched truth reference is available
through `hadron_tm_trackID` at training time, but at inference the model must produce the prong set without
knowing the matching; see §6.

## 6. Handling the variable-length output

This was the main open question. Four ways to do it, and what each buys:

**(a) Cardinality first, then contents: `p(R | X,c) = p(N | X,c) · p(r_1..r_N | N, X, c)`.**
This factorization is exact, not an approximation. Step 1 is a small categorical head (N ≤ 8 covers everything
in this sample). Step 2 is a conditional generative model over an N×d array with a mask, permutation-equivariant
in the N prongs. This is precisely how point-cloud generative models in HEP handle variable-size jets
(PC-JeDi, EPiC-FM, CaloClouds): sample the multiplicity from a learned conditional, then diffuse/flow the
points given it. The proposed two-step scheme is the standard approach, and diffusion does not remove the need
for it; it just makes step 2 easy.

**(b) Fixed maximum slots with existence flags.** The tuple itself does this (`pion_*[10]` with −1 fill).
Simple, but it forces an ordering on an unordered set (sort by momentum) and mixes a discrete existence bit into
a continuous model. Regression-style models trained this way are known to smear across slot boundaries. Fine as
a baseline; not the target design.

**(c) Autoregressive sequence (transformer decoder emitting prongs until a stop token).** Handles variable length
and mixed discrete/continuous natively; the cost is an imposed order and slower sampling. A solid alternative to
(a) if the set model struggles with the discrete score variables.

**(d) Sidestep: fixed-size targets only.** Tier 0 + Tier 1 are already fixed-size and already cover what most
MINERvA inclusive and CC0π-style analyses use. This is the MVP and the first milestone.

Recommendation: build (d) first with a conditional normalizing flow or flow matching over the ~15-dim Tier 1
vector plus classification heads for Tier 0 and the multiplicity, then extend to (a) for Tier 2 with a
conditional flow-matching model over the masked prong set. Keep (c) in reserve.

Two details that matter for (a):

- **Permutation symmetry.** Reco prongs are unordered, so the step-2 network should be permutation-equivariant
  (transformer without positional encoding over prong tokens, cross-attending to the encoded truth tokens). Then
  no canonical ordering is needed and the "which true particle made which prong" assignment is learned
  implicitly through attention instead of imposed.
- **Mixed variable types.** Scores in [0, 1] and flags are not Gaussian-friendly. Logit-transform the scores
  (with dequantization for values pinned at 0 or 1), and give hard flags their own discrete heads or use
  discrete-diffusion for them.

## 7. Extrapolation: what it means here and how to make it credible

"Extrapolation" covers at least four distinct shifts, and they are not equally hard:

| Shift | Example | Difficulty | What helps |
|---|---|---|---|
| Composition | New generator produces more 2p2h (two low-KE protons), fewer pions | Mostly interpolation in per-particle space, new combinations at event level | Set encoder (compositionality); response parameterization |
| Multiplicity | Events with 5+ visible hadrons, rare in GENIE 2 | Extrapolation in N | Equivariant architecture; multiplicity holdout tests; capacity to represent saturation |
| Kinematic tails | 2 GeV protons, very forward pions, high-KE neutrons | Extrapolation in a scalar | Physics-motivated features (log KE, path length); monotonic priors; response variables |
| Geometry | New vertex regions, target edges | Interpolation if the region is covered, extrapolation otherwise | Geometric derived features (§4) |

Design choices that give the model the right inductive bias, in order of expected impact:

1. **Detector-causal inputs only** (§3). The response then depends on the physics that actually differs between
   generators, not on labels.
2. **Compositional architecture.** A per-particle token encoder plus attention means the model can, in
   principle, compose per-particle responses it learned in low-multiplicity events into high-multiplicity ones.
   Whether it actually does is exactly what the multiplicity holdout test measures.
3. **Response parameterization** (§5). Smooth, bounded targets extrapolate better than raw ones.
4. **Symmetry augmentation.** Rotate the whole event about the beam axis (approximate hexagonal symmetry of the
   detector, exactly valid in 60° steps and the X/U/V plane structure breaks it beyond that); reflect x → −x.
   Cheap and physically justified in the tracker; check validity in the nuclear-target region.
5. **Ensembles for out-of-distribution flagging.** Train 5–10 models with different seeds; disagreement between
   them in a region of `(X, c)` is a usable OOD score. Report it with every surrogate prediction.

Validation protocol, which is the actual science of the project: **hold out physics-defined regions of phase
space, not random events**, and measure how well the model trained on the rest reproduces reco distributions
there. Proposed holdouts, each a separate experiment:

- Train on `mc_intType ∈ {QE, RES, 2p2h}`, test on DIS. Since intType is not an input, this is a pure
  composition and multiplicity shift.
- Train on events with ≤ 2 visible hadrons, test on ≥ 3.
- Train on proton KE < 400 MeV, test on the tail.
- Train on tracker vertices only, test on nuclear-target region (geometry shift).
- Train on the FHC file, test on an RHC file (antineutrino: different muon charge, more neutrons, different
  composition). This one also tests the beam-config conditioning.

Metrics: per-variable 1D distributions (KS / Wasserstein) and, more importantly, **conditional** ones: response
vs true KE, efficiency vs true multiplicity, multiplicity confusion matrix, correlations between muon and recoil.
Plus a binary classifier test (real vs surrogate reco given truth) as an overall figure of merit.

Honest expectation: composition shifts should work well, multiplicity extrapolation moderately, deep kinematic
tails are hard for any data-driven model. The ensemble OOD score exists to tell the user where the last case
applies.

## 8. Model architecture (proposal)

```
truth particles  ──► per-particle features + species embedding ──┐
                                                                  ├──► transformer encoder ──► z (event embedding)
context c        ──► MLP ──► global token ─────────────────────────┘        + per-token outputs
                                                                            │
   ┌────────────────────────────────────────────────────────────────────────┤
   ▼                       ▼                        ▼                        ▼
Tier 0 heads           cardinality head        Tier 1 generative        Tier 2 generative
(Bernoulli:            p(N | z)                 p(y_event | z)           p(prongs | N, z, tokens)
 reconstructed?,       categorical N ≤ 8        conditional flow /       conditional flow matching
 MINOS ok, helicity)                             flow matching, ~15-d    over masked N×d set,
                                                                          equivariant velocity net
```

Baselines to run first, because they define "good enough" and catch bugs: (i) gradient-boosted trees for Tier 0
and the cardinality head from hand-crafted event summaries (Σ KE by species, n protons above threshold, ...),
(ii) a mixture-density network for the muon response, (iii) a classical per-particle smearing table.

Generative model choice: flow matching (continuous normalizing flow trained by regression on a vector field) is
the current default in HEP fast simulation (EPiC-FM, FlashSim) and is simpler to train than score-based diffusion
at these dimensionalities while giving fast sampling. A discrete-time diffusion model is a fine alternative;
exact-likelihood normalizing flows are attractive for the low-dimensional Tier 1 block because the likelihood is
directly useful for validation. Start with flow matching for both tiers to keep one codebase.

## 9. Data and compute plan

- Slim each AnaTuple to Parquet with only the ~80 branches listed in `tuple_notes.md` (truth particles, context,
  Tier 0–2 targets, truth-match fields), using uproot in chunks. Expect well under 1 GB per 20 GB input file.
  Also slim the `Truth` tree (unreconstructed events) so Tier 0 has its denominator.
- The example file has 168k reco / 519k truth events. The full ME FHC playlist is many files; even 5–10 files
  give a million reco events, which is plenty for Tier 1 and adequate for Tier 2. Get files from
  https://minerva.fnal.gov/getdata.
- Local machine: 20 cores, 30 GB RAM, one RTX 3090 (24 GB). Enough for everything above. `torch` is not yet
  installed; only `uproot`/`awkward`/`numpy` are, as user packages on Python 3.9.

## 10. Open questions (to become GitHub issues)

1. **pi0 handling on the input side:** keep pi0 as one token, or decay to two photons so the encoder sees what
   Geant4 sees? Decaying is more "detector-causal" but adds sampling noise to the inputs.
2. **Neutrons and very-low-KE particles:** keep all, or apply a low-KE cut (e.g. drop protons < 20 MeV that
   cannot make a hit)? A cut is a modelling assumption; keeping everything is cleaner but adds noise tokens.
3. **Canonical prong table:** how exactly to merge `proton_*`, `sec_protons_*`, `pion_*[10]`, `hadron_*[10]`
   into one unordered set of prongs with consistent fields. Needs a careful look at how MasterAnaDev fills them.
4. **Which recoil definition(s) to target:** `recoil_E`, `hadron_recoil_CCInc`, or the radius-scan
   `recoil_energy_nonmuon_*` family. Different analyses want different ones; modelling several jointly is fine.
5. **Overlay / detector-condition conditioning:** confirm which branches are populated from the data overlay
   in ME MC and available identically in data (`numi_pot`, `phys_n_dead_discr_pair`, run period).
6. **Vertex as separately controllable context vs. plain input** (§4): agree on the recommendation and on the
   derived geometric features.
7. **Cross-playlist scope:** ME FHC only for the first pass, or FHC+RHC from the start? Antineutrino mode is
   also the cleanest extrapolation test.
8. **Systematics-aware training:** the tuple carries hundreds of universe weights and particle-response shifts.
   Ignore for now, or use `part_response_*` shifts as data augmentation so the surrogate can also emit
   detector-systematic variations?
9. **Definition of "the reco event exists":** the `MasterAnaDev` tree requires a muon candidate. Should Tier 0
   model the MasterAnaDev preselection, or a specific analysis' full cut chain?
10. **Target of extrapolation for the first physics study:** which alternative generator output do we want to
    push through first (GENIE 3 hA vs hN FSI? NuWro?), and in what format do we get its final-state particles
    (NUISANCE flat trees are the obvious route).

## 11. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | Slimming script, Parquet dataset from ≥ 3 MC files, EDA notebook reproducing `tuple_notes.md` | Dataset loads in seconds; truth/reco pairing verified with `eventID` |
| M1 | Baselines: GBDT efficiency + multiplicity, MDN muon response | Numbers to beat; sanity plots |
| M2 | Tier 0 + Tier 1 conditional flow-matching model, random split | Closure: real-vs-surrogate classifier AUC < 0.55 on held-out events |
| M3 | Physics-holdout extrapolation study (§7), ensemble OOD score | Written report of where it works and where it does not |
| M4 | Tier 2 prong set model (cardinality + set flow matching) | Reproduces multiplicity confusion and prong kinematics |
| M5 | First application: alternative-generator truth → surrogate reco → comparison to open data | Paper-quality reco-level comparison |

## 12. References and prior art

- MINERvA open data: https://minerva.fnal.gov/opendata/ (CC0, cite https://doi.org/10.15484/3022562).
  Branch docs: https://github.com/MinervaExpt/Tuple-Documentation. Tutorial: MINERvA-101-Cross-Section.
- FlashSim (CMS): end-to-end generator-level → reco-level (NanoAOD) simulation with normalizing flows / flow
  matching; the closest precedent to this project in spirit. Vaselli et al., arXiv:2402.13684.
- Point-cloud generative models with learned cardinality for variable-size sets in HEP: PC-JeDi (Leigh et al.,
  arXiv:2303.05376), EPiC-FM (Birk et al., arXiv:2310.00049), CaloClouds (Buhmann et al., arXiv:2305.04847).
- Conditional normalizing flows for detector simulation: CaloFlow (Krause & Shih, arXiv:2106.05285).
- Flow matching: Lipman et al., "Flow Matching for Generative Modeling", arXiv:2210.02747.
