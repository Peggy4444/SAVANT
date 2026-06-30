# SAVANT Corpus Guide

**For David Sumpter (and Axum). Prepared by Pegah, June 2026.**

Welcome — and thank you for helping. SAVANT is a research project building a curated, machine-readable database of the soccer **action-valuation** literature (xT, VAEP, EPV, possession-value, defensive-action value, off-ball metrics, RL-based decision frameworks, etc.). The output is a survey paper plus a reproducible benchmark, designed to be useful for *both* researchers in football analytics and data scientists at clubs.

Your role is to **verify** the rows that have already been auto-extracted from each paper. Pegah handles all the technical bits — you just read the paper, compare it to the row, and flip a flag.

This guide is short on purpose. If anything is unclear, ping Pegah.

---

## The two sheets

| Sheet | One row per | What goes in it |
|-------|-------------|------------------|
| **Papers**       | paper | Bibliographic info (title, authors, year, venue, DOI), screening decision, corpus-pool flag. |
| **Annotations**  | paper *(only candidates — see below)* | The methodological detail — what the model is, what data it uses, how it was evaluated, what action types it values, and so on. ~50 columns. |

A paper's row in **Papers** may or may not have a matching row in **Annotations**. That distinction is the most important rule in this guide.

---

## The candidate vs held-out rule

Each paper in **Papers** has a `corpus_pool` column with one of two values:

- **`candidate`** — these papers form the **"answer key"** we'll use later to evaluate an AI extraction model. Candidates have a full hand-verified annotation row.
- **`held-out`** — these papers go in **Papers** only, never in **Annotations**. They are the AI's test set. Later the AI will attempt to extract their details on its own, and we'll spot-check.

**Today's corpus contains 28 papers: 24 candidates + 4 held-out.** All 24 candidate rows already have a pre-filled annotation row (auto-extracted from the PDF), so your job is verification, not creation.

---

## What's in scope

SAVANT is **soccer-only**. A paper is in scope if it proposes, modifies, evaluates, or benchmarks a *soccer-specific* action-value or state-value model, or applies one to a downstream task (player ranking, scouting, tactical analysis, decision-recommendation) with methodological detail. Out of scope: other sports, pure data-engineering / referee / market-value papers, and match-outcome papers that don't break results down per action.

---

## If you're David — methodological verifier

For each paper Pegah sends you:

1. **Read the paper end-to-end**, focusing on whether the auto-extracted row faithfully reflects the paper's methodology and contribution. Most attention should go to:
   - `model_valuation_unit` (action / state / both)
   - `model_atomicity` (per-event / per-action-pair / per-frame / per-possession)
   - `model_supervision` (supervised / reinforcement / rule-based / physics-based / hybrid)
   - `action_coverage` (which action types the method actually values)
   - `outputs_value_scale` and `outputs_aggregation_levels`
   - `evaluation_metrics`, `evaluation_tasks`, `evaluation_baselines`, `splits`
   - `data_dataset_canonical` (which controlled dataset id, if any)
   - `key_claims` (are the cited section/table references correct?)
2. **If you agree the row faithfully reflects the paper**, flip in **Papers**: `david_verified=true`, `david_verified_date=YYYY-MM-DD`. And in **Annotations**: `verified_by_david=true`, `verified_by_david_date=YYYY-MM-DD`.
3. **If you disagree on something**, do NOT silently overwrite. Add a note in the `verification_notes` column prefixed with your name, e.g.:
   `David: model_atomicity should be per-frame, not per-event (see §3.2 of the paper).`
   Pegah will arbitrate.
4. You're **not expected to fill new rows** in Phase 1. Verification only.

**Priority papers** — please at least look at the 6 canonical methods, in this order:

1. Karun Singh, *Introducing Expected Threat (xT)*, 2018
2. Tom Decroos et al., *VAEP — Actions Speak Louder than Goals*, KDD 2019
3. Javier Fernández, Luke Bornn, Dan Cervone, *EPV — Decomposing the Immeasurable Sport*, MIT Sloan 2019
4. William Spearman, *Beyond Expected Goals*, MIT Sloan 2018
5. Sarah Rudd, *A Framework for Tactical Analysis Using Markov Chains*, NESSIS 2011 (slide deck — quicker)
6. Joel Brooks et al., *Pass-rank with predictive model weights*, KDD 2016

If time allows, the next 6 candidates that give us a real second-reader signal across methodological families:

7. Charbel Merhej et al., *DAxT — Defensive Action xT*, KDD 2021 (defensive)
8. Lucas Wu & Tim Swartz, *Evaluation of Off-the-Ball Actions in Soccer*, 2023 (off-ball)
9. Nakahara et al., *Multi-agent deep RL for on+off-ball valuation*, IEEE Access 2023 (RL)
10. Maaike Van Roy et al., *Learning a Markov reward function*, ICML workshop 2021 (MDP)
11. Guiliang Liu et al., *Deep soccer analytics*, DMKD 2020 (RL on event data)
12. Luca Pappalardo et al., *PlayeRank*, ACM TIST 2019 (learning-to-rank, unique paradigm)

---

## The gold-standard rule

A row becomes "gold-standard" once **at least one verifier** has signed off AND **that verifier is not the row's annotator**. Pegah has self-verified all 24 candidate rows (`verified_by_pegah=true`). David's independent sign-off is what unlocks the inter-annotator-agreement (IAA) signal we'll cite in the manuscript.

---

## Column quick reference — allowed values

A few columns have a restricted vocabulary. Please use exactly these strings (lowercase, with hyphens where shown):

**`venue_type`:** `journal`, `conference`, `workshop`, `preprint`, `blog`, `thesis`, `report`, `other`

**`screening_decision`:** `include`, `exclude`, `uncertain`

**`corpus_pool`:** `candidate`, `held-out`

**`primary_tag`:** `action-valuation`, `expected-threat`, `VAEP`, `atomic-VAEP`, `EPV`, `possession-value`, `pass-value`, `shot-value`, `defensive-action`, `off-ball-action`, `tracking-data`, `event-data`, `graph-neural-network`, `deep-learning`, `reinforcement-learning`, `markov-decision-process`, `interpretability`, `calibration`, `ranking-players`, `tactical-analysis`, `other`

**`model_family`:** `expected-threat`, `VAEP`, `atomic-VAEP`, `EPV`, `possession-value`, `markov`, `neural`, `graph-neural`, `tree-ensemble`, `rule-based`, `hybrid`, `other`

**`model_framework`:** `expectation-based`, `markov-decision-process`, `reinforcement-learning`, `physics-based`, `rule-based`, `hybrid` (or empty)

**`model_valuation_unit`:** `action`, `state`, `both`
- `action` — the model directly predicts a value per action (VAEP, OBV, g+, Atomic-VAEP, DAxT)
- `state` — the model predicts a value per game state; action values are derived as V(s_after) − V(s_before) (xT, EPV, Spearman pitch-control)
- `both` — the model explicitly trains two heads/losses, one per unit

**`model_supervision`:** `supervised`, `self-supervised`, `reinforcement`, `inverse-reinforcement`, `unsupervised`, `rule-based`, `physics-based`, `hybrid`

**`model_atomicity`:** `per-event`, `per-action-pair`, `per-frame`, `per-possession`, `per-zone`, `other`

**`action_coverage`:** semicolon-separated subset of `passes`, `shots`, `dribbles_carries`, `defensive_actions`, `off_ball_actions`, `set_pieces`, `goalkeeper_actions`. Include only those the method actually assigns values to.

**`outputs_value_scale`:** `probability`, `expected-goals`, `expected-points`, `unitless-rating`, `log-odds`, `value-delta`, `other`

**`outputs_aggregation_levels`:** semicolon-separated subset of `per-action`, `per-possession`, `per-zone`, `per-frame`, `per-player-match`, `per-player-season`, `per-team-match`, `per-team-season`

**`data_type`:** semicolon-separated subset of `event`, `tracking-2d`, `tracking-360`, `broadcast-video`, `synthetic`, `other`

**`data_dataset_canonical`:** semicolon-separated subset of `statsbomb-open`, `statsbomb-360-open`, `statsbomb-private`, `wyscout-public-2017-18`, `wyscout-private`, `opta-private`, `skillcorner-public`, `metrica-sports-public`, `sportlogiq-private`, `instat-private`, `club-proprietary`, `synthetic`, `other-public`, `other-private`, `unknown`

**`evaluation_tasks`:** semicolon-separated subset of `goal-prediction`, `action-ranking`, `player-ranking`, `team-ranking`, `tactical-pattern-discovery`, `calibration`, `transfer-recommendation`, `scouting`, `win-probability`, `decision-recommendation`, `other`

**`practical_use_cases`:** semicolon-separated subset of `scouting`, `in-game-decision-support`, `post-match-analysis`, `season-evaluation`, `opponent-analysis`, `youth-development`, `transfer-target-id`, `tactical-pattern-discovery`, `training-feedback`, `broadcasting-commentary`, `betting-markets`, `other`

**`practical_interpretability_level`:** `black-box`, `partially-interpretable`, `fully-interpretable`

**`practical_compute_intensity`:** `low`, `medium`, `high`, `not-reported` (or empty)

**`code_implementer`:** `authors`, `third-party`, `both` (or empty)

**`code_maintenance_status`:** `actively-maintained`, `frozen`, `abandoned`, `unknown` (or empty)

**`relationships`:** semicolon-separated tuples of the form `type:target_paper_id`, where `type` is one of `extends`, `compares-against`, `improves-upon`, `uses-dataset-of`, `critiques`. Example: `extends:decroos_2017_starss;compares-against:singh_2018_xt`

For booleans (e.g. `pegah_verified`, `needs_full_text`, `calibration_reported`, every `action_coverage` element): write `true` or `false` (lowercase) — or leave blank if the paper does not state.

For list-valued cells (authors, metrics, leagues, etc.): separate values with **`;`** (semicolon), not commas. Example: `Tom Decroos;Lotte Bransen;Jan Van Haaren;Jesse Davis`.

For dates: ISO format `YYYY-MM-DD`. Example: `2026-06-29`.

---

## A note on PDFs and file paths

Don't worry about PDF storage or any `file_path` columns. Just open the paper Pegah sends you (PDF / link), read it, and update the cells.

---

*Questions? Pegah's address is on the email this PDF was attached to.*
