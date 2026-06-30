# SAVANT — Top-5 Gap Rationales (Manuscript Draft, v1)

**Source.** Derived from `data/outputs/gaps/ranked.json` (auto-generated 2026-06-29 by `savant.gap_analysis.rank_gaps`) over the 24-paper SAVANT candidate corpus.

**Editorial note.** The mechanical ranking returned six near-identical entries at the top (ranks 1–6, all `MDP × action × *`). Rather than five near-duplicate paragraphs, this draft consolidates those into **Gap 1** and selects four other gaps that surface distinct stories — different framework, different action type, or a data/reproducibility angle the 3-D matrix can't see. Each gap names the auto-ranked cells it covers so traceability back to `ranked.json` is preserved.

These paragraphs are intended as drop-in material for the manuscript's *Open Problems* section. They will be re-checked and polished by the gap-agent (v2) once that agent is implemented.

---

## Gap 1 — MDP frameworks have not yet produced per-action values

**Auto-rank coverage:** #1, #2, #3, #4, #5, #6 (all `markov-decision-process × action × *`); score 2.66–2.79.

**The gap.** Four corpus papers use the MDP framework (Rudd 2011; Van Roy 2021 *Markov reward*; Van Roy 2021 *Leaving Goals*; Rahimian 2022 *Beyond AV*), but every one of them produces a **state value** rather than an explicit per-action value. No corpus paper proposes a method where the MDP's primary output is a per-action `V(a)` for the seven action types we track (passes, shots, dribbles/carries, defensive, off-ball, set-pieces, goalkeeper).

**Why it matters scientifically.** MDPs natively support per-action `Q(s, a)` decomposition via the Bellman equation; the absence of a per-action MDP method in the corpus is a structural opportunity, not a methodological barrier. An MDP-action method would unify the principled counterfactual reasoning of Van Roy 2021 (which can already answer *"what if the team shot less and passed more?"* at the policy level) with the actionable per-event values that VAEP-family methods provide. The closest existing work — VAEP itself — produces per-action values but lacks the policy-evaluation guarantees of an MDP.

**Why it matters for clubs.** A per-action MDP value gives an interpretable answer to *"should this player have passed instead of dribbled here?"* that is grounded in optimal-policy theory rather than a binary classifier. Coaches asking decision-recommendation questions today get heuristic answers; an MDP-action framework would let those answers carry uncertainty bounds.

**Closest related work** (shares 2/3 dimensions): `decroos_2019_vaep`, `bransen_2019_choke_or_shine`, `liu_2020_deep_soccer`, `vanroy_2021_markov_reward`, `vanroy_2021_leaving_goals`.

**Suggested first study.** Reuse Van Roy 2021's StatsBomb-open MDP state space; add a per-action `Q(s, a)` head trained via policy iteration with a held-out match-level test split. Compare per-action `Q` values against VAEP on the same matches. Report calibration (Brier, ECE) and ranking agreement (Spearman ρ on top-20 player lists).

---

## Gap 2 — Expectation-based methods are structurally blind to off-ball actions

**Auto-rank coverage:** #9 (`expectation-based × action × off_ball_actions`); score 2.49.

**The gap.** The expectation-based framework is the most populous family in the corpus (8 methods: Brooks 2016, STARSS 2017, Bransen 2018, Bransen 2019 *pass-value*, Bransen 2019 *Choke or Shine*, VAEP 2019, DAxT 2021, Pappalardo 2019 PlayeRank). None values off-ball actions, because all eight operate on event-stream data, which by definition records only on-ball events. Off-ball valuation in the corpus exists exclusively in physics-based (Spearman), tracking-based RL (Dick 2019, Dick 2021), or multi-agent RL (Nakahara 2023, Wu 2023) methods — all data-heavy.

**Why it matters scientifically.** The expectation-based framework is the lingua franca of soccer AV (it is the supervision target of every VAEP variant). Its off-ball blind spot is a methodological wall, not a research preference: extending it requires either (a) layering tracking-derived off-ball features onto event-data supervision (the Llana 2020 strategy), or (b) defining a synthetic off-ball event vocabulary from tracking data. Neither is yet a clean published method.

**Why it matters for clubs.** Players spend ~85% of match time off-ball. Clubs with event-data-only deals (the majority, given StatsBomb tracking is private) are effectively capped at valuing 15% of player behaviour with EB methods. A method that bridges this gap on a public dataset would have immediate commercial pull.

**Closest related work.** `wu_2023_offball` and `nakahara_2023_multiagent_rl` value off-ball but in different frameworks; `llana_2020_off_ball_advantage` is the only paper that layers off-ball signal onto an EB-style EPV target.

**Suggested first study.** Take VAEP's event-data labels and augment the feature vector with SkillCorner-public off-ball player positions. Train a per-action VAEP variant where the on-ball event's value depends on the positions of the 9 off-ball teammates. Evaluate vs. vanilla VAEP on rank-stability across rotating off-ball contexts.

---

## Gap 3 — Goalkeepers are invisible to every framework except VAEP

**Auto-rank coverage:** #19 (`markov-decision-process × action × goalkeeper_actions`), and the broader pattern across ranks 35–40 (`reinforcement-learning × * × goalkeeper_actions`); average score ~2.0.

**The gap.** Only **2 of 24** corpus methods (VAEP via SPADL's `keeper save` action; Bransen 2019 *Choke or Shine* via the same vocabulary) assign any value to goalkeeper actions. No RL method, no MDP method, no physics-based method, no learning-to-rank method values goalkeeper behaviour. Distribution choice — arguably the single highest-leverage decision a modern keeper makes — is unmeasured by every framework except the most generic event-classifier.

**Why it matters scientifically.** Goalkeepers occupy a structurally distinct role (single agent, low-frequency high-leverage decisions, asymmetric reward structure). A targeted GK-AV method has to reckon with reward sparsity (few shots per match faced) and the long-tail of distribution decisions whose outcomes only materialise 2–3 actions later. This is precisely the setting that RL with episode-bounded value functions was designed for, yet no corpus paper has applied it.

**Why it matters for clubs.** GK scouting is a recognised unsolved problem in club analytics. The transfer market for GKs currently relies on traditional shot-stopping metrics (save percentage, post-shot xG) that miss distribution, sweeping, and command-of-box. A first principled GK-AV method has commercial value disproportionate to its corpus-statistical "gap weight".

**Closest related work.** `decroos_2019_vaep` (only via SPADL's `keeper save` action type, not a goalkeeper-specific model); `bransen_2019_choke_or_shine` (inherits VAEP's GK action vocabulary).

**Suggested first study.** Frame GK distribution as an MDP where states are *(received-by zone, opponent press intensity)*, actions are *(short / medium / long / launch)*, and reward is the change in possession-value over the next k=10 actions. Train on StatsBomb-open EPL data restricted to GK passes. Compare expected-rewards against a baseline that always picks the most-frequent distribution length per state.

---

## Gap 4 — Physics-based methods are confined to off-ball offence

**Auto-rank coverage:** ranks 39, 46, 52 (`physics-based × * × defensive_actions`); score 1.60–1.86.

**The gap.** Exactly one corpus method (Spearman 2018) uses the physics-based framework. It models off-ball offensive scoring opportunities (OBSO) via pitch-control. No corpus method extends the physics-based approach to defensive valuation, even though the underlying pitch-control formalism is symmetric — the same Voronoi-style spatial-influence model that estimates *where the attacker can score* can in principle estimate *where the defender denies space*.

**Why it matters scientifically.** A "defensive pitch-control" extension would let researchers value defensive positioning principles (line shape, compactness, pressing trigger zones) on a probabilistic scale that matches the existing OBSO scale, enabling direct attacking-vs-defending value comparisons. This is the kind of cross-domain extension that bridges two otherwise-isolated branches of the corpus (physics-based + defensive valuation).

**Why it matters for clubs.** Defensive scouting today relies on aggregate tackle/interception counts and heuristic shape metrics. A physics-based defensive value would give the same interpretability that has made Spearman's OBSO uniquely loved by tactical analysts.

**Closest related work.** `spearman_2018_beyondxg` (the lone physics-based method); `wu_2023_offball` (defensive anticipation, but ML-based, not physics-based).

**Suggested first study.** Symmetrise Spearman's OBSO derivation: replace `P(team A scores | state)` with `P(team B fails to advance | state)`, using the same pitch-control field but evaluating defender-controlled zones. Validate on Metrica-public defensive sequences with a held-out test set.

---

## Gap 5 — The reproducibility wall: tracking-dependent methods do not use public tracking

**Auto-rank coverage:** *not visible in the 3-D matrix* — surfaces from the `dataset_x_action_type` projection.

**The gap.** Seven of 24 corpus methods require tracking data (Spearman, Fernández EPV 2019, Dick 2019, Llana 2020, Liu 2020, Power 2017, Dick 2021, Rahimian 2022 *Beyond AV*, Rahimian 2022 *Data-Driven*, Nakahara 2023). **Zero** of them are evaluated on a public tracking dataset. Every tracking-based result in the corpus uses STATS LLC, Hudl, Wyscout-private, Metrica-Sports-private, Stats Perform, club-proprietary, or vendor-NDA-bound data. The only public-dataset method in the entire corpus is `vanroy_2021_markov_reward` — and that uses *scraped event data*, not tracking.

**Why it matters scientifically.** This is **the** reproducibility crisis of soccer AV. Every published tracking-based method's experiments are unreproducible by anyone outside elite clubs or vendor partnerships. Cross-method comparison is impossible because no two papers benchmark on the same data. The community currently has no shared evaluation testbed for any tracking-dependent valuation idea.

**Why it matters for clubs and researchers.** Without a public reproducible baseline for tracking-based AV, the community cannot benchmark new methods, mid-sized clubs cannot evaluate vendor offerings against independent baselines, and PhD students need pricey data deals just to enter the field. SkillCorner-Open and StatsBomb-360 exist but are unused by the corpus.

**Closest related work.** None — this is a gap *in the literature itself*, not a methodological combination. Closest non-AV work is `xarles_2025_av_survey` (held-out), which flags the same concern qualitatively.

**Suggested first study.** Re-implement a single canonical tracking-based method (recommend: Spearman 2018 OBSO, because the math is fully specified) on the SkillCorner-Open or StatsBomb-360-Open dataset. Release the implementation + the evaluation script as the **SAVANT-Tracking-Bench v0.1**. Two consequences: (a) every subsequent tracking-based paper has a reproducible baseline to compare to, (b) the manuscript's empirical contribution is the benchmark itself.

---

## Summary table — for the manuscript figure

| # | Gap | Auto-rank | Framework | Valuation unit | Action type / dimension | Public data feasible? |
|---|-----|----------:|-----------|----------------|-------------------------|:---:|
| 1 | MDP per-action values | 1–6 | markov-decision-process | action | all 6 on-ball action types | ✅ |
| 2 | EB off-ball | 9 | expectation-based | action | off_ball_actions | ✅ (with SkillCorner) |
| 3 | Goalkeeper AV | 19, 35+ | any RL/MDP | any | goalkeeper_actions | ✅ |
| 4 | Physics-based defence | 39, 46, 52 | physics-based | state | defensive_actions | ✅ (with Metrica) |
| 5 | Reproducibility — tracking-public | n/a (data axis) | any | any | any tracking-dependent | the gap *is* the public-data axis |

All five gaps have a public-data path to a first study. None require proprietary infrastructure. This is the headline of SAVANT's gap-analysis output: **the field's stated open problems are addressable today, not blocked by data access**.
