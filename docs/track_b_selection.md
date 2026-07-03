# Track B — Benchmark Selection Deliberation

**Purpose.** This document is the authoritative record of the panel review that
produced the SAVANT Track B benchmark suite. It exists so any manuscript
reviewer or downstream researcher can trace *why* every method was included or
excluded from the leaderboard.

**Selection panel.** Simulated senior-researcher panel:
- Football analytics researcher (paradigmatic coverage of the field's evolution).
- Machine learning researcher (methodological rigor, reproducibility, statistical validity).
- Scientific benchmarking expert (fair comparison, computational tractability, publication credibility).
- Machine-Learning-with-Applications reviewer (post-panel critique, publication-readiness).

**Constraint.** All methods must be reproducible on **StatsBomb Open + 360**
using their published specification. Methods that require proprietary tracking
data or whose specification is not publicly complete are excluded.

**Corpus scope.** Selection is drawn ONLY from the 24 candidate papers in
`data/outputs/extractions/`. Methods outside the corpus (e.g. Atomic-VAEP,
which is in the socceraction library but not a paper in our corpus) are not
eligible.

---

## Included methods (8 + 3 baselines)

### Baselines

| id | Method | Rationale |
|----|--------|-----------|
| B1 | Goals per 90                | Naive reference every reviewer expects. Trivial to compute from event data. |
| B2 | Assists per 90              | Naive reference. Complements B1 for creative players. |
| B3 | Raw xG per shot             | StatsBomb-Open ships with per-shot xG. Sets the shot-value floor every downstream method must beat. |

### Corpus-derived methods

| # | id | corpus_paper_id | Family | Why included |
|:-:|----|-----------------|--------|--------------|
| 1a | `xt_state_singh` | `singh_2018_xt` | rule-based Markov, **state-value** | Canonical zone-based xT (Singh 2018 original). Every subsequent method benchmarks against xT. Cannot omit. Reference implementation via Sumpter's Soccermatics `plot_ExpectedThreat`. |
| 1b | `xt_action_sumpter` | `singh_2018_xt` | rule-based, **action-value** | Sumpter Soccermatics `xTAction`: V(a) = P(shot\|a) × xG(shot\|a). Distinct methodology from 1a despite sharing the xT name. **Included to empirically test the SAVANT `valuation_unit` axis** (state vs action); see § "The state vs action pair" below. |
| 2 | `vaep_socceraction` | `decroos_2019_vaep` | supervised tree-ensemble, action-value | Canonical action-value framework. The `socceraction` library (ML-KULeuven) provides a mature reference implementation. |
| 3 | `daxt_merhej` | `merhej_2021_daxt` | supervised MLP, defensive-action | Only well-specified defensive-action method in the corpus. Fills the empirical gap where 0/24 of our methods currently value defensive actions on public data. |
| 4 | `vanroy_mdp` | `vanroy_2021_markov_reward` | MDP + hierarchical Bayesian, state-value | Only reproducible MDP method with a public-data path. PyMC3 + Bellman value iteration are standard and well-documented. Statistically principled alternative to xT. |
| 5 | `vanroy_leaving_goals` | `vanroy_2021_leaving_goals` | MDP + counterfactual, both-unit | Second MDP application (decision-recommendation vs pure valuation) from the same group. Pair with #4 to validate MDP framework across use cases. |
| 6 | `playerank` | `pappalardo_2019_playerank` | learning-to-rank, action-value | Only LTR paradigm in corpus. **Code released** (github/mesosbrodleto/playerank). Adds real methodological diversity. Journal-quality validation (Wyscout scout agreement). |
| 7 | `obso_ff` | `spearman_2018_beyondxg` | physics pitch-control, state-value | Only physics-based method in corpus. Full math specified in the paper. StatsBomb-360 freeze frames enable an event-time OBSO approximation (labelled OBSO_FF with explicit caveat). |
| 8 | `liu_deep_rl` | `liu_2020_deep_soccer` | RL Q-function, LSTM | Only RL method in corpus that uses only event data (not tracking). Reproducible on StatsBomb-Open. Fills the "no RL representative" reviewer concern. |

---

## Excluded methods (16) — with rationale

Every exclusion is one of four categories:

- **[data]**   Method requires proprietary tracking (25 Hz) that public StatsBomb data cannot substitute for.
- **[spec]**   Method specification is incomplete or slide-deck-only; faithful reproduction requires guesswork.
- **[overlap]** Method is superseded by a cleaner in-corpus variant already selected.
- **[scope]** Method is a comparative-analysis paper, not a new model, so implementing it is out of scope for a leaderboard row.

| corpus_paper_id | Category | Rationale |
|-----------------|:--------:|-----------|
| `rudd_2011_markov` | [spec] | Historically foundational but the source is a slide deck; the transition matrix is only sketched. xT (Singh 2018) is the clean, documented successor — implement Singh, cite Rudd as antecedent in the manuscript's Historical Context section. |
| `brooks_2016_passrank` | [overlap] | Well-cited historically but methodologically superseded by xT + VAEP. Linear SVM on pitch-cell features adds no distinct paradigm the suite would miss. Cite as historical context. |
| `decroos_2017_starss` | [overlap] | VAEP predecessor by the same group. VAEP is the mature descendant already in the suite. STARSS' phase-DTW paradigm is nice conceptually but adds engineering effort disproportionate to the incremental methodological signal. |
| `power_2017_passes` | [data] | Feature vector explicitly requires micro-tracking (formations, defender proximity from 25 Hz tracking). StatsBomb-360 freeze frames provide only per-event snapshots; the temporal features cannot be reconstructed. |
| `bransen_2018_pass_contributions` | [data] | Method spec clear but SciSports private dataset. k-NN with hand-tuned distance would require re-tuning on public data, defeating the "reproduce the method" goal. |
| `bransen_2019_pass_value` | [data] + [overlap] | JQAS paper but reproduction requires Bayesian distance-metric optimization + hand-crafted feature engineering. High effort, low incremental value over VAEP. |
| `bransen_2019_choke_or_shine` | [scope] + [spec] | Uniquely valuable *conceptually* (mental pressure) but stitches 5 sub-models with expert-pairwise-ranking training data. Out of scope for a *unified* benchmark. |
| `dick_2019_positioning` | [data] | Deep CNN + TD-λ over 25 Hz DFL tracking. StatsBomb-360 freeze frames are insufficient for TD-λ over full trajectories. |
| `dick_2021_actions` | [data] | GRNN + MDN + RL on 22-player graph tracking. Same reason as Dick 2019 + graph is computationally heavier. |
| `fernandez_2019_epv` | [data] + [spec] | Requires proprietary STATS + Footvision tracking. Three-network decomposition undisclosed in detail. Cannot be faithfully reproduced. Spearman OBSO covers the "state-value on positional data" role. |
| `llana_2020_off_ball_advantage` | [data] + [spec] | Barcelona-specific model, requires proprietary Metrica Sports data + team-specific defensive assignment learner. Not reproducible. |
| `vanroy_2020_xt_vs_vaep` | [scope] | Comparative-analysis paper, not a new method. Include as *methodological antecedent* for the benchmark's design; do NOT implement a "xT-vs-VAEP model" — we implement both individually. |
| `nakahara_2023_multiagent_rl` | [data] + [spec] | Multi-agent RL is publication-worthy but reproducing 10-agent Q-networks on public data is a paper in itself, not a benchmark row. Meiji J1 tracking data not public. |
| `rahimian_2022_beyond_av` | [data] + [spec] | Belgian Pro tracking (proprietary), no code. Even self-reproduction on public data would require months of engineering. |
| `rahimian_2022_data_driven` | [data] + [overlap] | InStat private tracking, no code. Van Roy MDP is a cleaner MDP/RL representative for the benchmark. |
| `wu_2023_offball` | [data] | CSL private tracking, method depends on per-frame velocity prediction. Not reproducible on freeze frames. |

---

## Reviewer criticisms — how the selection defends against them

The Machine-Learning-with-Applications reviewer critique (see full report at
top of the SAVANT session log) flagged 7 anticipated criticisms. Below is how
the final selection addresses each:

| # | Criticism | Defense |
|:-:|-----------|---------|
| 1 | *"Benchmark has no RL method — RL is a major branch of the corpus."* | Included Liu 2020 as the RL representative (event-data-only, reproducible). |
| 2 | *"No naive baselines — how do we interpret the leaderboard?"* | Added B1 (goals/90), B2 (assists/90), B3 (xG/shot) as reference floor. |
| 3 | *"Evaluation protocol is not unified — each method reports its own metric."* | Four tasks (T1–T4) with fixed metrics that all methods must project onto. Report `null` where not defined. |
| 4 | *"Singh xT and Soccermatics xT — is this xT counted twice?"* | Sumpter's Soccermatics is used as the *implementation* of Singh 2018. Single row (`xt_action_sumpter`), two citations. |
| 5 | *"Rankings validated against what ground truth?"* | Scout ratings + FIFA rank + Ballon d'Or top-30 lists used for T3 validation (planned per configs/benchmark_tasks.yaml). |
| 6 | *"OBSO on freeze frames ≠ OBSO."* | Renamed OBSO_FF; explicit caveat in method registration; discussed as an approximation in the manuscript. |
| 7 | *"Why exclude Bransen ECOM 2019? It's a JQAS-published method."* | Exclusion documented above (this file). Referee can trace the reason. |

---

## Data expansion (2026-07-02)

The initial config used 5 competitions (150 training + 183 holdout matches).
Reviewer sanity check flagged that this was too tight for Liu 2020 (deep RL,
originally trained on ~3,000 matches) and PlayeRank (originally 19,619
matches). The final configuration expands to **7 training + 2 holdout
competitions = 843 matches (~2.9M events, 332 matches with SB-360)**.

### Design principle: SB-360 is method-specific, not benchmark-wide

Only OBSO_FF requires freeze-frame data. The other 7 methods run on event
data alone, so competitions without 360 are ELIGIBLE training data for them
even if OBSO_FF cannot use them. Each method reports on its eligible subset,
with sample size disclosed. This is standard leaderboard practice.

### Expanded training set (660 matches)

| Competition | Season | Matches | 360 | Rationale |
|-------------|--------|--------:|:---:|-----------|
| FIFA World Cup | 2022 | 64 | ✅ | Recent men's international peak, 360 available |
| UEFA Euro | 2024 | 51 | ✅ | Recent men's continental, 360 available |
| Women's World Cup | 2023 | 64 | ✅ | Gender diversity, 360 available |
| La Liga | 2020/21 | 35 | ✅ | Domestic top-flight with 360 |
| 1. Bundesliga | 2023/24 | 34 | ✅ | League diversity (Germany), 360 available |
| Ligue 1 | 2022/23 | 32 | ✅ | League diversity (France), 360 available |
| La Liga | 2015/16 | 380 | ❌ | Massive event-only sample — critical for Liu 2020 + PlayeRank convergence |

### Holdout for T4 (183 matches)

| Competition | Season | Matches | 360 | T4 contrast |
|-------------|--------|--------:|:---:|-------------|
| FA Women's Super League | 2020/21 | 131 | ❌ | Women's domestic vs men's domestic training |
| African Cup of Nations | 2023 | 52 | ✅ | African international vs Euro/Copa international |

### Method-specific eligible training samples

| Method | Eligible matches | Notes |
|--------|-----------------:|-------|
| xT (Sumpter Soccermatics) | 660 | All eligible; only ~12×8 grid to fit → generous |
| VAEP (socceraction) | 660 | Comfortable — 4x the original Wyscout-100 benchmark |
| DAxT (Merhej) | 660 | ~100k defensive events; adequate |
| Van Roy MDP | 660 | Bayesian posteriors will be well-constrained |
| Van Roy Leaving Goals | 660 | Same |
| PlayeRank | 660 | ~3% of original 19,619 corpus; will report with wider CIs |
| Liu 2020 (deep RL) | 660 | ~22% of original 2,976; enough for a smaller architecture; discuss in paper |
| OBSO_FF | **280** | 360-only subset; sufficient for the freeze-frame approximation |

## The state vs action pair (xt_state_singh vs xt_action_sumpter)

**Why we include both.** The SAVANT v0.3.0 schema has `model.valuation_unit`
as a first-class field with values `{action, state, both}`. It is the
primary structural axis of the SAVANT ontology, and a substantive
methodological claim: state-based methods (Singh xT, Spearman OBSO,
Fernández EPV, Van Roy MDP) are meaningfully distinct from action-based
methods (VAEP, DAxT, PlayeRank).

If Track B implements only one xT variant, we would be **failing to test
that schema-level claim**. Including both lets us empirically answer:

> For downstream tasks T1–T4, does it matter whether xT is implemented
> as state-value-with-derived-actions vs directly-as-action-value? Or
> are the two variants near-identical?

The community-level ambiguity (both are called "xT") is not a reason to
suppress the distinction — it is a reason to expose it in a controlled
benchmark.

**Methodological delta.**

| Aspect | `xt_state_singh` (Singh 2018) | `xt_action_sumpter` (Sumpter Soccermatics) |
|--------|------------------------------|-------------------------------------------|
| Primary target | V(z) = P(score in possession from zone z) | V(a) = P(shot \| a) × xG(shot \| a) |
| Math structure | Markov Bellman, 12×8 zone transition matrix | Two learned models multiplied per action |
| Zone map? | Yes, 12×8 | No — per-action features only |
| valuation_unit | `state` | `action` |
| Per-action value | Derived: V(z_end) − V(z_start) | Direct output |
| Family | expected-threat | expected-threat |

Both cite `singh_2018_xt` as the corpus paper because that blog is the
ideological ancestor of the whole xT concept; the implementation column
credits Sumpter Soccermatics for the code path.

**Reviewer defense.** *"Why is xT counted twice?"* → *"Because Singh's
original formulation is state-valued (Markov Bellman on zones) and
Sumpter's Soccermatics variant is action-valued (P(shot)·xG directly).
The two are commonly conflated as 'xT' in the community; we separate
them to empirically test whether the SAVANT schema's valuation_unit
distinction affects downstream benchmark performance. This is a
schema-validity test, not a duplicated method."*

## Empirical-agentic feedback loop

Once Track B produces empirical scores, each method's metrics flow back into
the KG as attributes on the Method node:

```
Method[vaep_socceraction] {
    ...existing attributes...
    empirical_t1_brier_sb_open:   0.184
    empirical_t2_log_loss:        0.412
    empirical_t3_pearson_rho:     0.71
    empirical_t4_delta_brier:     +0.06
    benchmark_run_id:             "2026-07-XX-1"
}
```

This unlocks:

- **Critic-agent v2** — can now compare paper's *claimed* metric (in
  extraction record) to our *reproduced* metric on public data. Any Δ > 10%
  is flagged for manuscript disclosure ("Method claims 0.83 on Wyscout;
  reproduces at 0.71 on StatsBomb-Open — declines 14% under data shift.").
- **Gap-agent v2** — can now propose empirically-grounded gaps. Rather than
  only "cell (X, Y, Z) is empty in the coverage matrix", the gap agent can
  say "even the best-performing MDP method achieves only 0.62 T2 log-loss —
  the empirical frontier for MDP-family valuation has not been broken."
- **UI (Coverage + Live Demo pages)** — surface empirical scores as a new
  column in the leaderboard view. Filter methods by "beats VAEP T1 by more
  than X%" or similar.

This is what makes Track B more than a leaderboard: it closes the loop
between the *paper-claim layer* (extraction records) and the *reality layer*
(measured metrics on public data), and lets the agentic pipeline reason
about the discrepancy.
