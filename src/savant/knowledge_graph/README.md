# SAVANT Knowledge Graph — Design (v1)

This document specifies the **structure** of the SAVANT knowledge graph: the
node types, the edge types, the attributes each carries, and the example
queries it must support.

The KG is built **deterministically** from the 24 candidate JSON records under
`data/outputs/extractions/`. It is the substrate for gap analysis, the
manuscript figures, and the practitioner-facing "find me a method that…"
queries.

## 1. Design principles

1. **Rich, not lean.** Author and Venue nodes are included even though they
   bloat the graph slightly. Rationale: the manuscript targets *Machine
   Learning with Applications* (academic) AND practitioners at football clubs
   — both audiences ask "who built this?" and "where was it published?". A
   lean KG cannot answer these without ad-hoc joins to the bibliographic
   block.
2. **Every node carries the minimum attribute set a practitioner needs.** A
   Method node is not just a label — it carries `interpretability_level`,
   `compute_intensity`, `inference_realtime_capable`, `code_released`,
   `data_type_required`. This means most "is this method right for me?"
   queries are answered without joining back to the JSON record.
3. **Every edge has a single, named relation type.** No generic "related-to"
   edges. Typed edges make the lineage diagrams readable and let gap analysis
   filter by edge type.
4. **No edge inferred from authorship overlap or co-citation.** The graph only
   records what the papers themselves state. Co-author co-citation cliques
   can be derived later but are not first-class.
5. **The KG is rebuildable.** `scripts/run_gap_analysis.py` (Step A8) will
   regenerate it from JSONs in one command, so any change to the 24
   records flows through without manual editing.

## 2. Node types

### 2.1 Paper (24 nodes today, the spine of the graph)

| Attribute              | Source field                          | Type    |
|------------------------|---------------------------------------|---------|
| `paper_id`             | `paper_id`                            | str (id) |
| `title`                | `bibliographic.title`                 | str     |
| `year`                 | `bibliographic.year`                  | int     |
| `venue_type`           | `bibliographic.venue_type`            | enum    |
| `doi`                  | `bibliographic.doi`                   | str?    |
| `url`                  | `bibliographic.url`                   | str?    |
| `open_access`          | `bibliographic.open_access`           | bool?   |
| `corpus_pool`          | (from papers.csv)                     | enum    |

### 2.2 Method (24 nodes — one per paper, 1:1 today)

| Attribute                       | Source field                       | Type    |
|---------------------------------|------------------------------------|---------|
| `method_id`                     | `{paper_id}::{model.name}`         | str (id) |
| `name`                          | `model.name`                       | str     |
| `family`                        | `model.family`                     | enum    |
| `framework`                     | `model.framework`                  | enum    |
| `valuation_unit`                | `model.valuation_unit`             | enum    |
| `atomicity`                     | `model.atomicity`                  | enum    |
| `supervision`                   | `model.supervision`                | enum    |
| `inference_realtime_capable`    | `practical.inference_realtime_capable` | bool? |
| `interpretability_level`        | `practical.interpretability_level` | enum    |
| `compute_intensity`             | `practical.compute_intensity`      | enum?   |
| `value_scale`                   | `model.outputs.value_scale`        | enum    |
| `code_released`                 | `code_release.released`            | bool    |
| `code_maintenance_status`       | `code_release.maintenance_status`  | enum?   |

### 2.3 Author (collected from `bibliographic.authors` arrays)

| Attribute  | Type |
|------------|------|
| `name`     | str (id; "FirstLast" slug)  |

### 2.4 Venue (collected from `bibliographic.venue`)

| Attribute    | Type |
|--------------|------|
| `name`       | str (id) |
| `venue_type` | enum |

### 2.5 Framework (controlled vocabulary, 6 nodes)

`expectation-based`, `markov-decision-process`, `reinforcement-learning`, `physics-based`, `rule-based`, `hybrid`.

### 2.6 ValuationUnit (controlled vocabulary, 3 nodes)

`action`, `state`, `both`.

### 2.7 ActionType (controlled vocabulary, 7 nodes)

`passes`, `shots`, `dribbles_carries`, `defensive_actions`, `off_ball_actions`, `set_pieces`, `goalkeeper_actions`.

### 2.8 Dataset (controlled vocabulary — `data.dataset_canonical` enum)

15 nodes: `statsbomb-open`, `statsbomb-360-open`, `statsbomb-private`,
`wyscout-public-2017-18`, `wyscout-private`, `opta-private`,
`skillcorner-public`, `metrica-sports-public`, `sportlogiq-private`,
`instat-private`, `club-proprietary`, `synthetic`, `other-public`,
`other-private`, `unknown`.

Each Dataset node also carries `is_public: bool` (derived: true iff the
identifier ends in `-open` or `-public` or contains `public`).

### 2.9 UseCase (controlled vocabulary, 12 nodes)

From `practical.use_cases` enum: `scouting`, `in-game-decision-support`,
`post-match-analysis`, `season-evaluation`, `opponent-analysis`,
`youth-development`, `transfer-target-id`, `tactical-pattern-discovery`,
`training-feedback`, `broadcasting-commentary`, `betting-markets`, `other`.

### 2.10 EvaluationTask (controlled vocabulary, 11 nodes)

From `evaluation.tasks` enum.

## 3. Edge types

All edges are directed.

### 3.1 Bibliographic edges

| Edge type           | From    | To      | Attributes |
|---------------------|---------|---------|------------|
| `AUTHORED_BY`       | Paper   | Author  | `position` (int) |
| `PUBLISHED_IN`      | Paper   | Venue   | — |

### 3.2 Method-structural edges

| Edge type             | From   | To             | Attributes |
|-----------------------|--------|----------------|------------|
| `PROPOSES`            | Paper  | Method         | — |
| `HAS_FRAMEWORK`       | Method | Framework      | — |
| `HAS_VALUATION_UNIT`  | Method | ValuationUnit  | — |
| `VALUES`              | Method | ActionType     | (only emitted if the corresponding `action_coverage.*` is `true`) |
| `USES_DATASET`        | Method | Dataset        | — |
| `EVALUATED_ON_TASK`   | Method | EvaluationTask | — |
| `SUITABLE_FOR`        | Method | UseCase        | — |

### 3.3 Paper-to-paper edges (typed lineage; from `relationships` array)

| Edge type           | From   | To      | Meaning |
|---------------------|--------|---------|---------|
| `EXTENDS`           | Paper  | Paper   | A's method is a generalisation of B's. |
| `IMPROVES_UPON`     | Paper  | Paper   | A reports better metrics than B on a shared protocol. |
| `COMPARES_AGAINST`  | Paper  | Paper   | A reports B in its baselines table. |
| `USES_DATASET_OF`   | Paper  | Paper   | A reuses the dataset introduced by B. |
| `CRITIQUES`         | Paper  | Paper   | A explicitly identifies a flaw in B's methodology. |

These are populated in **Step A2** from the `evaluation.baselines` free-text
field cross-referenced against the corpus `paper_id` list, plus hand-curated
review of the `scope.rationale` and extraction-agent critic notes.

## 4. Example queries the KG must support

If any of these is hard, the schema is wrong.

### Researcher / academic queries

1. **Lineage of a method family.** "Show me everything downstream of VAEP."
   → Traverse `EXTENDS` and `IMPROVES_UPON` edges out of `decroos_2019_vaep`.
2. **Methodological coverage.** "How many methods use the
   reinforcement-learning framework AND value off-ball actions?"
   → Methods where `HAS_FRAMEWORK→reinforcement-learning` AND `VALUES→off_ball_actions`.
3. **Coverage matrix.** "For each (framework, action_type) cell, count methods."
   → Group Method nodes by (`HAS_FRAMEWORK` target, `VALUES` target).
   This is the matrix the **gap-analysis step (A5)** consumes.
4. **Dataset reuse network.** "Which datasets are reused across multiple methods?"
   → Group methods by `USES_DATASET` target; count.
5. **Critic network.** "Which papers have been critiqued by others in the
   corpus?" → Inbound `CRITIQUES` edges per Paper.

### Practitioner / club-data-scientist queries

6. **"I have only StatsBomb Open Data. Which methods can I use?"**
   → Method nodes whose `USES_DATASET` target has `is_public=true`,
   or whose required data_type is event-only.
7. **"I need a real-time, interpretable method for in-game decision support."**
   → Method nodes WHERE `inference_realtime_capable=true` AND
   `interpretability_level ∈ {partially-interpretable, fully-interpretable}`
   AND `SUITABLE_FOR → in-game-decision-support`.
8. **"Show me methods with released, actively-maintained code that value
   defensive actions."**
   → Method nodes WHERE `code_released=true` AND
   `code_maintenance_status=actively-maintained` AND
   `VALUES → defensive_actions`.
9. **"Which papers cite this author?"**
   → For Author node A, find Methods proposed by Papers that
   `COMPARES_AGAINST` Papers authored by A.
10. **"What's the simplest scouting method that values passes?"**
    → Filter by `SUITABLE_FOR → scouting` AND `VALUES → passes` AND
    sort by `compute_intensity=low` then `interpretability_level=fully-interpretable`.

## 5. What the schema explicitly does NOT model

- **Co-author overlap clique structure.** Can be derived from `AUTHORED_BY`
  edges post-hoc if needed; not first-class.
- **Citation count or h-index** — drifts over time, not a property of the
  paper itself.
- **Inferred edges** — e.g. "A is similar to B because they share 3 inputs".
  All edges come from explicit statements in the papers themselves.
- **Multi-method papers as multiple Methods.** v1 treats every paper as
  proposing exactly one Method. If a future paper proposes 2 distinct
  methods, the schema supports adding a second Method node with a distinct
  `method_id`, but no current corpus paper triggers this.

## 6. Serialization formats

- **Primary:** NetworkX `MultiDiGraph` in-memory.
- **Tracked artifact:** `data/outputs/kg/savant_kg.json` (node-link JSON,
  human-readable, diff-friendly).
- **Tooling artifact:** `data/outputs/kg/savant_kg.graphml` (Gephi /
  Cytoscape compatible, for figure generation in **A4**).

Both files are regenerable from the 24 JSON records via
`scripts/run_gap_analysis.py` (built in **A8**), so they are gitignored.

## 7. Versioning

This schema is `kg_schema_version = "1.0"`. Any change that adds/removes a
node type, edge type, or required attribute bumps the major version. Adding a
new enum value or an optional attribute is a minor bump.
