# SAVANT

**S**occer **A**ction **V**aluation: an **A**gentic framework for k**N**owledge synthesis, gap discovery, and benchmarking of **T**ransparent action-value models.

Companion code and manuscript for the paper:

> *SAVANT: An Agentic Framework for Knowledge Synthesis, Gap Discovery, and Benchmarking of Soccer Action Valuation Models.*
> Submitted to *Machine Learning with Applications*.

---

## 1. What this project is

Soccer action valuation (xT, VAEP, Atomic-VAEP, EPV, possession-value models, etc.) has exploded over the last decade, but the literature is fragmented across analytics blogs, conference proceedings, and journal papers. There is no single, machine-readable map of what has been tried, on what data, under what assumptions, and how the methods compare on a common benchmark.

SAVANT is a reproducible research pipeline that tries to fix this. It is built around three coupled ideas:

1. **Agentic literature synthesis.** A team of LLM agents (screening, extraction, taxonomy, critic) ingests a curated paper corpus and produces a structured database describing each model: inputs, target, supervision signal, evaluation, dataset, claims, and limitations.
2. **Knowledge graph + gap analysis.** Extracted records are projected into a knowledge graph linking methods → assumptions → evaluation protocols → datasets. A gap-analysis agent walks the graph to surface under-explored regions (e.g. "no published Atomic-VAEP variant evaluated on StatsBomb 360 with calibration metrics").
3. **Reproducible empirical benchmark.** A reference implementation of xT, VAEP, Atomic-VAEP, and an EPV-style baseline is evaluated on StatsBomb Open Data (and StatsBomb 360 where licence permits), under a shared evaluation protocol designed by the benchmark agent.

The output is (a) a structured knowledge base of the field, (b) a ranked list of research gaps, (c) a benchmark leaderboard, and (d) the manuscript itself, written collaboratively with a manuscript agent that grounds every claim in the extracted database.

---

## 2. Repository layout

```
savant-action-valuation/
├── README.md                  ← this file
├── CLAUDE.md                  ← coding rules and workflow for the AI collaborator
├── requirements.txt           ← pinned Python deps
├── pyproject.toml             ← package metadata + tooling config
├── .gitignore
│
├── data/                      ← all data (gitignored content)
│   ├── papers/                ← PDFs of the corpus
│   ├── raw/                   ← raw extracted text per paper
│   ├── processed/             ← cleaned, chunked text + metadata
│   ├── statsbomb/             ← StatsBomb Open Data + 360 frames (where allowed)
│   ├── annotations/           ← human spot-checks for agent outputs
│   └── outputs/               ← extraction records, KG dumps, benchmark results
│
├── configs/
│   ├── paths.yaml             ← absolute/relative paths used by the pipeline
│   ├── agents.yaml            ← model, temperature, retry policy per agent
│   ├── extraction_schema.json ← JSON Schema for the structured paper record
│   └── benchmark_tasks.yaml   ← benchmark task definitions and splits
│
├── prompts/                   ← one Markdown file per agent role
│
├── src/savant/                ← installable Python package
│   ├── ingestion/             ← PDF → text → chunks
│   ├── extraction/            ← agent runners + schema validation
│   ├── taxonomy/              ← clustering/normalisation of extracted fields
│   ├── knowledge_graph/       ← graph construction + queries
│   ├── gap_analysis/          ← gap-detection algorithms
│   ├── benchmarking/          ← xT, VAEP, Atomic-VAEP, EPV implementations
│   ├── evaluation/            ← metrics, calibration, statistical tests
│   ├── visualization/         ← figures for the paper
│   ├── manuscript/            ← LaTeX helpers, table/figure injection
│   └── utils/                 ← logging, config loading, IO
│
├── notebooks/                 ← exploratory + demo notebooks
├── scripts/                   ← thin CLI entry points
├── manuscript/                ← LaTeX source of the paper
└── tests/                     ← pytest suite
```

---

## 3. End-to-end workflow

The pipeline is staged so any step can be re-run independently from a cached upstream artefact.

1. **Corpus ingestion** — drop PDFs into `data/papers/`, run `scripts/run_screening.py` to deduplicate, parse metadata, and filter to in-scope work.
2. **Agentic extraction** — `scripts/run_extraction.py` calls the extraction agent (with a critic in the loop) and writes one JSON record per paper into `data/outputs/extractions/`, validated against `configs/extraction_schema.json`.
3. **Taxonomy normalisation** — `scripts/build_knowledge_graph.py` first runs the taxonomy agent to canonicalise free-text fields (model family, supervision signal, evaluation metric, dataset), then materialises the knowledge graph.
4. **Gap analysis** — `scripts/run_gap_analysis.py` queries the KG and ranks underexplored (method × data × evaluation) cells.
5. **Benchmark execution** — `scripts/run_benchmark.py` trains and evaluates the reference models on StatsBomb Open / 360 under the protocol in `configs/benchmark_tasks.yaml`.
6. **Manuscript assembly** — `scripts/generate_figures.py` produces every figure and table in `manuscript/figures/` and `manuscript/tables/`; the manuscript agent then drafts/updates `.tex` sections from the structured artefacts.

---

## 4. Reproducibility

- Python 3.11, dependencies pinned in `requirements.txt`.
- All agent calls are logged with model id, prompt hash, and seed.
- StatsBomb Open Data is fetched on demand; 360 data and any non-open corpus material is **not** redistributed by this repo — paths are configurable.
- Every benchmark result is reproducible from a single seed + the cached extraction artefacts.

---

## 5. Status

**Phase 1 — literature review database** is in progress. Concretely:

- `configs/extraction_schema.json` is the source-of-truth schema for each paper record.
- `data/papers/papers.template.csv` and `data/annotations/annotations.template.csv` are the human-curated working sheets that seed the gold standard. They ship with example rows for xT, VAEP, Atomic-VAEP, EPV, OBV, Goals Added, and tracking-based EPV (most cells are intentional `<PLACEHOLDER>` markers awaiting verification).
- `src/savant/utils/{config,io,logging}.py` and `src/savant/ingestion/metadata.py` are the Phase 1 utilities: YAML config loader, atomic JSON IO, schema validation, and a typed CSV loader for the papers sheet.
- `tests/test_schema.py` and `tests/test_paper_metadata.py` validate the schema and the templated CSVs on every `pytest` run.
- See **`docs/phase1_workflow.md`** for what Axum fills, what Pegah verifies, and what the agentic pipeline (Stages 2–4) will later automate.

Stages 2 onwards (PDF ingestion, extraction agent, taxonomy, KG, gap analysis, benchmark, manuscript) are still scaffolded stubs. See `CLAUDE.md` for the implementation plan and conventions before extending the code.
