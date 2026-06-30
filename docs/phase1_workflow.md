# Phase 1 — Literature Review Database (Axum, Pegah, David)

Phase 1 is **all-human**. The pipeline does not call any LLM yet. The goal of
this phase is to produce a small, *trusted* gold-standard dataset of paper
records that:

1. seeds the SAVANT corpus,
2. acts as the ground truth against which Stage 3's extraction agent is later
   evaluated,
3. forces us to discover schema gaps **before** spending tokens on a hundred
   papers.

---

## 0. TL;DR — who does what, and in which file

| Person  | Role                        | What file(s) they edit                                                   |
| ------- | --------------------------- | ------------------------------------------------------------------------ |
| Axum    | First-pass annotator        | `data/papers/papers.csv` (their assigned rows) + `data/annotations/annotations.csv` (only for `corpus_pool=candidate` rows). |
| Pegah   | First-pass annotator **and** verifier | Same as Axum for her own assigned rows; plus flips the `pegah_verified` / `verified_by_pegah` flags on Axum's rows. |
| David   | Verifier (methodological / domain check) | Only flips `david_verified` / `verified_by_david` flags. Does **not** create new rows. |

**Two important rules:**

1. **Gold-standard rule.** A row is *gold-standard* once ≥1 verifier (Pegah
   OR David) has signed off AND that verifier is not the row's annotator.
   Self-verification does not count. Pegah is primary verifier on most rows;
   David adds an independent second-reader signal for inter-annotator
   agreement (IAA) on the seed papers.
2. **Candidate vs held-out rule.** Every row in `papers.csv` has a
   `corpus_pool` field set to either `candidate` or `held-out`.
   - **Candidate rows** get full hand-annotation: bibliographic row in
     `papers.csv` **AND** annotation row in `annotations.csv`, signed off by
     a verifier. There are 18 candidate rows in the template today.
   - **Held-out rows** get a bibliographic row in `papers.csv` **only**.
     They are the test pool for Stage 3's extraction agent — the agent will
     produce records for them in Stage 3 and Pegah will spot-check the
     output. Do **not** add annotation rows for held-out papers.
     There are 7 held-out rows in the template today.

---

## 1. The two files you will be working in

| File                                        | Purpose                                                       | Tracked? |
| ------------------------------------------- | ------------------------------------------------------------- | -------- |
| `data/papers/papers.template.csv`           | Corpus-level bibliographic & screening info per paper.        | **yes** (committed seed) |
| `data/annotations/annotations.template.csv` | Per-paper hand-extracted annotations (candidates only).       | **yes** (committed seed) |

**You do not edit the `*.template.csv` files** — they are the committed
seed and only change when the schema or the corpus list changes. You edit
the working copies, which are gitignored:

```bash
# One-time setup (or whenever the template changes — see "Re-seeding" below):
cp data/papers/papers.template.csv      data/papers/papers.csv
cp data/annotations/annotations.template.csv data/annotations/annotations.csv
```

The current `papers.template.csv` contains **25 rows**:

- **18 candidate rows** (the 7 original method seeds — xT, VAEP, Atomic-VAEP,
  EPV, OBV, Goals Added, tracking-based EPV — plus 11 new candidates
  harvested from the Xarles et al. 2025 survey on 2026-06-28).
- **7 held-out rows** (including the Xarles 2025 survey itself).

Some cells in those rows are deliberately `<PLACEHOLDER: …>` — that is the
work for Phase 1.

### Re-seeding when the template changes

If `papers.template.csv` is updated (e.g. by adding new papers, new columns,
or resolving placeholders), your local `papers.csv` will be **out of date**.
Two cases:

- **You haven't added any data to `papers.csv` yet** → just re-run the
  `cp` commands above to overwrite.
- **You have local edits in `papers.csv`** → merge by hand: add the new
  rows / columns from the template; keep your edits.

Currently (2026-06-28) the template has just been expanded with 18 new rows
and a new `corpus_pool` column. If your local `papers.csv` predates this,
re-cp it now.

---

## 1a. Axum — first-pass annotator (for papers Axum reads)

Axum's work is split across two cases:

### Case A: rows that already exist in `papers.csv` with `corpus_pool=candidate`

These are the 18 candidate rows seeded from the literature. For each, Axum:

1. **Locate the row in `data/papers/papers.csv`** (do not re-create it).
2. **Resolve `<PLACEHOLDER: …>` cells** with verified data from the PDF.
3. **Drop the PDF into `data/papers/`** and set `file_path` to its relative
   path (e.g. `data/papers/decroos_2019_vaep.pdf`).
4. **Add a matching row to `data/annotations/annotations.csv`** with
   `annotator=axum` and the per-paper details Axum can read from the paper.
   The `paper_id` MUST match the `papers.csv` row exactly. Fields map 1:1 to
   slots in `configs/extraction_schema.json`. When in doubt, leave the cell
   empty rather than guessing — empty cells normalise to `null` downstream.

### Case B: brand-new papers Axum wants to add

For papers not yet in `papers.csv`:

1. **Add a row to `data/papers/papers.csv`** with bibliographic metadata.
   The `paper_id` is a lowercase slug of the form `firstauthor_year_shortname`
   (e.g. `decroos_2019_vaep`).
2. **Set `corpus_pool`** to either `candidate` (if Axum thinks the paper
   should be hand-annotated as part of the gold standard) or `held-out` (if
   it should join the extraction-agent test pool). When in doubt, default
   to `held-out` — Pegah can upgrade it.
3. **Make a first screening decision** (`include` / `exclude` / `uncertain`)
   using the criteria in `prompts/screening_agent.md` §"In-scope criteria".
   Set `screening_decision_by=axum`.
4. **Drop the PDF and set `file_path`** as above.
5. **Only if `corpus_pool=candidate`**, also add a row to
   `annotations.csv` (same details as Case A step 4). Held-out rows do NOT
   get an annotations row.

### Rules that apply to both cases

- **Flag genuinely uncertain cells** with `<PLACEHOLDER: …>` and a short
  note in the `notes` / `verification_notes` column describing what a
  verifier should resolve. **Do not invent numbers, venues, or DOIs.**
- **Rule of thumb:** if you would not stake the paper's conclusion on a
  cell, either leave it blank or mark it `<PLACEHOLDER: …>`.
- **Leave the four verification columns** (`pegah_verified`,
  `pegah_verified_date`, `david_verified`, `david_verified_date`) set to
  `false` with empty dates — those are for the verifier to flip.

---

## 1b. Pegah — first-pass annotator AND verifier

Pegah wears two hats. The hats apply to **different** papers; the same paper
cannot have Pegah as both annotator and verifier.

### As first-pass annotator (Pegah-assigned papers)

Same steps as Axum (§1a). Set `annotator=pegah` and
`screening_decision_by=pegah`. A Pegah-annotated row can only be verified by
David — Pegah cannot sign off on her own row.

### As verifier (Axum-annotated and David-annotated rows)

For each Axum-annotated row Pegah:

1. **Cross-check bibliographic data** against the PDF's first page and the
   DOI resolver. Fix anything Axum got wrong.
2. **Resolve `<PLACEHOLDER: …>` cells.** Either fill them with verified data
   or downgrade `screening_decision` to `exclude` if the source cannot be
   located.
3. **Sanity-check the schema-relevant fields** that the extraction agent will
   later try to predict — especially `model.target`, `model.supervision`,
   `model.atomicity`, `data.data_type`, `evaluation.metrics`,
   `evaluation.splits`. These are the slots where a hallucinated LLM
   extraction would do the most damage.
4. **Flip the verification flags** on the `papers.csv` row:
   - `pegah_verified=true`
   - `pegah_verified_date=YYYY-MM-DD` (today, ISO format)
   And on the matching `annotations.csv` row:
   - `verified_by_pegah=true`
   - `verified_by_pegah_date=YYYY-MM-DD`
5. **Note anything Axum should redo** in `verification_notes` rather than
   silently overwriting — the diffs are valuable training signal for the
   extraction agent's prompt.

Pegah is the **final arbiter** on disagreements between Axum and David, and
on schema-ambiguity calls.

---

## 1c. David Sumpter — verifier

David is the second independent verifier. He does **not** annotate from
scratch in Phase 1; his role is to read a paper Pegah or Axum has already
annotated, then judge whether the annotation faithfully reflects the
paper's methodology and contribution.

David's checklist is the same as Pegah's (§1b "As verifier"), with these
adjustments:

- Set `david_verified=true` and `david_verified_date=YYYY-MM-DD` on
  `papers.csv`; set `verified_by_david=true` and `verified_by_david_date=…`
  on `annotations.csv`.
- David is **not** expected to fix bibliographic-only placeholders — that's
  more Pegah's lane. David's value is on the methodological cells:
  `model.target`, `model.supervision`, `model.atomicity`,
  `data.data_type`, `evaluation.metrics`, `evaluation.tasks`,
  `evaluation.baselines`, `splits`, `key_claims`, `limitations`.
- If David disagrees with the annotator on a methodological cell, he records
  the disagreement in `verification_notes` (e.g. `"David: model.atomicity
  should be per-frame, not per-event — see §3.2"`) and Pegah arbitrates.
- Priority papers for David's pass: **the seed 7** (xT, VAEP, Atomic-VAEP,
  EPV, OBV, Goals Added, tracking-based EPV), then any paper Pegah flags as
  methodologically borderline.

---

## 1d. What the agentic pipeline will later automate (Stages 2–4)

Phase 1's hand-curated rows are the **gold standard against which everything
downstream is measured**. Once the pipeline exists, the following stops being
manual:

| Phase 1 (manual)                              | Stage 2–4 (automated)                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Axum/Pegah read each PDF cover-to-cover.      | `savant.ingestion.parse_pdf` produces section-aware text + `savant.ingestion.chunk` chops it into chunks. |
| Axum/Pegah fill `model.*`, `data.*`, `evaluation.*` cells in CSV. | `extraction_agent` produces a JSON record in `data/outputs/extractions/{paper_id}.json`, validated against `configs/extraction_schema.json`. |
| Axum/Pegah screening decision in `papers.csv`. | `screening_agent` proposes the same decision from title + abstract; disagreements with Phase 1 gold are inspected. |
| Pegah + David verify each row by hand.        | `critic_agent` checks every extraction against its source chunks and either accepts, asks for revision, or rejects; humans only spot-check. |
| Free-text values (e.g. "XGBoost", "ROC-AUC"). | `taxonomy_agent` canonicalises to controlled vocabularies for the knowledge graph. |
| Subjective "is this gap interesting?"         | `gap_agent` enumerates uncovered (method × data × evaluation) cells and ranks them; `benchmark_agent` turns the top-K into benchmark cards. |

**The CSVs do not go away when the pipeline lands.** They become:

- the **screening gold standard** the screening agent is evaluated against, and
- the **extraction gold standard** the extraction agent is evaluated against
  (rows where the gold-standard rule of §0 is satisfied).
- the **inter-annotator agreement (IAA) sample** for the manuscript: rows
  verified by both Pegah and David let us report Cohen's κ / Krippendorff's α
  on the schema fields.

Phase 1 outputs therefore directly determine what "good" means for every later
agent.

---

## 2. Quick reference — column meanings

### `data/papers/papers.csv`

| Column                       | Filled by               | Notes                                                                                          |
| ---------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------- |
| `paper_id`                   | Annotator               | Slug. MUST match `annotations.csv`.                                                            |
| `title` … `open_access`      | Annotator → Verifier(s) | Bibliographic. Mirror of `bibliographic` in `extraction_schema.json`.                          |
| `file_path`                  | Annotator               | Relative path under `data/papers/`. Empty if no PDF held locally (e.g. grey literature).        |
| `primary_tag`                | Annotator → Verifier(s) | One value from `scope.tags` enum in `extraction_schema.json`. The *primary* scope category.    |
| `screening_decision`         | Annotator → Verifier(s) | `include` / `exclude` / `uncertain`. Verifier may overrule.                                    |
| `screening_decision_by/date` | Annotator               | Who decided (`axum` / `pegah`) and when (ISO date).                                            |
| `needs_full_text`            | Annotator               | `true` if abstract alone was not enough.                                                       |
| `pegah_verified` / `_date`   | Pegah only              | Flip to `true` once Pegah signs off. Pegah cannot flip this on her own annotation rows.        |
| `david_verified` / `_date`   | David only              | Flip to `true` once David signs off.                                                           |
| `corpus_pool`                | Annotator               | Either `candidate` (hand-annotated, gold standard) or `held-out` (extraction-agent test pool, no annotation row). Required for every row. |
| `notes`                      | Anyone                  | Free-text. Anything the next reader needs to know.                                             |

### `data/annotations/annotations.csv`

Each column corresponds to a slot in `configs/extraction_schema.json`. Lists
are written as `;`-separated values (e.g. `event;tracking-2d`). Enums must
come from the schema's allowed values — see `model.family`, `model.supervision`,
`model.atomicity`, `data.data_type`, `evaluation.tasks`, `scope.tags`.

Special columns:

| Column                       | Filled by  | Notes                                                                                  |
| ---------------------------- | ---------- | -------------------------------------------------------------------------------------- |
| `annotator`                  | Annotator  | `axum` or `pegah`.                                                                     |
| `annotation_date`            | Annotator  | ISO date.                                                                              |
| `key_claims`                 | Annotator → Verifier(s) | `;`-separated; each item is `Section/Table | statement`. Section pointer is required. |
| `limitations`                | Annotator  | Only what the **authors** state, not the annotator's opinion (this is the same rule the extraction agent will follow). |
| `verified_by_pegah` / `_date` | Pegah only | Pegah's verification flag for this annotation row.                                     |
| `verified_by_david` / `_date` | David only | David's verification flag for this annotation row.                                     |
| `verification_notes`         | Verifier(s) | What was wrong, what was added, what is still uncertain. Prefix multi-verifier notes (e.g. `"David: …"`, `"Pegah: …"`) when both have something to say. |

---

## 3. How to validate your work

A loader for the corpus CSV is already in place:

```python
from savant.ingestion import load_paper_metadata
records = load_paper_metadata()  # auto-falls back to .template.csv on fresh clone
```

Run the tests to confirm the templates parse and that example records satisfy
the extraction schema:

```bash
pytest tests/test_schema.py tests/test_paper_metadata.py
```

If you add a paper whose row contains illegal enum values or a malformed date,
both the loader and the test suite will fail loudly. That is intentional — the
schema is the contract Stage 3 will rely on.

---

## 4. Exit criteria for Phase 1

Phase 1 is done when:

- [ ] Every **candidate row** in `papers.template.csv` (18 today) has a
      `papers.csv` row and a matching `annotations.csv` row that satisfies
      the gold-standard rule from §0 — i.e. at least one verifier (Pegah OR
      David) other than the annotator has signed off on both rows.
- [ ] The **original 7 seed candidates** (xT, VAEP, Atomic-VAEP, EPV, OBV,
      Goals Added, tracking-based EPV) have **both** `pegah_verified=true`
      AND `david_verified=true` — this gives the manuscript its inter-annotator
      agreement (IAA) sample on the most-cited rows.
- [ ] Every `primary_tag` that appears in the candidate pool is covered by
      ≥1 verified row (so the gold standard spans the design space, not just
      "the obvious models").
- [ ] Held-out rows do **not** need annotation rows or verification flags —
      they are pure corpus entries until Stage 3's extraction agent processes
      them.
- [ ] `pytest -q` is green.
- [ ] An at-a-glance count of `include` / `exclude` / `uncertain` rows, and
      of `candidate` / `held-out` rows, is added to the manuscript notes.

Only then should we proceed to Stage 2 (PDF ingestion) and Stage 3
(extraction agent), which together replace most of the manual work described
here.
