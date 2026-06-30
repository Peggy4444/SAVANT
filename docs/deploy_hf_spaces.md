# Deploying SAVANT to Hugging Face Spaces

This guide walks through publishing the SAVANT UI to
`https://huggingface.co/spaces/<your-username>/savant`. Total time: ~15
minutes; recurring effort: `git push` after any code change.

## Why Hugging Face Spaces

- Free for public Spaces.
- Native Streamlit support — no Docker required.
- Discoverable by ML researchers (the audience SAVANT targets).
- Built-in social features (likes, comments, fork-to-duplicate).
- No environment management — `requirements.txt` is enough.

## One-time setup

### 1. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Name: `savant` (or any URL slug you prefer).
3. License: **MIT**.
4. SDK: **Streamlit**.
5. Hardware: **CPU basic** (free tier — sufficient; no GPU needed).
6. Visibility: **Public** (or Private if you want to gate access).
7. Click **Create Space**. It initialises an empty git repo at
   `huggingface.co/spaces/<your-username>/savant`.

### 2. Push the SAVANT repo into the Space

The Space metadata HF needs goes in `README_HF.md` (already in our repo).
HF Spaces requires that metadata to be in the file named `README.md`. To
avoid clobbering our project README, we maintain both:

- `README.md` — the GitHub repo's README (audience: developers).
- `README_HF.md` — the HF Space metadata + landing copy.

When you push to HF Spaces, copy `README_HF.md` to `README.md` IN THE
HF REMOTE ONLY. Two approaches:

#### Option A — separate remote (recommended)

```bash
# Add HF as a second remote
git remote add hf https://huggingface.co/spaces/<your-username>/savant

# One-time: prepare a deploy branch where README.md is the HF version
git checkout -b hf-deploy
cp README_HF.md README.md
git add README.md
git commit -m "deploy: HF Spaces README"

# Push to HF
git push hf hf-deploy:main

# Keep main branch unchanged for GitHub
git checkout main
```

After this, the HF Space rebuilds automatically. Every subsequent deploy:

```bash
git checkout main
# … make your changes, commit on main …
git checkout hf-deploy
git merge main                           # pull in the new changes
cp README_HF.md README.md && git add README.md && git commit --amend --no-edit
git push hf hf-deploy:main --force-with-lease
git checkout main
```

#### Option B — single repo with a build-time README swap

In `README_HF.md` add a tiny `pre-build.sh` that HF runs before
the Streamlit launch. But HF doesn't reliably support arbitrary
pre-build scripts for Streamlit SDKs — stick to Option A.

### 3. Verify the build

HF Spaces will run:
1. `pip install -r requirements.txt`
2. Read the YAML frontmatter from `README.md`
3. Start `streamlit run streamlit_app.py` on port 7860

Watch the build logs in the Space's "Logs" tab. First build takes ~3-5
minutes (Pip is installing everything in `requirements.txt`, including
`torch` and `xgboost`).

If the build fails, the most common causes:

- **`torch` install timeout** — comment out `torch` and `xgboost` in
  `requirements.txt`; the UI does not need them (they're for the
  Stage-6 benchmark, which doesn't ship with the UI).
- **`PyMuPDF` install failure** — already pinned to a binary wheel
  version; should be fine. If not, swap to `pdfminer.six` only.
- **Memory limit** — free CPU tier has 16 GB. The UI keeps the KG in
  memory (~5 MB), so this is generously sufficient.

### 4. Set Space secrets (optional)

The Live Demo and Chat pages use **user-supplied** API keys typed into
the UI — they do NOT require any Space-level secret to function.

If you want the demo to work without users supplying their own key (e.g.
for a private Space you control the audience of), add a secret:

1. Go to your Space → **Settings** → **Variables and secrets**.
2. Add `ANTHROPIC_API_KEY` as a **Secret** (not Variable).
3. The UI's `AnthropicClient` will read it automatically when the user
   leaves the key input empty.

**Do not do this for a public Space** — anyone could rack up your bill.

## Recurring deploy

Once Option A is set up, the flow per change is:

```bash
git checkout main
# … edit code, commit on main …
git checkout hf-deploy
git merge main
cp README_HF.md README.md && git add README.md && git commit --amend --no-edit
git push hf hf-deploy:main --force-with-lease
git checkout main
```

HF Spaces rebuilds on push automatically.

## What to expect at the URL

The full 9-page UI mounts at
`https://huggingface.co/spaces/<your-username>/savant`, with the SAVANT
hero on the landing page. First load takes a few seconds while Streamlit
warms; subsequent navigation is instant (cached data loaders).

## Slimming the deploy

If the Space build is too slow, trim `requirements.txt` to the UI-only
subset. The full set is needed for tests and the agentic pipeline, but the
UI itself only requires:

```
streamlit>=1.30
plotly>=5.20
matplotlib>=3.8
networkx>=3.1
pymupdf>=1.24
pandas>=2.2
numpy>=1.26
pyyaml>=6.0
jsonschema>=4.21
pydantic>=2.6
anthropic>=0.40
python-dotenv>=1.0
pyvis>=0.3.2
beautifulsoup4>=4.12
lxml>=5.1
```

Drop `torch`, `xgboost`, `lightgbm`, `scipy`, `sentence-transformers`,
`scikit-learn`, `tenacity`, `statsbombpy`, `mplsoccer`, `rdflib`,
`pdfminer.six`, plus all dev tooling — none are touched by the UI.
