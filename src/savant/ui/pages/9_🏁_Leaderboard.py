"""SAVANT UI — Track-B leaderboard.

Renders the leaderboard produced by:

    PYTHONPATH=src python scripts/run_benchmark.py
    PYTHONPATH=src python scripts/evaluate_benchmark.py

That pair produces:
    data/outputs/benchmark/leaderboard.csv
    data/outputs/benchmark/per_task/t3_pearson_bootstrap.csv
    data/outputs/benchmark/per_task/t3_wilcoxon_pairwise.csv
    data/outputs/benchmark/run_manifest.json
    data/outputs/benchmark/evaluation_manifest.json

If those files don't exist yet (fresh clone, HF Spaces without a run),
the page shows an empty-state panel with the exact commands to run.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from savant.ui.components import (
    PALETTE, apply_theme, badges, footer, hero, section_eyebrow,
)
from savant.ui.data import (
    load_evaluation_manifest, load_leaderboard, load_run_manifest,
    load_t1_bootstrap, load_t2_bootstrap, load_t3_bootstrap,
    load_t3_per_competition, load_t3_wilcoxon, load_t4_delta,
)


st.set_page_config(
    page_title="SAVANT — Track-B leaderboard",
    page_icon="🏁",
    layout="wide",
)
apply_theme()


hero(
    eyebrow="Track B · Empirical benchmark",
    title="Nine methods, one open dataset, one leaderboard.",
    subtitle=(
        "T3: does the method's team-match aggregate value correlate with the "
        "StatsBomb-native xG differential of that match? All nine corpus "
        "methods plus three trivial baselines, evaluated on the same "
        "match-level split with match-level bootstrap 95% CIs and pairwise "
        "Wilcoxon signed-rank testing (Benjamini-Hochberg corrected)."
    ),
)


leaderboard = load_leaderboard()
bootstrap = load_t3_bootstrap()
wilcoxon = load_t3_wilcoxon()
t1_bootstrap = load_t1_bootstrap()
t2_bootstrap = load_t2_bootstrap()
t3_per_comp = load_t3_per_competition()
t4_delta = load_t4_delta()
run_manifest = load_run_manifest()
eval_manifest = load_evaluation_manifest()


# --- Empty state ---------------------------------------------------


if leaderboard is None:
    st.info(
        "**No benchmark run found under `data/outputs/benchmark/`.**\n\n"
        "This page renders artefacts produced by the Track-B benchmark. "
        "Run the pair below to populate it — the first command builds "
        "per-method predictions, the second computes the leaderboard.",
    )
    st.code(
        "PYTHONPATH=src python scripts/run_benchmark.py --max-matches 60\n"
        "PYTHONPATH=src python scripts/evaluate_benchmark.py --max-matches 60",
        language="bash",
    )
    footer()
    st.stop()


# --- Run-context badges ------------------------------------------


badge_items: list[tuple[str, str]] = []
if run_manifest is not None:
    n_matches = int(run_manifest.get("n_matches_total", 0))
    seed = int(run_manifest.get("split_seed", 0))
    audit = run_manifest.get("audit", {})
    n_test = int(audit.get("n_matches_test", 0)) if isinstance(audit, dict) else 0
    n_train = int(audit.get("n_matches_train", 0)) if isinstance(audit, dict) else 0
    badge_items.extend([
        (f"{n_matches} matches",   "primary"),
        (f"train / test {n_train} / {n_test}", "info"),
        (f"seed {seed}",           "neutral"),
    ])
if eval_manifest is not None:
    n_boot = int(eval_manifest.get("n_bootstrap", 0))
    badge_items.append((f"bootstrap n={n_boot}", "accent"))
if badge_items:
    badges(badge_items)


# --- Small-sample warning ----------------------------------------


if run_manifest is not None and int(run_manifest.get("n_matches_total", 0)) < 50:
    st.warning(
        "**Small-sample warning.** This benchmark run covers "
        f"{run_manifest.get('n_matches_total', '?')} matches. Bootstrap CIs "
        "will be wide and rankings unstable. Re-run without `--max-matches` "
        "for the full corpus (843 matches across 7 training competitions).",
    )


# --- Leaderboard table -------------------------------------------


section_eyebrow("Ranked by T3 Pearson ρ")

st.markdown(
    "The primary T3 metric is **Pearson ρ** between per-method team-match "
    "values and the StatsBomb-native team match-xG differential. Higher is "
    "better. Bootstrap CIs are computed on match-level resamples (both "
    "team-rows of a sampled match are kept together, preserving their "
    "correlation)."
)

# Present it as a styled dataframe. Highlight top 3 by ρ point estimate.
lb = leaderboard.copy()
lb.insert(0, "rank", range(1, len(lb) + 1))
lb["pearson_rho_95_ci"] = lb.apply(
    lambda r: f"[{r['pearson_rho_lo']:.3f}, {r['pearson_rho_hi']:.3f}]",
    axis=1,
)
display_cols = [
    "rank", "model_id",
    "pearson_rho_point", "pearson_rho_95_ci",
    "spearman_rho_point", "mae_point",
]
rename = {
    "rank":                "#",
    "model_id":            "method",
    "pearson_rho_point":   "Pearson ρ",
    "pearson_rho_95_ci":   "95% CI (bootstrap)",
    "spearman_rho_point":  "Spearman ρ",
    "mae_point":           "MAE",
}
format_map = {
    "Pearson ρ":   "{:.3f}",
    "Spearman ρ":  "{:.3f}",
    "MAE":         "{:.3f}",
}
# When per-competition data is available, surface robustness alongside the
# pooled ρ — the paper's "which method generalises" answer lives here.
if "robustness_std_rho" in lb.columns and lb["robustness_std_rho"].notna().any():
    display_cols += ["robustness_std_rho", "n_competitions"]
    rename["robustness_std_rho"] = "robustness (std ρ)"
    rename["n_competitions"]     = "n_comp"
    format_map["robustness (std ρ)"] = "{:.3f}"

display = lb[display_cols].rename(columns=rename)
styled = display.style.format(format_map).highlight_max(
    subset=["Pearson ρ", "Spearman ρ"], color=PALETTE["primary"] + "22",
)
if "robustness (std ρ)" in display.columns:
    # For std, LOWER is better — highlight the min.
    styled = styled.highlight_min(
        subset=["robustness (std ρ)"], color=PALETTE["primary"] + "22",
    )
st.dataframe(styled, hide_index=True, use_container_width=True)


# --- Bootstrap CI forest plot -----------------------------------


if bootstrap is not None and len(bootstrap):
    section_eyebrow("Bootstrap 95% CIs on Pearson ρ")

    bs = bootstrap.sort_values("pearson_rho_point", ascending=True).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bs["pearson_rho_point"],
        y=bs["model_id"],
        mode="markers",
        marker=dict(color=PALETTE["primary"], size=11,
                    line=dict(color=PALETTE["primary_dk"], width=1.5)),
        error_x=dict(
            type="data",
            symmetric=False,
            array=(bs["pearson_rho_hi"] - bs["pearson_rho_point"]).clip(lower=0),
            arrayminus=(bs["pearson_rho_point"] - bs["pearson_rho_lo"]).clip(lower=0),
            color=PALETTE["ink_2"],
            thickness=1.5,
            width=6,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ρ = %{x:.3f}<br>"
            "95%% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
            "<extra></extra>"
        ),
        customdata=list(zip(bs["pearson_rho_lo"], bs["pearson_rho_hi"])),
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=PALETTE["ink_3"],
                  annotation_text="null (no correlation)",
                  annotation_position="top right")
    fig.update_layout(
        height=max(360, 32 * len(bs)),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis_title="Pearson ρ",
        yaxis_title="",
        plot_bgcolor=PALETTE["surface"],
        paper_bgcolor=PALETTE["surface"],
        font=dict(family="Inter", color=PALETTE["ink"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Wilcoxon pairwise heatmap ----------------------------------


if wilcoxon is not None and len(wilcoxon):
    section_eyebrow("Pairwise significance (Wilcoxon signed-rank, BH-corrected)")

    st.markdown(
        "Each cell is the BH-adjusted p-value from a paired Wilcoxon "
        "signed-rank test on per-match absolute errors. Darker green = "
        "more significant difference between the two methods. Empty "
        "cells (self-pairs) are grey."
    )

    import numpy as np

    methods = sorted(set(wilcoxon["method_a"]) | set(wilcoxon["method_b"]))
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for _, r in wilcoxon.iterrows():
        matrix.loc[r["method_a"], r["method_b"]] = r["p_value_bh"]
        matrix.loc[r["method_b"], r["method_a"]] = r["p_value_bh"]

    # -log10 so smaller p → darker (more significant). Cap very small p's at
    # 1e-12 to keep the colour scale bounded.
    m_arr = matrix.astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        clipped = np.where(m_arr <= 0, 1e-12, m_arr)
        log_p = -np.log10(clipped)
    log_p = np.where(np.isnan(m_arr), np.nan, log_p)

    fig = go.Figure(data=go.Heatmap(
        z=log_p,
        x=methods,
        y=methods,
        # rgba() rather than 8-char hex — plotly's newer validator rejects
        # the shorthand `#RRGGBBAA` form.
        colorscale=[[0.0, PALETTE["surface_2"]],
                    [0.5, "rgba(6, 167, 125, 0.5)"],
                    [1.0, PALETTE["primary_dk"]]],
        zmin=0, zmax=4,     # p=1 → 0, p=0.05 → 1.3, p=0.001 → 3
        showscale=True,
        colorbar=dict(title="−log₁₀(p_BH)"),
        hovertemplate=(
            "%{y} vs %{x}<br>"
            "p_BH = %{customdata:.4f}"
            "<extra></extra>"
        ),
        customdata=m_arr,
    ))
    fig.update_layout(
        height=max(500, 32 * len(methods)),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(tickangle=-45),
        plot_bgcolor=PALETTE["surface"],
        paper_bgcolor=PALETTE["surface"],
        font=dict(family="Inter", color=PALETTE["ink"]),
    )
    st.plotly_chart(fig, use_container_width=True)


# --- T1 pane ------------------------------------------------------


if t1_bootstrap is not None and len(t1_bootstrap):
    section_eyebrow("T1 — per-action goal prediction")
    st.markdown(
        "Each method's per-action value is calibrated with **isotonic "
        "regression fit on the training split** and evaluated on the "
        "held-out matches. Target: **binary — did the same team score in "
        "the next 10 SPADL actions**. Brier / log-loss / ECE lower = "
        "better; ROC-AUC higher = better. Bootstrap resamples matches, so "
        "the CI collapses when only one test match is available."
    )
    t1 = t1_bootstrap.copy()
    t1.insert(0, "rank", range(1, len(t1) + 1))
    t1["brier_95_ci"] = t1.apply(
        lambda r: f"[{r['brier_lo']:.4f}, {r['brier_hi']:.4f}]"
                  if pd.notna(r['brier_lo']) else "—",
        axis=1,
    )
    show = t1[[
        "rank", "model_id",
        "brier_point", "brier_95_ci",
        "log_loss_binary_point", "ece_point", "roc_auc_point",
    ]].rename(columns={
        "rank":                     "#",
        "model_id":                 "method",
        "brier_point":              "Brier",
        "brier_95_ci":              "95% CI (bootstrap)",
        "log_loss_binary_point":    "log loss",
        "ece_point":                "ECE",
        "roc_auc_point":            "ROC-AUC",
    })
    st.dataframe(
        show.style.format({
            "Brier":    "{:.4f}",
            "log loss": "{:.4f}",
            "ECE":      "{:.4f}",
            "ROC-AUC":  "{:.3f}",
        }).highlight_min(subset=["Brier", "log loss", "ECE"],
                         color=PALETTE["primary"] + "22")
           .highlight_max(subset=["ROC-AUC"],
                          color=PALETTE["primary"] + "22"),
        hide_index=True,
        use_container_width=True,
    )


# --- T2 pane ------------------------------------------------------


if t2_bootstrap is not None and len(t2_bootstrap):
    section_eyebrow("T2 — multiclass possession outcome")
    st.markdown(
        "Possessions are segmented by team changes / period boundaries and "
        "terminated as `goal · no_goal · turnover · foul`. Each method's "
        "per-action value is fed through a multinomial LogisticRegression "
        "**fit on the training split**, projecting V(a) → a 4-way probability "
        "distribution over outcomes on the test split. Uniform prior over 4 "
        "classes gives log-loss ≈ 1.386 (ln 4) — anything lower means the "
        "method's value carries real signal about how the possession will "
        "end. log-loss / ECE lower = better; top-1 accuracy higher = better."
    )
    t2 = t2_bootstrap.copy()
    t2.insert(0, "rank", range(1, len(t2) + 1))
    t2["log_loss_95_ci"] = t2.apply(
        lambda r: f"[{r['log_loss_multiclass_lo']:.3f}, {r['log_loss_multiclass_hi']:.3f}]"
                  if pd.notna(r['log_loss_multiclass_lo']) else "—",
        axis=1,
    )
    show = t2[[
        "rank", "model_id",
        "log_loss_multiclass_point", "log_loss_95_ci",
        "top1_accuracy_point", "ece_multiclass_point",
    ]].rename(columns={
        "rank":                          "#",
        "model_id":                      "method",
        "log_loss_multiclass_point":     "log loss",
        "log_loss_95_ci":                "95% CI (bootstrap)",
        "top1_accuracy_point":           "top-1 acc",
        "ece_multiclass_point":          "ECE",
    })
    st.dataframe(
        show.style.format({
            "log loss":  "{:.3f}",
            "top-1 acc": "{:.3f}",
            "ECE":       "{:.3f}",
        }).highlight_min(subset=["log loss", "ECE"],
                         color=PALETTE["primary"] + "22")
           .highlight_max(subset=["top-1 acc"],
                          color=PALETTE["primary"] + "22"),
        hide_index=True,
        use_container_width=True,
    )


# --- T3 per-competition matrix ------------------------------------


if t3_per_comp is not None and len(t3_per_comp):
    section_eyebrow("T3 per-competition — where does each method work?")
    st.markdown(
        "Pooled ρ conflates method quality with competition-level task "
        "difficulty (tournaments give small per-team samples; round-robin "
        "leagues give many). This matrix shows ρ per (method × competition) "
        "so the paper's robustness claim is directly readable. Lower "
        "row-wise std = more robust across styles + samples."
    )

    # Sort methods by their pooled ρ so it visually aligns with the top table.
    if bootstrap is not None:
        method_order = list(bootstrap.sort_values(
            "pearson_rho_point", ascending=False,
        )["model_id"])
    else:
        method_order = sorted(t3_per_comp["model_id"].unique())

    # Sort competitions: train first (in the order they appear in the CSV), then holdout.
    comp_order = t3_per_comp.drop_duplicates("competition").sort_values(
        ["corpus", "index"], ascending=[False, True],   # 'train' > 'holdout'
    )["competition"].tolist()

    matrix = t3_per_comp.pivot_table(
        index="model_id", columns="competition",
        values="pearson_rho", aggfunc="first",
    ).reindex(index=method_order, columns=comp_order)

    # Which competitions are train vs holdout (for annotations).
    train_labels = set(t3_per_comp[t3_per_comp["corpus"] == "train"]["competition"])

    # Plotly heatmap: diverging colour ramp centred on 0.
    import numpy as np
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=[f"{c}{'  (holdout)' if c not in train_labels else ''}" for c in matrix.columns],
        y=matrix.index,
        colorscale="RdYlGn",
        zmin=-0.7, zmax=0.9,
        zmid=0,
        colorbar=dict(title="ρ"),
        hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.3f}<extra></extra>",
        text=matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Inter"},
    ))
    fig.update_layout(
        height=max(360, 40 * len(matrix)),
        margin=dict(l=8, r=8, t=32, b=8),
        xaxis=dict(tickangle=-30, side="top"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor=PALETTE["surface"],
        paper_bgcolor=PALETTE["surface"],
        font=dict(family="Inter", color=PALETTE["ink"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # A short auto-observation panel — surfaces the finding we found in C.
    comp_means = t3_per_comp.groupby("competition")["pearson_rho"].mean()
    hardest = comp_means.idxmin()
    easiest = comp_means.idxmax()
    st.markdown(
        f"**Range across competitions:** mean ρ = "
        f"**{comp_means.min():+.2f}** ({hardest}) → **{comp_means.max():+.2f}** ({easiest}). "
        "Tournaments (small samples per team) systematically produce lower ρ than "
        "round-robin leagues, regardless of method. When reading T4, adjust "
        "expectations by holdout difficulty."
    )


# --- T4 pane ------------------------------------------------------


if t4_delta is not None and len(t4_delta) and bootstrap is not None:
    section_eyebrow("T4 — cross-competition robustness")
    st.markdown(
        "Each method's T3 Pearson ρ was recomputed on the two **holdout** "
        "competitions from `configs/benchmark_tasks.yaml` (WSL 2020/21 + "
        "AFCON 2023 — 183 matches, entirely disjoint from the training "
        "corpus). `Δρ = ρ_train − ρ_holdout`. A method with |Δρ| ≈ 0 "
        "transfers cleanly across leagues and styles; a large |Δρ| in "
        "either direction indicates the method's ρ depended on the training "
        "distribution."
    )

    # Join T4 with T3 train points to compute holdout ρ inline.
    t3_train = bootstrap.set_index("model_id")["pearson_rho_point"]
    t4 = t4_delta.copy().set_index("model_id")
    t4["ρ_train"]   = t3_train
    t4["ρ_holdout"] = t4["ρ_train"] - t4["pearson_rho_delta"]
    t4["abs_delta"] = t4["pearson_rho_delta"].abs()
    t4 = t4.reset_index().sort_values("abs_delta").reset_index(drop=True)
    t4.insert(0, "#", range(1, len(t4) + 1))

    show = t4[["#", "model_id", "ρ_train", "pearson_rho_delta", "ρ_holdout"]].rename(
        columns={"model_id": "method", "pearson_rho_delta": "Δρ (train − holdout)"},
    )
    st.dataframe(
        show.style.format({
            "ρ_train":              "{:+.3f}",
            "Δρ (train − holdout)": "{:+.3f}",
            "ρ_holdout":            "{:+.3f}",
        }).background_gradient(
            subset=["Δρ (train − holdout)"],
            cmap="RdYlGn_r",
            vmin=-0.25, vmax=0.25,   # symmetric colour scale around zero
        ),
        hide_index=True,
        use_container_width=True,
    )

    # Callout: negative Δρ = method does BETTER on holdout — worth naming.
    n_better_on_holdout = int((t4["pearson_rho_delta"] < 0).sum())
    if n_better_on_holdout > len(t4) / 2:
        st.info(
            f"**Note on sign.** {n_better_on_holdout}/{len(t4)} methods "
            "score *higher* ρ on the holdout leagues than on the training "
            "corpus (`Δρ < 0`). Not overfitting — the holdout leagues "
            "(WSL, AFCON) have fewer shots per match, so team-level xG "
            "differential is more separable. The training corpus is dragged "
            "down by 380 matches of La Liga 2015/16, which has denser "
            "possession play. Report signed Δρ alongside |Δρ| in any "
            "manuscript."
        )


# --- Reproducibility footer -------------------------------------


section_eyebrow("Reproduce")

st.markdown(
    "Every leaderboard row is traceable back to a corpus paper. Every method "
    "id in the `method` column maps 1:1 to a `paper_id` under "
    "`data/outputs/extractions/`. To re-run the full benchmark:"
)
st.code(
    "PYTHONPATH=src python scripts/run_benchmark.py           # ~30 min\n"
    "PYTHONPATH=src python scripts/evaluate_benchmark.py",
    language="bash",
)

with st.expander("Per-method run manifest"):
    if run_manifest is not None:
        m_rows = []
        for section in ("baselines", "methods"):
            for mid, info in (run_manifest.get(section) or {}).items():
                m_rows.append({
                    "kind":            section[:-1],
                    "id":              mid,
                    "n_predictions":   info.get("n_predictions"),
                    "n_games":         info.get("n_distinct_games"),
                    "n_teams":         info.get("n_distinct_teams"),
                    "duration_s":      info.get("duration_s"),
                })
        if m_rows:
            st.dataframe(pd.DataFrame(m_rows), hide_index=True,
                         use_container_width=True)


footer()
