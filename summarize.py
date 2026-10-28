"""Summaries and report charts built from results/results.csv.

python summarize.py --results results/results.csv --out results/summary

Writes
  by_dataset.csv   one row per dataset: size, average NRMS / AE (VAEAC vs mean/mode), wins, timings
  by_pattern.csv   one row per missingness code (C, AE, AG, ...): average change vs mean/mode
  by_rate.csv      one row per missing rate (1, 5, 10, 20 %)
  pattern_types.csv  how many files step 2 classified as univariate / monotone / arbitrary
  nrms_by_dataset.png, ae_by_dataset.png, improvement_by_rate.png, complexity.png
"""
import argparse
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VAEAC, BASE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def paired_bars(table, metric, label, path, lower_better):
    """Horizontal paired bars, VAEAC vs mean/mode, one pair per dataset."""
    t = table.dropna(subset=[metric]).sort_values(metric, ascending=not lower_better)
    if t.empty:
        return
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(t) + 1.2), dpi=160)
    ax.barh(y + h / 2, t[metric], height=h - 0.04, color=VAEAC, label="VAEAC")
    ax.barh(y - h / 2, t[metric + "_meanmode"], height=h - 0.04, color=BASE, label="mean / mode")
    ax.set_yticks(y, t.index)
    ax.set_xlabel(label + (" (lower is better)" if lower_better else " (higher is better)"), color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = pd.read_csv(a.results)

    r["nrms_win"] = r.NRMS < r.NRMS_meanmode
    r["ae_win"] = r.AE > r.AE_meanmode
    r["nrms_change_pct"] = 100 * (r.NRMS_meanmode - r.NRMS) / r.NRMS_meanmode   # + = VAEAC better
    r["ae_change_pts"] = 100 * (r.AE - r.AE_meanmode)                            # + = VAEAC better

    g = r.groupby("dataset")
    by_ds = pd.DataFrame({
        "files": g.size(), "rows": g.rows.first(), "features": g.features.first(),
        "NRMS": g.NRMS.mean(), "NRMS_meanmode": g.NRMS_meanmode.mean(),
        "NRMS_wins": g.nrms_win.sum().where(g.NRMS.count() > 0),
        "AE": g.AE.mean(), "AE_meanmode": g.AE_meanmode.mean(),
        "AE_wins": g.ae_win.sum().where(g.AE.count() > 0),
        "train_s_mean": g.train_s.mean(), "impute_s_mean": g.impute_s.mean(),
        "n_params": g.n_params.first(),
    })
    by_ds.round(4).to_csv(os.path.join(a.out, "by_dataset.csv"))

    for key, name in (("pattern", "by_pattern.csv"), ("rate", "by_rate.csv")):
        k = r.groupby(key)
        pd.DataFrame({"files": k.size(),
                      "NRMS_change_pct_mean": k.nrms_change_pct.mean(),
                      "NRMS_win_rate": k.nrms_win.sum() / k.NRMS.count(),
                      "AE_change_pts_mean": k.ae_change_pts.mean(),
                      "AE_win_rate": k.ae_win.sum() / k.AE.count().replace(0, np.nan)}
                     ).round(3).to_csv(os.path.join(a.out, name))

    pd.crosstab(r.dataset, r.pattern_type).to_csv(os.path.join(a.out, "pattern_types.csv"))

    paired_bars(by_ds, "NRMS", "average NRMS", os.path.join(a.out, "nrms_by_dataset.png"), True)
    paired_bars(by_ds, "AE", "average AE", os.path.join(a.out, "ae_by_dataset.png"), False)

    # Improvement over mean/mode by missing rate (single series, so no legend).
    k = r.groupby("rate").nrms_change_pct.mean()
    fig, ax = plt.subplots(figsize=(5, 3), dpi=160)
    ax.bar([f"{int(x)}%" for x in k.index], k.values, color=VAEAC, width=0.55)
    for x, v in enumerate(k.values):
        ax.text(x, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("NRMS reduction vs mean/mode, by missing rate", fontsize=9, color=INK, loc="left")
    ax.set_ylabel("average reduction (%)", color=MUTED)
    style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "improvement_by_rate.png"))
    plt.close(fig)

    # Computational cost: two panels, one measure each (never a dual axis).
    fig, (p1, p2) = plt.subplots(1, 2, figsize=(8, 3.2), dpi=160)
    p1.scatter(r.features, 1000 * r.train_s / r.steps, s=14, color=VAEAC, alpha=0.6)
    p1.set_xlabel("features in the dataset", color=MUTED)
    p1.set_ylabel("training time per step (ms)", color=MUTED)
    p1.set_title("Training cost per mini-batch step", fontsize=9, color=INK, loc="left")
    p2.scatter(r.rows * r.features, r.impute_s, s=14, color=VAEAC, alpha=0.6)
    p2.set_xscale("log")
    p2.set_yscale("log")
    p2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    p2.set_xlabel("rows x features (log)", color=MUTED)
    p2.set_ylabel("imputation time (s, log)", color=MUTED)
    p2.set_title("Imputation time vs data size", fontsize=9, color=INK, loc="left")
    for p in (p1, p2):
        style(p)
        p.grid(axis="y", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "complexity.png"))
    plt.close(fig)
    print(f"summarised {len(r)} files from {r.dataset.nunique()} datasets -> {a.out}")


if __name__ == "__main__":
    main()
