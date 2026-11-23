"""
STEP 4 - Return the imputed data and compare it with the complete (original) data.

Metrics from the course Instructions.pdf:
* NRMS = ||X_original - X_imputed||_F / ||X_original||_F   over the numerical features
* AE   = (1/n) * sum_i I(imputed_i == true_i)               over the categorical features
  We take n = number of imputed (missing) categorical entries, so AE is the share of missing
  categories that were filled correctly.

A mean/mode baseline (column mean for numbers, most frequent value for categories) is scored on
exactly the same missing cells so we can see whether VAEAC is worth its cost.
"""
import numpy as np
import pandas as pd


def nrms(original, imputed, is_cat):
    num = [c for j, c in enumerate(original.columns) if not is_cat[j]]
    if not num:
        return None
    A = original[num].apply(pd.to_numeric).values.astype(float)
    B = imputed[num].apply(pd.to_numeric).values.astype(float)
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))


def ae(original, imputed, incomplete, is_cat):
    hits, n = 0, 0
    for j, c in enumerate(original.columns):
        if not is_cat[j]:
            continue
        miss = incomplete[c].isna().values
        n += int(miss.sum())
        truth = original[c].astype(str).str.strip().values[miss]      # strip: ignore stray spaces
        guess = imputed[c].astype(str).str.strip().values[miss]
        hits += int((truth == guess).sum())
    return None if n == 0 else hits / n


def mean_mode_impute(incomplete, is_cat):
    out = incomplete.copy()
    for j, c in enumerate(incomplete.columns):
        if out[c].notna().sum() == 0:
            continue
        if is_cat[j]:
            out[c] = out[c].fillna(out[c].mode().iloc[0])
        else:
            out[c] = pd.to_numeric(out[c]).fillna(pd.to_numeric(out[c]).mean())
    return out


def check_observed_unchanged(incomplete, imputed, is_cat):
    """Sanity check: imputation must not alter observed values, and must leave no gaps."""
    assert not imputed.isna().any().any(), "imputed data still has missing values"
    for j, c in enumerate(incomplete.columns):
        obs = incomplete[c].notna().values
        a, b = incomplete[c].values[obs], imputed[c].values[obs]
        if is_cat[j]:
            assert (a.astype(str) == b.astype(str)).all(), f"observed values changed in column {c}"
        else:
            assert np.allclose(a.astype(float), b.astype(float), atol=1e-6), f"observed changed in {c}"


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training(history, path, title=""):
    """Training loss and validation IWAE across epochs."""
    plt = _plt()
    ep = [h[0] for h in history]
    fig, ax1 = plt.subplots(figsize=(6, 3.5), dpi=150)
    ax1.plot(ep, [h[1] for h in history], color="#C46A1B", label="training loss (per feature)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h[2] for h in history], color="#2A6F97", label="validation IWAE")
    ax2.set_ylabel("validation IWAE (higher is better)")
    ax1.set_title(title)
    fig.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_imputed_vs_true(original, imputed, incomplete, is_cat, path, title=""):
    """For numerical features: imputed value against the true value for every missing cell."""
    plt = _plt()
    xs, ys = [], []
    for j, c in enumerate(original.columns):
        if is_cat[j]:
            continue
        miss = incomplete[c].isna().values
        if miss.any():
            o = pd.to_numeric(original[c]).values.astype(float)
            s = o.std() or 1.0
            xs.append((o[miss] - o.mean()) / s)
            ys.append((pd.to_numeric(imputed[c]).values.astype(float)[miss] - o.mean()) / s)
    if not xs:
        return False
    x, y = np.concatenate(xs), np.concatenate(ys)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.scatter(x, y, s=8, alpha=0.6, color="#2A6F97")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color="#14213D", linewidth=1)
    ax.set_xlabel("true value (standardised)")
    ax.set_ylabel("imputed value (standardised)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_rate_curves(results, path, metric="NRMS"):
    """Small multiples: one panel per dataset, VAEAC vs mean/mode by missing rate.

    Each panel shows the metric averaged over that dataset's 11 patterns at each rate.
    Panels have their own y-scale, so datasets with small values are still readable.
    """
    import math
    plt = _plt()
    r = results.dropna(subset=[metric])
    if r.empty:
        return False
    blue, orange, ink, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
    datasets = sorted(r["dataset"].unique(), key=str.lower)
    cols = 5 if len(datasets) > 8 else min(len(datasets), 4)
    rows = math.ceil(len(datasets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows + 0.9), dpi=150, squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        a = r[r["dataset"] == ds].groupby("rate")[[metric, metric + "_meanmode"]].mean().sort_index()
        x = [str(int(v)) for v in a.index]
        ax.plot(x, a[metric], color=blue, linewidth=2, marker="o", markersize=5, label="VAEAC")
        ax.plot(x, a[metric + "_meanmode"], color=orange, linewidth=2, linestyle="--", marker="o",
                markersize=5, label="mean/mode")
        ax.set_title(ds, fontsize=10, color=ink, loc="left")
        ax.tick_params(colors=muted, labelsize=8)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("missing rate (%)", fontsize=8, color=muted)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric, fontsize=8, color=muted)
    better = "lower is better" if metric == "NRMS" else "higher is better"
    fig.suptitle(f"{metric} by missing rate, per dataset ({better}; average over the 11 patterns)",
                 fontsize=12, color=ink, x=0.01, ha="left", y=0.995)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return True
