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
        hits += int((original[c].astype(str).values[miss] == imputed[c].astype(str).values[miss]).sum())
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
    """Average metric per missing rate, VAEAC vs baseline, one line pair per dataset."""
    plt = _plt()
    r = results.dropna(subset=[metric])
    if r.empty:
        return False
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for ds, g in r.groupby("dataset"):
        a = g.groupby("rate")[[metric, metric + "_meanmode"]].mean()
        line, = ax.plot(a.index, a[metric], marker="o", label=f"{ds} VAEAC")
        ax.plot(a.index, a[metric + "_meanmode"], marker="o", linestyle="--",
                color=line.get_color(), label=f"{ds} mean/mode")
    ax.set_xlabel("missing rate (%)")
    ax.set_ylabel(metric + (" (lower is better)" if metric == "NRMS" else " (higher is better)"))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
