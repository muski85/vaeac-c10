"""
STEP 2 - Identify the missing data: its ratio and its pattern.

Requirement (project description): "discover the pattern and ratio of missing data ... (univariate,
monotone, arbitrary missing data)".

Definitions used (Little & Rubin, Statistical Analysis with Missing Data):
* univariate - all missing values are in a single feature (column).
* monotone   - the features can be ordered Y1, ..., Yk so that whenever Yj is missing in a row,
               every later feature Yj+1, ..., Yk is missing in that row too. Equivalently, the sets
               of rows missing each feature are nested inside each other.
* arbitrary  - any other arrangement (also called "general" missingness).
"""
import numpy as np
import pandas as pd


def missing_mask(df):
    """Boolean matrix, True where a value is missing."""
    return df.isna().values


def classify_pattern(mask):
    """Return 'none', 'univariate', 'monotone' or 'arbitrary' for a boolean missing mask."""
    cols = np.where(mask.any(axis=0))[0]           # features that have at least one gap
    if len(cols) == 0:
        return "none"
    if len(cols) == 1:
        return "univariate"
    # Order features from fewest to most missing values. For a monotone pattern the set of rows
    # missing feature j must be contained in the set of rows missing the next feature.
    order = cols[np.argsort(mask[:, cols].sum(axis=0), kind="stable")]
    for a, b in zip(order[:-1], order[1:]):
        rows_a, rows_b = mask[:, a], mask[:, b]
        if np.any(rows_a & ~rows_b):               # a row misses a but not b -> not nested
            return "arbitrary"
    return "monotone"


def summarise(df, is_cat=None):
    """Ratios and counts that describe the missingness of one dataset."""
    m = missing_mask(df)
    n, d = m.shape
    per_row = m.sum(axis=1)
    distinct_patterns = len({row.tobytes() for row in m[per_row > 0]})
    out = {
        "rows": n,
        "features": d,
        "missing_values": int(m.sum()),
        "missing_ratio_pct": round(100 * m.mean(), 2),
        "rows_with_missing_pct": round(100 * (per_row > 0).mean(), 2),
        "complete_rows": int((per_row == 0).sum()),
        "features_with_missing": int(m.any(axis=0).sum()),
        "max_missing_in_a_row": int(per_row.max()) if n else 0,
        "distinct_row_patterns": distinct_patterns,
        "pattern_type": classify_pattern(m),
    }
    if is_cat is not None:
        cat = np.array(is_cat, dtype=bool)
        out["missing_numerical"] = int(m[:, ~cat].sum())
        out["missing_categorical"] = int(m[:, cat].sum())
    return out


def per_feature(df):
    """Missing count and ratio for every feature (column)."""
    m = missing_mask(df)
    return pd.DataFrame({"feature": list(df.columns),
                         "missing": m.sum(axis=0),
                         "missing_pct": np.round(100 * m.mean(axis=0), 2)})


def plot_mask(df, path, title=""):
    """Save a picture of the missing-value matrix (rows x features, dark = missing)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = missing_mask(df)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.imshow(m, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_xlabel("feature")
    ax.set_ylabel("row")
    ax.set_title(title or "Missing values (dark = missing)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
