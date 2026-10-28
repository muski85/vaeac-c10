"""
STEP 1 - Load the data.

Requirement (project description): "read some excel data ... do not limit your code to a specific
data size or data dimension".

* load_table() reads .xlsx / .xls / .csv files of any shape. The course files have no header row,
  so header=None is the default.
* detect_types() decides, column by column, whether a feature is numerical or categorical using
  ONLY the incomplete file (we never look at the original/complete data before step 4).
  A column is numerical if every observed value can be read as a number; otherwise categorical.
"""
import os

import numpy as np
import pandas as pd


def load_table(path, header=None):
    """Read an Excel or CSV file into a DataFrame. Empty cells become NaN (= missing).

    Scalability: parsing a large .xlsx (e.g. KDD, 494,020 x 41, 66 MB) takes minutes. If the
    environment variable VAEAC_CACHE names a folder, the parsed table is saved there as a pickle
    and re-used next time (the key includes the file's size and modification time)."""
    cache = os.environ.get("VAEAC_CACHE")
    if cache:
        st = os.stat(path)
        key = os.path.join(cache, f"{os.path.basename(path)}.{st.st_size}.{int(st.st_mtime)}.{header}.pkl")
        if os.path.exists(key):
            return pd.read_pickle(key)
        df = _read(path, header)
        os.makedirs(cache, exist_ok=True)
        df.to_pickle(key)
        return df
    return _read(path, header)


def _read(path, header):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, header=header)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(path, header=header)
    elif ext == ".tsv":
        df = pd.read_csv(path, header=header, sep="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Some files mark missing values with '?' or blank strings; treat them as missing too.
    df = df.replace({"?": np.nan, "": np.nan, " ": np.nan})
    return df


def detect_types(df):
    """Return a list of booleans: True = categorical column, False = numerical column."""
    is_cat = []
    for c in df.columns:
        observed = df[c].dropna()
        numeric = pd.to_numeric(observed, errors="coerce")
        is_cat.append(bool(numeric.isna().any()))  # any value that is not a number -> categorical
    return is_cat


def describe(df, is_cat):
    """Small text summary used in the console output and the report."""
    n_cat = sum(is_cat)
    return (f"{df.shape[0]} rows x {df.shape[1]} features "
            f"({df.shape[1] - n_cat} numerical, {n_cat} categorical)")
