"""Re-score saved imputed files without re-running the imputation.

Used after fixing the AE comparison (the Splice ORIGINAL file stores " G" with a leading space,
the incomplete files store "G", so every categorical cell looked wrong). For each file of the given
datasets it reloads incomplete / original / imputed, recomputes NRMS, AE and the mean/mode baseline,
and updates those columns in results.csv.

python recompute_ae.py --data_root "<course folder>" --results results/results.csv --imputed results/imputed --datasets Splice
"""
import argparse
import os

import pandas as pd

import step1_load as s1
import step4_evaluate as s4

ap = argparse.ArgumentParser()
ap.add_argument("--data_root", required=True)
ap.add_argument("--results", default="results/results.csv")
ap.add_argument("--imputed", default="results/imputed")
ap.add_argument("--datasets", nargs="+", required=True)
a = ap.parse_args()

res = pd.read_csv(a.results)
for ds in a.datasets:
    original = s1.load_table(os.path.join(a.data_root, "Original Datasets Without Labels", ds + ".xlsx"))
    for i in res.index[res["dataset"] == ds]:
        name = res.at[i, "file"]
        incomplete = s1.load_table(os.path.join(a.data_root, "Incomplete Datasets Without Labels", ds, name + ".xlsx"))
        imputed = s1.load_table(os.path.join(a.imputed, name + "_imputed.xlsx"))
        is_cat = s1.detect_types(incomplete)
        baseline = s4.mean_mode_impute(incomplete, is_cat)
        before = res.at[i, "AE"]
        res.at[i, "NRMS"] = s4.nrms(original, imputed, is_cat)
        res.at[i, "AE"] = s4.ae(original, imputed, incomplete, is_cat)
        res.at[i, "NRMS_meanmode"] = s4.nrms(original, baseline, is_cat)
        res.at[i, "AE_meanmode"] = s4.ae(original, baseline, incomplete, is_cat)
        print(f"{name:16s} AE {before} -> {res.at[i, 'AE']:.4f}   mean/mode AE {res.at[i, 'AE_meanmode']:.4f}", flush=True)
res.to_csv(a.results, index=False)
print("updated", a.results)
