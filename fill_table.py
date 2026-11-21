"""Copy NRMS / AE results from results.csv into a copy of the course Table-NRMS-AE.xlsx.

python fill_table.py --table "Table-NRMS-AE.xlsx" --results results/results.csv --out Table-NRMS-AE_C10.xlsx

Also fills the two summary rows at the bottom of the course table:
  "Average NRMS/AE"      mean NRMS over every numerical result, mean AE over every categorical result
  "Complexity Analysis"  a one-line summary of the cost (full analysis is in the report)
"""
import argparse

import openpyxl
import pandas as pd

# The table spells a few dataset names differently from the folder names.
ALIAS = {"Bupa": "BUPA", "TTTTEG": "TTTEG", "CREDIT": "Credit"}

COMPLEXITY = ("Training O(E*n*(L*h^2 + d*h)), imputation O(n*K*(L*h^2 + d*h)): linear in rows n and "
              "features d, quadratic in hidden width h. Here h=256, L=4 layers, K=25 samples, about 1.2-1.6M "
              "parameters (~1.2M multiply-adds per row). Large data: train on a 10,000-row sample and impute "
              "only rows with gaps. Details and measured timings in the final report.")

ap = argparse.ArgumentParser()
ap.add_argument("--table", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

res = pd.read_csv(args.results)
rows = {}
for _, r in res.iterrows():
    ds, rest = r["file"].split("_", 1)
    rows[f"{ALIAS.get(ds, ds)}_{rest}"] = r

wb = openpyxl.load_workbook(args.table)
ws = wb.active
filled = 0
for i in range(2, ws.max_row + 1):
    label = ws.cell(i, 1).value
    if label == "Average NRMS/AE":
        ws.cell(i, 2).value = round(float(res["NRMS"].mean()), 4)
        ws.cell(i, 3).value = round(float(res["AE"].mean()), 4)
        ws.cell(i, 4).value = (f"Mean over {int(res['NRMS'].count())} numerical and "
                               f"{int(res['AE'].count())} categorical results "
                               f"(mean/mode baseline: NRMS {res['NRMS_meanmode'].mean():.4f}, "
                               f"AE {res['AE_meanmode'].mean():.4f})")
        continue
    if label == "Complexity Analysis":
        ws.cell(i, 2).value = COMPLEXITY
        continue
    r = rows.get(label)
    if r is None:
        continue
    if pd.notna(r.get("NRMS")):
        ws.cell(i, 2).value = round(float(r["NRMS"]), 4)
    if pd.notna(r.get("AE")):
        ws.cell(i, 3).value = round(float(r["AE"]), 4)
    ws.cell(i, 4).value = r["key_parameters"]
    filled += 1
wb.save(args.out)
print(f"filled {filled} rows (+ average and complexity rows) -> {args.out}")
