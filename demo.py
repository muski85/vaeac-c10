"""Live demo: impute one course file and compare it with the original.

    python demo.py                  asks which dataset, pattern and rate
    python demo.py Iris C 10        runs that file straight away
    python demo.py file             asks for an incomplete file (and its original) anywhere on disk
    python demo.py check Iris C 10  scores the SAVED file in results/imputed (no training, matches the table)

Prints VAEAC vs mean/mode (NRMS / AE) and a few missing cells: true value, VAEAC, mean/mode.
The imputed file is saved in demo/imputed/. Nothing in results/ is touched.
"""
import os
import sys
import time
from pathlib import Path

import main as m
import step1_load as s1
import step3_vaeac as s3
import step4_evaluate as s4

# Folder holding "Incomplete Datasets Without Labels" and "Original Datasets Without Labels".
# Change this line if your copy of the course data is somewhere else.
DATA_ROOT = Path.home() / "Downloads" / "data_mining_project" / "Course Project Datasets" / "Course Project Datasets"
OUT = "demo"
SHOW_CELLS = 10
BIG = {"Letter", "Adult", "C4", "KDD"}


def ask(question, default):
    answer = input(f"{question} [{default}]: ").strip()
    return answer or default


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def run(incomplete_path, original_path):
    config = dict(s3.DEFAULTS)
    name = Path(incomplete_path).stem
    if name.split("_")[0] in BIG:                      # same settings we used for the big runs
        config.update(max_train_rows=10000, missing_rows_only=1)
    print(f"\nImputing {name} ... (training the networks, usually under a minute for small datasets)")
    t = time.time()
    row = m.run_one(str(incomplete_path), str(original_path) if original_path else None, OUT, config,
                    make_plots=False)
    print(f"Done in {time.time() - t:.0f}s. Missing-data pattern: {row['pattern_type']}, "
          f"{row['missing_values']} missing cells ({row['missing_ratio_pct']}%).\n")

    if not original_path:
        print(f"No original given, so no score. Imputed file: {OUT}/imputed/{name}_imputed.xlsx")
        return

    print("                 VAEAC     mean/mode")
    if row.get("NRMS") is not None:
        print(f"NRMS (lower)   {row['NRMS']:.4f}    {row['NRMS_meanmode']:.4f}")
    if row.get("AE") is not None:
        print(f"AE   (higher)  {row['AE']:.4f}    {row['AE_meanmode']:.4f}")

    # A few of the missing cells, side by side.
    incomplete = s1.load_table(str(incomplete_path))
    original = s1.load_table(str(original_path))
    imputed = s1.load_table(os.path.join(OUT, "imputed", name + "_imputed.xlsx"))
    baseline = s4.mean_mode_impute(incomplete, s1.detect_types(incomplete))
    cells = list(zip(*incomplete.isna().values.nonzero()))
    print(f"\nFirst {min(SHOW_CELLS, len(cells))} of {len(cells)} missing cells:")
    print(f"{'row':>5} {'col':>4}  {'original':>10}  {'VAEAC':>10}  {'mean/mode':>10}")
    for r, c in cells[:SHOW_CELLS]:
        print(f"{r:>5} {c:>4}  {fmt(original.iat[r, c]):>10}  {fmt(imputed.iat[r, c]):>10}  "
              f"{fmt(baseline.iat[r, c]):>10}")
    print(f"\nImputed file saved: {OUT}/imputed/{name}_imputed.xlsx")


def course_file(dataset, pattern, rate):
    folder = DATA_ROOT / "Incomplete Datasets Without Labels"
    if folder.exists():                                # accept "iris" for "Iris", "kdd" for "KDD", ...
        dataset = next((p.name for p in folder.iterdir() if p.name.lower() == dataset.lower()), dataset)
    pattern = pattern.upper()
    inc = DATA_ROOT / "Incomplete Datasets Without Labels" / dataset / f"{dataset}_{pattern}_{rate}.xlsx"
    org = DATA_ROOT / "Original Datasets Without Labels" / f"{dataset}.xlsx"
    if not inc.exists():
        print(f"Not found: {inc}\nCheck the dataset name (case matters) and DATA_ROOT at the top of demo.py.")
        return None, None
    return inc, org


def check(dataset, pattern, rate):
    """Score the SAVED imputed file in results/imputed (no training): gives exactly the table value."""
    inc, org = course_file(dataset, pattern, rate)
    if not inc:
        return
    name = inc.stem
    saved = Path("results") / "imputed" / (name + "_imputed.xlsx")
    if not saved.exists():
        print(f"No saved imputed file: {saved}")
        return
    incomplete = s1.load_table(str(inc))
    original = s1.load_table(str(org))
    imputed = s1.load_table(str(saved))
    is_cat = s1.detect_types(incomplete)
    baseline = s4.mean_mode_impute(incomplete, is_cat)
    print(f"\nSaved file {saved} vs original {org.name}")
    print("                 VAEAC     mean/mode")
    n = s4.nrms(original, imputed, is_cat)
    a = s4.ae(original, imputed, incomplete, is_cat)
    if n is not None:
        print(f"NRMS (lower)   {n:.4f}    {s4.nrms(original, baseline, is_cat):.4f}")
    if a is not None:
        print(f"AE   (higher)  {a:.4f}    {s4.ae(original, baseline, incomplete, is_cat):.4f}")


def main():
    args = sys.argv[1:]
    if len(args) == 4 and args[0].lower() == "check":
        check(*args[1:])
        return
    if args and args[0].lower() == "file":
        inc = ask("Incomplete file path", "").strip('"')
        org = ask("Original file path (Enter to skip)", "").strip('"') or None
        run(inc, org)
        return
    if len(args) == 3:
        inc, org = course_file(*args)
        if inc:
            run(inc, org)
        return

    folder = DATA_ROOT / "Incomplete Datasets Without Labels"
    names = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
    print("Datasets:", ", ".join(names) or f"(none found - check DATA_ROOT: {DATA_ROOT})")
    print("Patterns: " + " ".join(m.PATTERNS) + "    Rates: " + " ".join(m.RATES))
    while True:
        inc, org = course_file(ask("\nDataset", "Iris"), ask("Pattern", "C").upper(), ask("Rate", "10"))
        if inc:
            run(inc, org)
        if ask("\nRun another? (y/n)", "n").lower() != "y":
            break


if __name__ == "__main__":
    main()
