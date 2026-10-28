"""
CS 6725 Data Mining - Project C10: VAEAC missing-data imputation.

Runs the four project steps on the course files:
  step 1  load the incomplete file                   (step1_load.py)
  step 2  find the missing ratio and pattern type    (step2_missingness.py)
  step 3  impute with our VAEAC implementation        (step3_vaeac.py)
  step 4  save the imputed file, score NRMS / AE      (step4_evaluate.py)

Examples
  # every pattern and rate for three datasets
  python main.py --data_root "Course Project Datasets/Course Project Datasets" --datasets Iris HOV Aheart

  # one file of your own, no original available (steps 1-3 only)
  python main.py --input my_data.xlsx --out results_single
"""
import argparse
import csv
import os
import re
import time

import pandas as pd

import step1_load as s1
import step2_missingness as s2
import step3_vaeac as s3
import step4_evaluate as s4

PATTERNS = ["C", "AE", "AG", "AL", "AN", "AW", "NE", "NG", "NL", "NN", "NW"]
RATES = ["1", "5", "10", "20"]


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def done_files(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["file"])


def run_one(incomplete_path, original_path, out_dir, config, make_plots=True):
    name = os.path.splitext(os.path.basename(incomplete_path))[0]
    # ---- step 1 -------------------------------------------------------------------------
    incomplete = s1.load_table(incomplete_path)
    is_cat = s1.detect_types(incomplete)
    # ---- step 2 -------------------------------------------------------------------------
    miss = s2.summarise(incomplete, is_cat)
    # ---- step 3 -------------------------------------------------------------------------
    imputed, info = s3.fit_and_impute(incomplete, is_cat, config)
    s4.check_observed_unchanged(incomplete, imputed, is_cat)
    os.makedirs(os.path.join(out_dir, "imputed"), exist_ok=True)
    imputed.to_excel(os.path.join(out_dir, "imputed", name + "_imputed.xlsx"), header=False, index=False)
    # ---- step 4 -------------------------------------------------------------------------
    row = dict(file=name, **{k: v for k, v in miss.items()})
    m = re.match(r"(.+)_([A-Z]+)_(\d+)$", name)
    row["dataset"], row["pattern"], row["rate"] = (m.groups() if m else (name, "", ""))
    if original_path:
        original = s1.load_table(original_path)
        baseline = s4.mean_mode_impute(incomplete, is_cat)
        row.update(NRMS=s4.nrms(original, imputed, is_cat), AE=s4.ae(original, imputed, incomplete, is_cat),
                   NRMS_meanmode=s4.nrms(original, baseline, is_cat),
                   AE_meanmode=s4.ae(original, baseline, incomplete, is_cat))
    cfg = info["config"]
    row.update(epochs=info["epochs"], steps=info["steps"], best_epoch=info["best_epoch"],
               train_rows=info["train_rows"], val_rows=info["val_rows"], n_params=info["params"],
               train_s=info["train_s"], impute_s=info["impute_s"], device=info["device"],
               key_parameters=(f"own VAEAC width={cfg['width']} depth={cfg['depth']} latent={cfg['latent']} "
                               f"lr={cfg['lr']} batch={cfg['batch_size']} hide p={cfg['hide_prob']} "
                               f"epochs={info['epochs']} K={cfg['num_imputations']} seed={cfg['seed']}"))
    if make_plots:
        pdir = os.path.join(out_dir, "plots")
        os.makedirs(pdir, exist_ok=True)
        s4.plot_training(info["history"], os.path.join(pdir, name + "_training.png"), name)
        s2.plot_mask(incomplete, os.path.join(pdir, name + "_missing.png"), name + ": missing values")
        if original_path:
            s4.plot_imputed_vs_true(original, imputed, incomplete, is_cat,
                                    os.path.join(pdir, name + "_imputed_vs_true.png"), name)
    hdir = os.path.join(out_dir, "histories")
    os.makedirs(hdir, exist_ok=True)
    pd.DataFrame(info["history"], columns=["epoch", "train_loss", "val_iwae"]).to_csv(
        os.path.join(hdir, name + ".csv"), index=False)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", help="folder holding 'Incomplete Datasets Without Labels' and "
                                        "'Original Datasets Without Labels'")
    ap.add_argument("--datasets", nargs="+", default=[])
    ap.add_argument("--patterns", nargs="+", default=PATTERNS)
    ap.add_argument("--rates", nargs="+", default=RATES)
    ap.add_argument("--input", help="a single incomplete file (no original needed)")
    ap.add_argument("--original", help="optional complete version of --input, for NRMS / AE")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_plots", action="store_true")
    for k, v in s3.DEFAULTS.items():                         # every model setting is a flag
        ap.add_argument("--" + k, type=type(v) if v is not None else int, default=v)
    args = ap.parse_args()
    config = {k: getattr(args, k) for k in s3.DEFAULTS}
    os.makedirs(args.out, exist_ok=True)
    results_csv = os.path.join(args.out, "results.csv")

    jobs = []
    if args.input:
        jobs.append((args.input, args.original))
    for ds in args.datasets:
        for pat in args.patterns:
            for rate in args.rates:
                inc = os.path.join(args.data_root, "Incomplete Datasets Without Labels", ds,
                                   f"{ds}_{pat}_{rate}.xlsx")
                org = os.path.join(args.data_root, "Original Datasets Without Labels", ds + ".xlsx")
                jobs.append((inc, org))

    finished = done_files(results_csv)
    for inc, org in jobs:
        name = os.path.splitext(os.path.basename(inc))[0]
        if name in finished:
            continue
        t = time.time()
        row = run_one(inc, org, args.out, config, make_plots=not args.no_plots)
        row["wall_s"] = round(time.time() - t, 1)
        append_row(results_csv, row)
        print(f"{name:16s} pattern={row['pattern_type']:10s} NRMS={row.get('NRMS')} "
              f"AE={row.get('AE')} | mean/mode NRMS={row.get('NRMS_meanmode')} "
              f"AE={row.get('AE_meanmode')} | {row['wall_s']}s", flush=True)

    if os.path.exists(results_csv):
        res = pd.read_csv(results_csv)
        res["rate"] = pd.to_numeric(res["rate"], errors="coerce")
        s4.plot_rate_curves(res, os.path.join(args.out, "nrms_by_rate.png"), "NRMS")
        s4.plot_rate_curves(res, os.path.join(args.out, "ae_by_rate.png"), "AE")


if __name__ == "__main__":
    main()
