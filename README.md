# CS 6725 Project C10 — VAEAC missing-data imputation

Our implementation of the Variational Autoencoder with Arbitrary Conditioning
(Ivanov, Figurnov, Vetrov, ICLR 2019), written from the paper. One file per project step.

**Group:** Ahmed Mubarak Olatunji · Chimaobi Ogbonna · Martins Akingboye
**Course:** CS 6725 Data Mining, UNB, Fall 2026 · Prof. Roozbeh Razavi-Far

## Status

All **1,144 files** (26 datasets × 11 missingness patterns × 4 rates) are imputed and scored.

| | VAEAC | mean/mode baseline | VAEAC better on |
|---|---|---|---|
| Average NRMS (numerical, lower is better) | **0.118** | 0.146 | 802 of 924 files |
| Average AE (categorical, higher is better) | **0.732** | 0.534 | 389 of 406 files |

Weakest datasets: Credit, Difdoug and KDD (NRMS about level with mean/mode).
Per-dataset, per-pattern and per-rate numbers are in `results/summary/`.

## Files

| File | Project step | What it does |
|---|---|---|
| `step1_load.py` | 1. Read the data | Reads `.xlsx` / `.xls` / `.csv` of any size; strips stray spaces in text cells; decides numerical vs categorical per column from the incomplete file only. |
| `step2_missingness.py` | 2. Identify missing data | Missing ratio, per-feature and per-row counts, and the pattern type: **univariate**, **monotone** or **arbitrary**. |
| `step3_vaeac.py` | 3. Impute with VAEAC | The model (proposal, prior and generator networks), the training objective, model selection on validation IWAE, and imputation by sampling from the prior. |
| `step4_evaluate.py` | 4. Compare with the complete data | NRMS and AE from the course instructions, a mean/mode baseline on the same cells, checks, and plots. |
| `main.py` | all | Runs steps 1–4 over the course folders, or over a single file. |
| `fill_table.py` | reporting | Writes results into a copy of `Table-NRMS-AE.xlsx`, including the *Average NRMS/AE* and *Complexity Analysis* rows. |
| `summarize.py` | reporting | Summary tables and charts → `results/summary/`. |
| `recompute_ae.py` | reporting | Re-scores saved imputed files without re-running the imputation. |
| `benchmark_complexity.py` | complexity study | Times training and imputation while changing rows, features and network width one at a time. |
| `run_small.bat`, `test_one.bat` | Windows shortcuts | Double-click to run (paths inside may need editing). |

## Setup (once)

Python 3.10+. On Windows, if pip says "Access is denied", add `--user`.

```
python -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --user pandas openpyxl matplotlib numpy
```

The course datasets are **not** in this repo. Use your own copy and point `ROOT` at the folder
that contains `Incomplete Datasets Without Labels` and `Original Datasets Without Labels`.

## Run (Git Bash)

```
ROOT="C:/Users/<you>/Downloads/data_mining_project/Course Project Datasets/Course Project Datasets"

# chosen datasets, all 44 files each
python main.py --data_root "$ROOT" --datasets Iris Wine Glass --out results --no_plots

# only some patterns / rates
python main.py --data_root "$ROOT" --datasets Iris --patterns C AE --rates 5 10 --out results --no_plots

# any single file (e.g. one the lecturer brings)
python main.py --input "his_incomplete.xlsx" --original "his_original.xlsx" --out demo --no_plots
```

In cmd use `set ROOT=...` and `"%ROOT%"`; in PowerShell `$ROOT = "..."` and `$ROOT`.

**Live demo:** use a fresh `--out` folder (e.g. `--out demo`). `main.py` skips every file already
listed in that folder's `results.csv`, so running into `results` would do nothing.
The same skipping makes an interrupted run safe to restart.

**Large datasets** (Letter, Adult, C4, KDD):

```
export VAEAC_CACHE="C:/Users/<you>/vaeac_cache"
python main.py --data_root "$ROOT" --datasets Letter Adult C4 KDD --out results --no_plots --max_train_rows 10000 --missing_rows_only 1
```

## After a run

```
python fill_table.py --table "$ROOT/Table-NRMS-AE.xlsx" --results results/results.csv --out results/Table-NRMS-AE_C10.xlsx
python summarize.py
```

**Compare imputed values with the original, cell by cell:**

```
D=Iris; F=Iris_C_10
python -c "
import step1_load as s
o=s.load_table('$ROOT/Original Datasets Without Labels/$D.xlsx')
i=s.load_table('$ROOT/Incomplete Datasets Without Labels/$D/$F.xlsx')
m=s.load_table('results/imputed/${F}_imputed.xlsx')
for r,c in zip(*i.isna().values.nonzero()): print(f'row {r} col {c}: original={o.iat[r,c]}  imputed={m.iat[r,c]}')
"
```

## Outputs (`--out` folder)

* `results.csv` — one row per file: missingness summary and pattern type (step 2), NRMS / AE for
  VAEAC and for mean/mode (step 4), training details and timing.
* `imputed/<file>_imputed.xlsx` — the completed dataset, same layout as the course files
  (not in git; delivered separately at the final demo).
* `Table-NRMS-AE_C10.xlsx` — the filled course table.
* `summary/` — `by_dataset.csv`, `by_pattern.csv`, `by_rate.csv`, `pattern_types.csv` and charts.
* `nrms_by_rate.png`, `ae_by_rate.png` — VAEAC vs mean/mode by missing rate.
* `histories/` — training loss and validation IWAE per checkpoint.

## How the model works (step 3), with paper references

* **Masks.** `b` marks unobserved features (1 = unobserved), `m` marks truly missing values.
  During training every observed value is hidden with probability 0.2 and truly missing values are
  always unobserved (App. A.2). Only values that were hidden *and* are known enter the loss (Eq. 9).
* **Inputs.** Numerical features are standardised with observed means / standard deviations;
  categorical features are one-hot encoded. Unobserved entries are set to 0 and the mask is fed
  alongside, so the network can tell a placeholder 0 from a real 0 (Sec. 4.2).
* **Networks** (`Encoder`, `Generator`): residual MLPs. The proposal sees `[x, b, m]` (training only);
  the prior sees `[x_{1-b}, b]`; the generator sees `z`, the prior's hidden layers (skip connections
  by concatenation, App. A.1) and `[x_{1-b}, b]`. Numerical outputs are Gaussian means with σ = 1
  (App. A.2); categorical outputs are softmax logits (Sec. 4.2).
* **Loss** (`VAEAC.elbo`): reconstruction log-likelihood of hidden values − KL(proposal ‖ prior)
  + the Normal-Gamma prior regulariser with σ_μ = 10⁴, σ_σ = 10⁻⁴ (Eq. 6, 8).
* **Model selection.** A validation split (25% of rows, at most 2,000) with fixed masks; the epoch
  with the best IWAE log-likelihood (25 samples) is kept (App. A.2).
* **Imputation.** 25 latent samples from the prior per row; numbers are averaged and categories
  decided by majority vote. Observed values are copied unchanged (checked in step 4).

Default settings: width 256, 4 residual layers per network, latent size 32, Adam lr 3·10⁻⁴,
batch 64, about 2,000 training steps per file (at least 20 epochs). Every setting in `DEFAULTS`
(`step3_vaeac.py`) is also a command-line flag, e.g. `--steps 4000 --width 128`.

## Metrics (step 4)

* NRMS = ‖X_est − X_orig‖_F / ‖X_orig‖_F over the numerical features (lower is better).
* AE = (1/n) Σ I(x̂ᵢ = xᵢ) over the n missing categorical entries (higher is better).
* Mixed datasets (Aheart, Credit, Abalone, Adult, KDD) get both.

## Computational complexity and scalability

* One training step: O(B · (L·h² + d·h)) for batch B, L layers of width h, d input features.
  Training: O(E · n · (L·h² + d·h)) — linear in rows n and features d, quadratic in width h.
* Imputation: O(n · K · (L·h² + d·h)) with K = 25 samples.
* Our model: h = 256, L = 4, latent 32 → about 1.2–1.6M parameters, ~1.2M multiply-adds per row.
* Scaling options used for the large datasets: `--max_train_rows` (train on a subsample),
  `--missing_rows_only 1` (impute only rows with gaps), `--max_val_rows` (cap validation cost),
  `VAEAC_CACHE` (parse each large Excel file once). Imputation is done in chunks, so memory does
  not grow with n. A GPU is used automatically when PyTorch can see one.

## Data notes

* The Splice **original** file stores letters with a leading space (`" G"`); the incomplete files do
  not (`"G"`). Text cells are stripped on load and before comparison; without that, Splice's AE drops
  to about 0.003.
* `List of Datasets.xlsx` lists Credit as 9 numerical / 6 categorical; the file actually has
  6 numerical / 9 categorical. It lists HOV with 234 rows; the file has 232.
* Table-NRMS-AE spells three names differently from the folders (BUPA, TTTEG, Credit);
  `fill_table.py` maps them.
