# CS 6725 Project C10 — VAEAC missing-data imputation

Our implementation of the Variational Autoencoder with Arbitrary Conditioning
(Ivanov, Figurnov, Vetrov, ICLR 2019), written from the paper. One file per project step.

| File | Project step | What it does |
|---|---|---|
| `step1_load.py` | 1. Read the data | Reads `.xlsx` / `.xls` / `.csv` of any size; decides numerical vs categorical per column from the incomplete file only. |
| `step2_missingness.py` | 2. Identify missing data | Missing ratio, per-feature and per-row counts, and the pattern type: **univariate**, **monotone** or **arbitrary**. Saves a picture of the missing-value matrix. |
| `step3_vaeac.py` | 3. Impute with VAEAC | The model (proposal, prior and generator networks), the training objective, model selection on validation IWAE, and imputation by sampling from the prior. |
| `step4_evaluate.py` | 4. Compare with the complete data | NRMS and AE from the course instructions, a mean/mode baseline on the same cells, checks, and plots. |
| `main.py` | all | Runs steps 1–4 over the course folders, or over a single file of your own. |
| `fill_table.py` | reporting | Writes the results into a copy of `Table-NRMS-AE.xlsx`. |

## Run

```
pip install -r requirements.txt

# all 11 patterns x 4 rates for three datasets
python main.py --data_root "<...>/Course Project Datasets/Course Project Datasets" --datasets Iris HOV Aheart --out results

# only some patterns / rates
python main.py --data_root "<...>" --datasets Wine --patterns C NE --rates 10 20

# any single file (original optional; without it, steps 1-3 only)
python main.py --input my_incomplete.xlsx --original my_complete.xlsx --out results_single

# results into the course table
python fill_table.py --table "<...>/Table-NRMS-AE.xlsx" --results results/results.csv --out Table-NRMS-AE_C10.xlsx
```

`main.py` skips files already in `results/results.csv`, so an interrupted run can be restarted.
Every model setting in `DEFAULTS` (top of the training section in `step3_vaeac.py`) is also a
command-line flag, e.g. `--steps 4000 --width 128`.

Outputs in `--out`:
* `results.csv` — one row per file: missingness summary and pattern type (step 2), NRMS / AE for
  VAEAC and for mean/mode (step 4), training details and timing.
* `imputed/<file>_imputed.xlsx` — the completed dataset, same layout as the course files.
* `plots/` — missing-value matrix, training curve, imputed-vs-true scatter for each file.
* `nrms_by_rate.png`, `ae_by_rate.png` — VAEAC vs mean/mode by missing rate.
* `histories/` — training loss and validation IWAE per checkpoint.

## How the model works (step 3), with paper references

* **Masks.** `b` marks unobserved features (1 = unobserved), `m` marks truly missing values.
  During training every observed value is hidden with probability 0.2 and truly missing values are
  always unobserved (App. A.2). Only values that were hidden *and* are known enter the loss (Eq. 9).
* **Inputs.** Numerical features are standardised with observed means / standard deviations;
  categorical features are one-hot encoded. Unobserved entries are set to 0 and the mask is fed
  alongside, so the network can tell a placeholder 0 from a real 0 (Sec. 4.2).
* **Networks** (`Encoder`, `Generator`): residual MLPs. The proposal sees `[x, b, m]`; the prior
  sees `[x_{1-b}, b]`; the generator sees `z`, the prior's hidden layers (skip connections by
  concatenation, App. A.1) and `[x_{1-b}, b]`. Numerical outputs are Gaussian means with σ = 1
  (App. A.2); categorical outputs are softmax logits (Sec. 4.2).
* **Loss** (`VAEAC.elbo`): reconstruction log-likelihood of hidden values − KL(proposal ‖ prior)
  + the Normal-Gamma prior regulariser with σ_μ = 10⁴, σ_σ = 10⁻⁴ (Eq. 6, 8).
* **Model selection.** A validation split (25% of rows, at most 2,000) with fixed masks; the epoch
  with the best IWAE log-likelihood (25 samples) is kept (App. A.2).
* **Imputation.** 25 latent samples from the prior per row; for each, the generator's most likely
  value; numbers are averaged and categories decided by majority vote. Observed values are copied
  unchanged (checked in step 4).

Default settings: width 256, 4 residual layers per network, latent size 32, Adam lr 3·10⁻⁴,
batch 64, about 2,000 training steps per file (at least 20 epochs).

## Metrics (step 4)

* NRMS = ‖X_orig − X_imp‖_F / ‖X_orig‖_F over the numerical features (lower is better).
* AE = share of the missing categorical entries imputed with the correct category (higher is better).

## Scaling options for the large datasets

`--max_val_rows` caps the validation rows (IWAE costs 25 passes per row). `--max_train_rows`
trains on a random subset of rows while still imputing every row. A GPU is used automatically
when PyTorch can see one.
