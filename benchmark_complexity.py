"""Controlled timing study for the complexity / scalability section.

Run it when nothing else is using the CPU:
    python benchmark_complexity.py --out results/complexity

It times our VAEAC on synthetic tables, changing ONE thing at a time:
  * rows n      (features fixed)   -> training cost per step should stay flat, imputation grows ~ n
  * features d  (rows fixed)       -> both grow slowly, because the hidden width h dominates (L*h^2 >> d*h)
  * width h     (data fixed)       -> cost grows ~ h^2
Each setting runs a fixed number of training steps and one full imputation pass with K samples.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import step3_vaeac as s3

VAEAC, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def synthetic(n, d, rate=0.10, seed=0):
    """Correlated numeric table with `rate` of the cells hidden completely at random."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, max(2, d // 4)))
    X = z @ rng.normal(size=(z.shape[1], d)) + 0.3 * rng.normal(size=(n, d))
    X[rng.random((n, d)) < rate] = np.nan
    return torch.tensor(X, dtype=torch.float32)


def time_one(n, d, width, steps, K, depth=4, latent=32, bs=64):
    torch.manual_seed(0)
    X = synthetic(n, d)
    X = (X - torch.nanmean(X, 0)) / 1.0
    model = s3.VAEAC([1] * d, width, depth, latent)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(0)
    params = sum(p.numel() for p in model.parameters())

    model.train()
    t0 = time.perf_counter()
    for s in range(steps):
        batch = X[torch.randint(0, n, (bs,), generator=gen)]
        b, m = s3.training_mask(batch, 0.2, gen)
        loss = -model.elbo(batch, b, m).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    train_ms = 1000 * (time.perf_counter() - t0) / steps

    model.eval()
    t1 = time.perf_counter()
    for i in range(0, n, 1024):
        model.impute_samples(X[i:i + 1024], K)
    impute_s = time.perf_counter() - t1
    return dict(rows=n, features=d, width=width, params=params,
                train_ms_per_step=round(train_ms, 2), impute_s=round(impute_s, 3),
                impute_ms_per_1000_rows=round(1e6 * impute_s / n, 2))


def panel(ax, x, y, xlabel, ylabel, title, logx=False):
    ax.plot(x, y, marker="o", color=VAEAC, linewidth=2, markersize=6)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_ylim(bottom=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/complexity")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="tiny settings, just to test the script")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    rows = [500, 2000, 8000, 32000, 128000]
    feats = [5, 10, 20, 40, 80]
    widths = [64, 128, 256, 512]
    if a.quick:
        rows, feats, widths, a.steps = [200, 400], [4, 8], [32, 64], 5

    recs = []
    for n in rows:
        recs.append(dict(sweep="rows", **time_one(n, 20 if not a.quick else 4, 256 if not a.quick else 32, a.steps, a.K)))
        print(recs[-1], flush=True)
    for d in feats:
        recs.append(dict(sweep="features", **time_one(2000 if not a.quick else 200, d, 256 if not a.quick else 32, a.steps, a.K)))
        print(recs[-1], flush=True)
    for h in widths:
        recs.append(dict(sweep="width", **time_one(2000 if not a.quick else 200, 20 if not a.quick else 4, h, a.steps, a.K)))
        print(recs[-1], flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(a.out, "timings.csv"), index=False)

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), dpi=160)
    for col, (sweep, xcol, xl, logx) in enumerate([("rows", "rows", "rows n (log)", True),
                                                     ("features", "features", "features d", False),
                                                     ("width", "width", "hidden width h", False)]):
        g = df[df.sweep == sweep]
        panel(axs[0, col], g[xcol], g.train_ms_per_step, xl, "ms per training step",
              f"Training step vs {sweep}", logx)
        panel(axs[1, col], g[xcol], g.impute_s, xl, "imputation time (s)",
              f"Full imputation (K={a.K}) vs {sweep}", logx)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "complexity_sweeps.png"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
