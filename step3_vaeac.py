"""
STEP 3 - Impute the missing values with VAEAC.

Our implementation of: O. Ivanov, M. Figurnov, D. Vetrov, "Variational Autoencoder with Arbitrary
Conditioning", ICLR 2019. Section and equation numbers below refer to that paper.

Notation
  x  : one record (row) with d features
  b  : mask of UNOBSERVED features (1 = unobserved). It contains the truly missing values plus
       values we hide on purpose during training so the model has something to learn to predict.
  m  : mask of TRULY missing values (m <= b). Their true value is unknown, so they never enter the loss.
  z  : latent vector

Model (Sec. 4.2 and 4.3)
  proposal  q_phi(z | x, b)               - sees the full row; used only during training
  prior     p_psi(z | x_{1-b}, b)          - sees only observed values; used for training and imputation
  generator p_theta(x_b | z, x_{1-b}, b)   - turns z into a distribution for every feature:
                                             Gaussian with sigma = 1 for numerical features (App. A.2)
                                             Softmax over categories for categorical features
  Skip connections from prior to generator, implemented by concatenation (App. A.1).

Training objective (Eq. 6, 8, 9): maximise
  sum_{i: b_i = 1, m_i = 0} log p_theta(x_i | z, x_{1-b}, b)    reconstruction of hidden values
  - KL( q_phi(z|x,b) || p_psi(z|x_{1-b},b) )                     keep proposal close to prior
  - mu_psi^2 / (2 sigma_mu^2) + sigma_sigma (log sigma_psi - sigma_psi)   prior regulariser (Eq. 8)

Mask distribution for training (App. A.2): every value is hidden with probability 0.2, and truly
missing values are always unobserved: p(b_i = 1 | x_i missing) = 1.
Model selection (App. A.2): keep the epoch with the best IWAE log-likelihood on a validation split.
"""
import math
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# Feature coding: DataFrame  <->  numeric matrix of "codes" (NaN = missing)
# --------------------------------------------------------------------------------------------
class FeatureCodec:
    """Numerical features are standardised; categorical features become integer codes 0..K-1.
    Everything is fitted on OBSERVED values of the incomplete file only."""

    def __init__(self, is_cat):
        self.is_cat = list(is_cat)

    def fit(self, df):
        self.mean, self.std, self.categories, self.sizes = [], [], [], []
        for j, c in enumerate(df.columns):
            obs = df[c].dropna()
            if self.is_cat[j]:
                cats = sorted(obs.astype(str).unique())
                self.categories.append(cats)
                self.sizes.append(max(2, len(cats)))      # at least 2 classes for a softmax
                self.mean.append(0.0)
                self.std.append(1.0)
            else:
                v = pd.to_numeric(obs).astype(float)
                self.categories.append(None)
                self.sizes.append(1)
                self.mean.append(float(v.mean()) if len(v) else 0.0)
                sd = float(v.std()) if len(v) > 1 else 1.0
                self.std.append(sd if sd > 1e-12 else 1.0)  # constant column -> do not divide by 0
        return self

    def encode(self, df):
        X = np.full(df.shape, np.nan, dtype=np.float32)
        for j, c in enumerate(df.columns):
            col = df[c]
            ok = col.notna().values
            if self.is_cat[j]:
                index = {v: k for k, v in enumerate(self.categories[j])}
                X[ok, j] = [index.get(str(v), np.nan) for v in col[ok]]
            else:
                X[ok, j] = (pd.to_numeric(col[ok]).astype(float).values - self.mean[j]) / self.std[j]
        return X

    def decode(self, X, template):
        """Turn codes back into original units / labels, keeping the template's observed cells."""
        out = template.copy().astype(object)
        for j, c in enumerate(template.columns):
            miss = template[c].isna().values
            if not miss.any():
                continue
            if self.is_cat[j]:
                codes = np.clip(np.rint(X[miss, j]).astype(int), 0, len(self.categories[j]) - 1)
                out.loc[miss, c] = [self.categories[j][k] for k in codes]
            else:
                out.loc[miss, c] = X[miss, j] * self.std[j] + self.mean[j]
        for j, c in enumerate(template.columns):
            if not self.is_cat[j]:
                out[c] = pd.to_numeric(out[c])
        return out


def to_network_input(X, sizes):
    """One-hot encode categorical features and put 0 where a value is unobserved (NaN).
    Together with the mask this lets the network tell a placeholder 0 from a real 0."""
    cols = []
    for j, k in enumerate(sizes):
        v = X[:, j]
        nan = torch.isnan(v)
        if k == 1:
            cols.append(torch.where(nan, torch.zeros_like(v), v).unsqueeze(1))
        else:
            idx = torch.where(nan, torch.zeros_like(v), v).long()
            oh = F.one_hot(idx, k).float()
            oh[nan] = 0.0
            cols.append(oh)
    return torch.cat(cols, dim=1)


# --------------------------------------------------------------------------------------------
# The three networks
# --------------------------------------------------------------------------------------------
class ResidualLayer(nn.Module):
    """h -> h + LeakyReLU(Linear(h)). Residual connections make deeper MLPs easier to train."""

    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, h):
        return h + F.leaky_relu(self.lin(h))


class Encoder(nn.Module):
    """Used for both the proposal and the prior network: input -> Gaussian over z.
    The prior also returns its hidden states so the generator can use them (skip connections)."""

    def __init__(self, in_dim, width, depth, latent):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList([ResidualLayer(width) for _ in range(depth)])
        self.out = nn.Linear(width, 2 * latent)

    def forward(self, x):
        h = F.leaky_relu(self.inp(x))
        hidden = [h]
        for layer in self.layers:
            h = layer(h)
            hidden.append(h)
        mu, raw = self.out(h).chunk(2, dim=1)
        sigma = F.softplus(raw) + 1e-3                     # sigma must be positive
        return torch.distributions.Normal(mu, sigma), hidden


class Generator(nn.Module):
    """z (+ prior hidden states + observed input) -> parameters of p(x | z, x_{1-b}, b).
    Output per feature: 1 number (Gaussian mean) for numerical, K logits for categorical."""

    def __init__(self, cond_dim, out_dim, width, depth, latent):
        super().__init__()
        self.inp = nn.Linear(latent, width)
        # layer i concatenates the prior hidden state from the mirrored depth (U-Net style)
        self.layers = nn.ModuleList([nn.Linear(2 * width, width) for _ in range(depth + 1)])
        self.out = nn.Linear(width + cond_dim, out_dim)

    def forward(self, z, prior_hidden, cond):
        h = F.leaky_relu(self.inp(z))
        for lin, skip in zip(self.layers, reversed(prior_hidden)):
            h = h + F.leaky_relu(lin(torch.cat([h, skip], dim=1)))
        return self.out(torch.cat([h, cond], dim=1))


class VAEAC(nn.Module):
    def __init__(self, sizes, width=256, depth=4, latent=32, sigma_mu=1e4, sigma_sigma=1e-4):
        super().__init__()
        self.sizes = list(sizes)
        d = len(sizes)
        onehot = sum(sizes)                                 # width of the one-hot data input
        self.proposal = Encoder(onehot + 2 * d, width, depth, latent)   # x, b, m
        self.prior = Encoder(onehot + d, width, depth, latent)          # x_{1-b}, b
        self.generator = Generator(onehot + d, onehot, width, depth, latent)
        self.sigma_mu, self.sigma_sigma = sigma_mu, sigma_sigma

    # ---- building the inputs --------------------------------------------------------------
    def prior_input(self, X, b):
        X_obs = X.masked_fill(b.bool(), float("nan"))       # hide unobserved values
        return torch.cat([to_network_input(X_obs, self.sizes), b], dim=1)

    def proposal_input(self, X, b, m):
        return torch.cat([to_network_input(X, self.sizes), b, m], dim=1)

    # ---- likelihood of the hidden values -------------------------------------------------
    def log_likelihood(self, params, X, target):
        """Sum of log p(x_i | ...) over features where target == 1 (hidden AND known)."""
        total = torch.zeros(X.shape[0], device=X.device)
        col = 0
        for j, k in enumerate(self.sizes):
            t = target[:, j]
            x = torch.nan_to_num(X[:, j], nan=0.0)
            if k == 1:
                mean = params[:, col]
                lp = -0.5 * (x - mean) ** 2 - 0.5 * math.log(2 * math.pi)   # Gaussian, sigma = 1
                col += 1
            else:
                logp = F.log_softmax(params[:, col:col + k], dim=1)
                lp = logp.gather(1, x.long().clamp(0, k - 1).unsqueeze(1)).squeeze(1)
                col += k
            total = total + lp * t
        return total

    def prior_regulariser(self, prior):
        mu, sigma = prior.loc, prior.scale
        return (-(mu ** 2).sum(1) / (2 * self.sigma_mu ** 2)
                + self.sigma_sigma * (torch.log(sigma) - sigma).sum(1))

    # ---- training objective (Eq. 6 + 8 + 9) ------------------------------------------------
    def elbo(self, X, b, m):
        q, _ = self.proposal(self.proposal_input(X, b, m))
        p, hidden = self.prior(self.prior_input(X, b))
        z = q.rsample()                                     # reparameterisation trick
        params = self.generator(z, hidden, self.prior_input(X, b))
        target = b * (1 - m)                                # hidden on purpose and truly known
        rec = self.log_likelihood(params, X, target)
        kl = torch.distributions.kl_divergence(q, p).sum(1)
        return rec - kl + self.prior_regulariser(p)

    @torch.no_grad()
    def iwae(self, X, b, m, samples=25):
        """Importance-weighted estimate of log p(x_b | x_{1-b}, b), used to pick the best epoch."""
        q, _ = self.proposal(self.proposal_input(X, b, m))
        p, hidden = self.prior(self.prior_input(X, b))
        cond = self.prior_input(X, b)
        target = b * (1 - m)
        logw = []
        for _ in range(samples):
            z = q.sample()
            params = self.generator(z, hidden, cond)
            logw.append(self.log_likelihood(params, X, target)
                        + p.log_prob(z).sum(1) - q.log_prob(z).sum(1))
        return torch.logsumexp(torch.stack(logw, 1), 1) - math.log(samples)

    # ---- imputation ------------------------------------------------------------------------
    @torch.no_grad()
    def impute_samples(self, X, num_samples):
        """For each row draw num_samples latent vectors from the PRIOR and return, per sample,
        the most likely value of every feature (Gaussian mean / most probable category).
        Observed values are copied through unchanged. Shape: [n, num_samples, d]."""
        m = torch.isnan(X).float()
        p, hidden = self.prior(self.prior_input(X, m))
        cond = self.prior_input(X, m)
        out = []
        for _ in range(num_samples):
            params = self.generator(p.sample(), hidden, cond)
            values, col = [], 0
            for k in self.sizes:
                if k == 1:
                    values.append(params[:, col])
                    col += 1
                else:
                    values.append(params[:, col:col + k].argmax(1).float())
                    col += k
            sample = torch.stack(values, 1)
            out.append(torch.where(torch.isnan(X), sample, X))
        return torch.stack(out, 1)


# --------------------------------------------------------------------------------------------
# Training loop and the full imputation procedure
# --------------------------------------------------------------------------------------------
DEFAULTS = dict(width=256, depth=4, latent=32, lr=3e-4, batch_size=64, hide_prob=0.2,
                val_ratio=0.25, max_val_rows=2000, max_train_rows=None, steps=2000,
                min_epochs=20, max_epochs=2000, checks=50, iwae_samples=25,
                num_imputations=25, seed=0, missing_rows_only=0)


def training_mask(X, hide_prob, generator):
    """b = truly missing OR hidden with probability hide_prob (App. A.2)."""
    m = torch.isnan(X).float()
    hide = (torch.rand(X.shape, generator=generator) < hide_prob).float()
    return torch.maximum(m, hide), m


def fit_and_impute(df, is_cat, config=None, log=print):
    """Train VAEAC on one incomplete DataFrame and return (imputed DataFrame, info dict)."""
    cfg = dict(DEFAULTS, **(config or {}))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codec = FeatureCodec(is_cat).fit(df)
    X = torch.from_numpy(codec.encode(df))
    n = len(X)

    # Train / validation split of the rows (validation picks the best epoch).
    perm = np.random.permutation(n)
    n_val = min(int(math.ceil(cfg["val_ratio"] * n)), cfg["max_val_rows"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if cfg["max_train_rows"] and len(train_idx) > cfg["max_train_rows"]:
        train_idx = train_idx[:cfg["max_train_rows"]]        # scalability: train on a subsample
    X_train, X_val = X[train_idx].to(device), X[val_idx].to(device)
    # Fixed validation masks so every epoch is scored on the same hidden values.
    b_val, m_val = training_mask(X[val_idx], cfg["hide_prob"], gen)
    b_val, m_val = b_val.to(device), m_val.to(device)

    model = VAEAC(codec.sizes, cfg["width"], cfg["depth"], cfg["latent"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    bs = cfg["batch_size"]
    batches_per_epoch = max(1, math.ceil(len(train_idx) / bs))
    epochs = int(min(cfg["max_epochs"], max(cfg["min_epochs"],
                                            math.ceil(cfg["steps"] / batches_per_epoch))))
    check_every = max(1, epochs // cfg["checks"])

    best_score, best_state, history = -float("inf"), None, []
    running, seen = 0.0, 0          # average training loss since the last validation check
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_idx), generator=gen)
        for s in range(0, len(order), bs):
            batch = X_train[order[s:s + bs].to(device)]
            b, m = training_mask(batch.cpu(), cfg["hide_prob"], gen)
            loss = -model.elbo(batch, b.to(device), m.to(device)).mean() / len(codec.sizes)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
            seen += 1
        if epoch % check_every == 0 or epoch == epochs:
            model.eval()
            score = float(torch.cat([model.iwae(X_val[i:i + 512], b_val[i:i + 512], m_val[i:i + 512],
                                                cfg["iwae_samples"])
                                     for i in range(0, len(X_val), 512)]).mean())
            history.append((epoch, running / max(seen, 1), score))
            running, seen = 0.0, 0
            if score >= best_score:
                best_score, best_state = score, deepcopy(model.state_dict())
    train_time = time.time() - t0

    # Imputation with the best checkpoint.
    model.load_state_dict(best_state)
    model.eval()
    t1 = time.time()
    # Scalability: optionally skip complete rows (nothing to fill there), and aggregate the
    # samples chunk by chunk so memory stays O(chunk x samples x d) instead of O(n x samples x d).
    rows = torch.isnan(X).any(1).nonzero().flatten() if cfg["missing_rows_only"] else torch.arange(n)
    agg = np.nan_to_num(X.numpy().copy())                    # complete rows keep their own codes
    for i in range(0, len(rows), 1024):
        idx = rows[i:i + 1024]
        samples = model.impute_samples(X[idx].to(device), cfg["num_imputations"]).cpu()
        part = samples.mean(1).numpy()                       # numerical: mean of the samples
        for j, k in enumerate(codec.sizes):
            if k > 1:                                        # categorical: majority vote
                codes = samples[:, :, j].long().numpy()
                part[:, j] = [np.bincount(r, minlength=k).argmax() for r in codes]
        agg[idx.numpy()] = part
    imputed = codec.decode(agg, df)
    impute_time = time.time() - t1

    info = dict(epochs=epochs, steps=epochs * batches_per_epoch,
                best_epoch=max(history, key=lambda h: h[2])[0], best_val_iwae=best_score,
                train_rows=len(train_idx), val_rows=len(val_idx),
                params=sum(p.numel() for p in model.parameters()),
                train_s=round(train_time, 2), impute_s=round(impute_time, 3),
                history=history, device=device, config=cfg)
    return imputed, info
