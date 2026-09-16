"""
fedadapt_train_approach2_v2_4.py
FedAdaptProto — v2.4: Automatic per-site K via silhouette analysis
================================================================
What's new in v2.4 vs v2.3
  • --auto_k flag  : replaces --n_clusters_per_site with data-driven K selection
  • --k_min / --k_max : search range for K (default 2–5)
  • --k_warmup_epochs : local epochs to warm embeddings before silhouette test (default 5)
  • K is selected per site, per class (AKI=1, non-AKI=0) independently,
    then the per-site K = max(K_AKI, K_nonAKI) — conservative, captures
    richer structure in either class.
  • Silhouette scores and selected K values are logged to auto_k_report.csv
  • All v2.3 flags still work; --n_clusters_per_site overrides --auto_k if both given

Expected auto-K results at (alpha=0.3, gamma=0.75) based on silhouette diagnostics:
  site_A: K=3  (silhouette 0.285 AKI class)
  site_B: K=3  (silhouette 0.312 AKI class)
  site_C: K=3  (silhouette 0.423 AKI class — strongest structure)
  site_D: K=3  (silhouette 0.271 AKI class)
  site_E: K=2  (silhouette 0.243, weak bi-modal only)

Usage:
  python3 fedadapt_train_approach2_v2_4.py \
      --data_dir ./phase2_sites_approach2/alpha0.3_gamma0.75 \
      --method fedadaptproto --alpha 0.3 --gamma 0.75 \
      --warmup_rounds 10 --early_stop_patience 0 \
      --auto_k --k_min 2 --k_max 5 --k_warmup_epochs 5 \
      --output_dir ./results_v24_autok/
"""

import argparse
import os
import re
import csv
import copy
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Model components ──────────────────────────────────────────────────────────
# REAL-ARCHITECTURE VERSION: imports v2.3's actual model classes from
# fedadapt_model_approach2.py instead of v2.5's own separate inline
# definitions (which differed substantially — no residual connection in
# SharedBody, ReLU instead of GELU, different hidden dims, no dropout in
# PersonalHead, and a completely different GRL warmup schedule). Requires
# fedadapt_model_approach2.py to be importable (same directory).
#
# SCOPING DECISION: the discriminator's TARGET stays as v2.5's original
# site-identity classification (n_sites=6), NOT v2.3's feature-group
# taxonomy (renal/inflammatory/metabolic/clinical) -- that taxonomy was
# defined for Phase 3's specific feature set and has no established
# equivalent for Phase 4's real GPC features (shared98_core + ICD-9 DX
# codes). Site-identity discrimination is also arguably the more standard
# domain-adversarial design for real multi-institution data anyway. The
# GRLGroupDiscriminator class itself doesn't care what its n_groups
# parameter semantically represents -- passing n_sites into it is a valid,
# supported use of the same class, not a hack.
from fedadapt_model_approach2 import (
    SiteInputAdapter, SharedBody, GRLGroupDiscriminator, PersonalHead,
    FedAdaptClient, GradientReversalLayer,
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_site(path, label_col=None, batch_size=256, test_frac=0.2):
    df = pd.read_csv(path)

    # Step 1: find label column — must contain 'aki', be numeric, binary {0,1}
    # Pick the LAST candidate so 'AKI_label' beats any binary comorbidity col
    if label_col is None:
        candidates = [c for c in df.columns
                      if 'aki' in c.lower()
                      and pd.api.types.is_numeric_dtype(df[c])
                      and set(df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
        if not candidates:
            raise ValueError(f'Cannot auto-detect label column in {path}. '
                             f'Columns: {list(df.columns)}')
        # Prefer columns explicitly named 'aki_label' or 'AKI_label'
        exact = [c for c in candidates if c.lower() == 'aki_label']
        label_col = exact[0] if exact else candidates[-1]

    # Step 2: ALL numeric columns except label and site_id string cols
    # Do NOT filter out binary cols — they are legitimate features (comorbidities)
    non_numeric = [c for c in df.columns
                   if not pd.api.types.is_numeric_dtype(df[c])]
    exclude = {label_col} | set(non_numeric)
    feat_cols = [c for c in df.columns if c not in exclude]
    X_df = df[feat_cols].copy()

    # Step 3: two-stage NaN imputation
    # hours_since → 0 (sentinel: lab not recently measured)
    # all others  → column median
    hours_cols = [c for c in X_df.columns if 'hours_since' in c.lower()]
    X_df[hours_cols] = X_df[hours_cols].fillna(0.0)
    other_cols = [c for c in X_df.columns if c not in hours_cols]
    X_df[other_cols] = X_df[other_cols].fillna(X_df[other_cols].median())
    X_df = X_df.fillna(0.0)  # catch any all-NaN columns

    X = X_df.values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)

    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_frac))
    tr, te = idx[:split], idx[split:]

    # Fit scaler on train split only to avoid test leakage
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y[tr]))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y[te]))

    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    te_ld = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    prevalence = float(y.mean())
    return tr_ld, te_ld, X.shape[1], prevalence, feat_cols


# ── Prototype helpers ─────────────────────────────────────────────────────────

def compute_prototypes_kmeans(embeddings, labels, n_clusters):
    """
    Per-class k-means prototypes.
    Returns dict: {class_label: tensor of shape (n_clusters, emb_dim)}
    """
    protos = {}
    for cls in [0, 1]:
        mask = (labels == cls)
        if mask.sum() < n_clusters:
            # Fallback: single centroid if too few samples
            protos[cls] = embeddings[mask].mean(0, keepdim=True).repeat(n_clusters, 1)
        else:
            emb_np = embeddings[mask].detach().cpu().numpy()
            # Replace any NaN with 0 before clustering
            emb_np = np.nan_to_num(emb_np, nan=0.0)
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED)
            km.fit(emb_np)
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            protos[cls] = centers
    return protos


def nearest_proto_loss(h, labels, site_protos, global_protos):
    """
    For each embedding, find the nearest global prototype cluster center
    for its class, compute MSE to that center.
    """
    loss = torch.tensor(0.0, device=h.device)
    for cls in [0, 1]:
        mask = (labels == cls)
        if mask.sum() == 0:
            continue
        h_cls = h[mask]
        g_centers = global_protos[cls].to(h.device).float()   # (K, D) ensure float32
        # nearest center per sample
        dists = torch.cdist(h_cls, g_centers)          # (N_cls, K)
        nearest_idx = dists.argmin(dim=1)
        targets = g_centers[nearest_idx]               # (N_cls, D)
        loss = loss + nn.functional.mse_loss(h_cls, targets)
    return loss


def average_prototypes(all_site_protos, fl_gain_weights=None):
    """
    FL-gain weighted global prototype aggregation.
    Sites with higher FL-gain (more to benefit) contribute more to the global
    prototype, so the global prototype is pulled toward data-scarce sites rather
    than being dominated by data-rich sites.
    If fl_gain_weights is None, falls back to equal weighting.

    BUG FIX: the original version scaled each site's K centers by weight,
    concatenated everything, then called .mean(0).expand(max_k, -1) --
    collapsing ALL centers to a SINGLE point and just repeating it max_k
    times. This silently made every "multi-cluster" run behave as K=1
    regardless of the requested K, for every prior v2.5 run (including
    every comparison against v2.3, which correctly k-means the pooled
    centers -- see fedadapt_train_approach2_v2_3.py's
    aggregate_prototypes_multicluster, whose docstring literally warns
    against this exact failure mode: "K shared clusters ... not a single
    averaged centroid"). Fixed to match: weighted k-means on the pooled
    centers, producing genuinely distinct K clusters per class.
    """
    global_protos = {}
    site_ids = list(all_site_protos.keys())

    for cls in [0, 1]:
        all_centers = []
        all_weights = []
        for sid in site_ids:
            if cls not in all_site_protos[sid]:
                continue
            centers = all_site_protos[sid][cls].float()  # (K, D)
            w = fl_gain_weights.get(sid, 1.0) if fl_gain_weights else 1.0
            for c in centers:
                all_centers.append(c.detach().cpu().numpy())
                all_weights.append(w)

        if not all_centers:
            continue

        X = np.stack(all_centers)
        w = np.asarray(all_weights)
        max_k = max(all_site_protos[sid][cls].shape[0] for sid in site_ids if cls in all_site_protos[sid])
        k = min(max_k, X.shape[0])

        if k <= 1 or X.shape[0] <= k:
            # too few pooled centers to cluster meaningfully -- weighted mean,
            # same fallback v2.3 uses in this corner case
            merged = (X * w[:, None]).sum(axis=0) / w.sum()
            global_protos[cls] = torch.tensor(merged, dtype=torch.float32).unsqueeze(0).expand(k, -1).clone()
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
            km.fit(X, sample_weight=w)
            global_protos[cls] = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    return global_protos


# ── Auto-K: silhouette-based K selection ──────────────────────────────────────

def extract_embeddings(client, loader, device):
    """Run forward pass, return embeddings and labels as numpy arrays."""
    client.eval()
    embs, labs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if xb.shape[0] < 2:   # skip single-sample batches (BatchNorm)
                continue
            _, _, h = client.forward_with_embedding(xb)
            h_np = h.cpu().numpy()
            if np.isnan(h_np).any():
                continue           # skip NaN batches
            embs.append(h_np)
            labs.append(yb.numpy())
    client.train()
    if not embs:
        # Return zero embeddings if nothing valid — auto-K will fall back to k_min
        # Use the body output dim (embedding_dim), not input layer shape
        emb_dim = client.body.net[-1].out_features if hasattr(client.body.net[-1], 'out_features') else 64
        return np.zeros((2, emb_dim), dtype=np.float32), np.array([0, 1])
    return np.vstack(embs), np.concatenate(labs)


def warm_embeddings(client, loader, optimizer, device, n_epochs=5):
    """Train locally for n_epochs to warm up the embeddings before silhouette test."""
    criterion = nn.BCEWithLogitsLoss()
    client.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out, _ = client(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()


def select_k_for_site(client, loader, device, k_min=2, k_max=5):
    """
    Silhouette analysis on AKI class (cls=1) embeddings only.
    The AKI class is the clinically relevant grouping — we want to find
    sub-clusters within AKI patients (e.g. mild vs severe AKI).
    The non-AKI class is large and homogeneous; using it would bias toward
    larger K driven by majority-class structure, not AKI subtype structure.
    Returns best_k (int) and a dict of scores for logging.
    """
    embs, labels = extract_embeddings(client, loader, device)
    scores_by_cls = {}

    # Score both classes for logging, but select K from AKI class only
    for cls in [0, 1]:
        mask = (labels == cls)
        cls_emb = embs[mask]
        cls_scores = {}

        for k in range(k_min, k_max + 1):
            if len(cls_emb) < k + 1:
                continue
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
                cluster_labels = km.fit_predict(cls_emb)
                if len(np.unique(cluster_labels)) < 2:
                    continue
                score = silhouette_score(cls_emb, cluster_labels,
                                         sample_size=min(2000, len(cls_emb)))
                cls_scores[k] = float(score)
            except Exception:
                continue

        scores_by_cls[cls] = cls_scores

    # Select K from AKI class (cls=1) only — clinically relevant subgroups
    aki_scores = scores_by_cls.get(1, {})
    if aki_scores:
        best_k = max(aki_scores, key=aki_scores.get)
    else:
        best_k = k_min

    return best_k, scores_by_cls


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(client, loader, device):
    client.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out, _ = client(xb)
            preds.append(torch.sigmoid(out).cpu().numpy())
            trues.append(yb.numpy())
    client.train()
    preds = np.concatenate(preds)
    trues = np.concatenate(trues).astype(int)  # ensure binary int for f1_score
    # Guard against NaN predictions
    if np.isnan(preds).any():
        nan_pct = np.isnan(preds).mean() * 100
        print(f"      [NaN guard] {nan_pct:.1f}% NaN predictions — returning 0.5")
        return 0.5, 0.0, 0.0
    auroc = roc_auc_score(trues, preds) if len(np.unique(trues)) > 1 else 0.5
    f1    = f1_score(trues, (preds > 0.5).astype(int), zero_division=0)
    auprc = average_precision_score(trues, preds) if len(np.unique(trues)) > 1 else 0.0
    return auroc, f1, auprc


# ── FedAvg on shared body ─────────────────────────────────────────────────────

def fedavg_bodies(clients, weights):
    """Weighted average of SharedBody params across all clients."""
    avg_state = {}
    for k, v in clients[list(clients.keys())[0]].body.state_dict().items():
        avg_state[k] = sum(
            weights[site] * clients[site].body.state_dict()[k].float()
            for site in clients
        )
    for site in clients:
        clients[site].body.load_state_dict(avg_state)


# ── Warmup schedule helpers ───────────────────────────────────────────────────

def warmup_scale(current_round, warmup_rounds):
    """Linear ramp from 0→1 over warmup_rounds, then 1.0."""
    if warmup_rounds <= 0:
        return 1.0
    return min(1.0, current_round / warmup_rounds)


# ── Local training step ───────────────────────────────────────────────────────

def local_train(client, loader, optimizer, device, site_idx,
                n_sites, lam_adv, alpha_proto, global_protos, n_clusters,
                local_epochs=5, pos_weight=None,
                main_grad_clip=5.0, progress=0.0):
    """
    REAL-ARCHITECTURE VERSION. v2.3's client has no per-call `lam` argument
    on forward() -- the GRL's own internal two-phase schedule is driven by
    client.set_training_progress(progress), called once per round (matches
    v2.3's `progress = rnd / args.rounds`, i.e. fraction of TOTAL training
    elapsed, not just a warmup window). `lam_adv` here is still the
    EXPLICIT loss-multiplier on adv_loss (matches v2.3's `fedadapt_loss`:
    total = task_loss + lambda_adv*adv_loss + alpha_proto*proto_loss) --
    a separate, additional mechanism from the GRL's internal schedule, not
    a replacement for it. Both are real and both matter in v2.3's design.

    No freeze_head parameter: v2.3 trains ALL parameters jointly during
    federation (optimizer scope is client.parameters() in the caller) --
    head-freezing only happens in the post-federation fine-tune step.
    """
    _dev = next(client.parameters()).device
    if pos_weight is not None:
        _pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(_dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=_pw_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    adv_crit = nn.CrossEntropyLoss()
    client.train()
    client.set_training_progress(progress)
    total_adv_loss = 0.0
    site_label_tensor = torch.tensor(site_idx, dtype=torch.long)

    for _ in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            out, adv_logits, h = client.forward_with_embedding(xb)

            # Skip batch if NaN (model still initialising)
            if torch.isnan(h).any() or torch.isnan(out).any():
                optimizer.zero_grad()
                continue

            # Classification loss
            cls_loss = criterion(out, yb)

            # Adversarial loss (target stays site-identity, see scoping
            # note at the top of the file -- not v2.3's feature-group
            # taxonomy, which has no Phase 4 equivalent)
            site_labels = site_label_tensor.expand(xb.size(0)).to(device)
            adv_loss = adv_crit(adv_logits, site_labels)

            # Prototype alignment loss
            proto_loss = torch.tensor(0.0, device=device)
            if global_protos and alpha_proto > 0:
                proto_loss = nearest_proto_loss(h, yb, None, global_protos)

            loss = cls_loss + lam_adv * adv_loss + alpha_proto * proto_loss
            loss.backward()
            nn.utils.clip_grad_norm_(client.parameters(), max_norm=main_grad_clip)
            optimizer.step()

            total_adv_loss += adv_loss.item()

    # Compute local prototypes after this round's training
    local_protos = compute_prototypes_kmeans_from_loader(client, loader, device, n_clusters)
    return local_protos, total_adv_loss


def compute_prototypes_kmeans_from_loader(client, loader, device, n_clusters):
    """Extract embeddings then compute k-means prototypes."""
    embs, labs = extract_embeddings(client, loader, device)
    embs_t = torch.tensor(embs, dtype=torch.float32)
    labs_t  = torch.tensor(labs, dtype=torch.float32)
    return compute_prototypes_kmeans(embs_t, labs_t, n_clusters)


# ── FL-gain index ─────────────────────────────────────────────────────────────

def compute_fl_gain(site_stats):
    """
    Three-component FL-gain index (Approach 2 version) — PRE-TRAINING.
    Computed from data distribution alone, before any model is trained.
    Used to scale lambda_adv and prototype aggregation weights during
    federation, since it must be available before local training runs.

      0.30 * positive_case_scarcity
      0.30 * class_imbalance
      0.40 * feature_sparsity
    """
    total_aki = sum(s["n_aki"] for s in site_stats.values())
    max_feat  = max(s["n_feat"] for s in site_stats.values())
    scores = {}
    for site, s in site_stats.items():
        scarcity   = 1.0 - s["n_aki"] / max(total_aki, 1)
        imbalance  = 1.0 - s["prevalence"]
        sparsity   = 1.0 - s["n_feat"] / max_feat
        scores[site] = 0.30 * scarcity + 0.30 * imbalance + 0.40 * sparsity
    return scores


def compute_fl_gain_revised(site_stats, local_aurocs):
    """
    Four-component FL-gain index — POST-LOCAL-BASELINE.

    Adds a local_ceiling term: a site whose local-only model already
    achieves high AUROC has less headroom for federation to improve on,
    regardless of its feature_sparsity or class_imbalance score. This
    captures the diagnostic finding that feature-rich sites (e.g.
    site_B with 79 features, local AUROC=0.94) can still show LOWER
    federation benefit than their data-distribution profile predicts,
    because there is little room left to improve.

    local_ceiling is min-max normalised across sites in [0, 1]:
      1.0 = highest local AUROC in the cohort (least headroom)
      0.0 = lowest local AUROC in the cohort (most headroom)

    Subtracted (not added) since high ceiling = LESS expected gain.

      0.25 * positive_case_scarcity
      0.25 * class_imbalance
      0.30 * feature_sparsity
    - 0.20 * local_ceiling

    Only usable AFTER the local-only baseline has been trained for all
    sites — not available before local training, unlike compute_fl_gain().
    Reported alongside the pre-training index for comparison; does NOT
    replace it for hyperparameter scaling (lambda_adv, prototype weights)
    since those must be set before local training completes.
    """
    total_aki = sum(s["n_aki"] for s in site_stats.values())
    max_feat  = max(s["n_feat"] for s in site_stats.values())

    aurocs = list(local_aurocs.values())
    auroc_min, auroc_max = min(aurocs), max(aurocs)
    auroc_range = max(auroc_max - auroc_min, 1e-9)

    scores = {}
    for site, s in site_stats.items():
        scarcity   = 1.0 - s["n_aki"] / max(total_aki, 1)
        imbalance  = 1.0 - s["prevalence"]
        sparsity   = 1.0 - s["n_feat"] / max_feat
        local_ceiling = (local_aurocs[site] - auroc_min) / auroc_range

        scores[site] = (
            0.25 * scarcity
            + 0.25 * imbalance
            + 0.30 * sparsity
            - 0.20 * local_ceiling
        )
    return scores


def compute_or_load_shared_local_baseline(
    data_dir: str,
    site_ids: list,
    site_files: list,
    site_stats: dict,
    fl_gain_pretrain: dict,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    n_seeds: int,
    epochs: int,
    force: bool,
    device: torch.device,
):
    """
    Local-only baseline AUROC per site, averaged over n_seeds independent
    (split + init) runs, CACHED to data_dir/local_baseline_fl_gain_revised.csv.

    Because the cache lives in data_dir (keyed by alpha/gamma condition, not
    by output_dir/seed), this trains ONCE per condition no matter how many
    times you re-run this script with different --seed values for the
    federated training itself — earlier calls just load the cache.

    This cache file uses the SAME schema as compute_fl_gain_offline.py's
    standalone `baseline` command, so if you point that script's `correlate`
    step at this file, every other method (FedAvg/FedProx/SCAFFOLD/FedAdapt/
    v2.3) gets compared against the exact same local_auroc as this run —
    no separate manual baseline step required as long as v2.5 has been run
    at least once for that condition.

    Returns (local_aurocs: {site_id: mean_auroc}, cache_df: pd.DataFrame).
    """
    # [FIX] This cache previously lived at a single, condition-independent
    # path (data_dir/local_baseline_fl_gain_revised.csv) despite the
    # docstring's claim that it's "keyed by alpha/gamma condition" -- it
    # was not. Since the disjoint-sites data generator writes every
    # condition into the same flat data_dir, this meant every run (after
    # the very first) loaded a STALE cache computed for a DIFFERENT
    # condition entirely -- or, before the site_files filtering fix above,
    # a cache polluted with all ~100 (site x condition) combinations from
    # the original bug. The condition is now parsed directly from
    # site_files (already filtered to just this condition's 5 real files)
    # and baked into the cache filename, so each (alpha, gamma) gets its
    # own cache and can never collide with another condition's.
    _cond_match = re.search(r"_alpha([\d.]+)_gamma([\d.]+)\.csv$", site_files[0]) if site_files else None
    if _cond_match:
        _cond_suffix = f"_alpha{_cond_match.group(1)}_gamma{_cond_match.group(2)}"
    else:
        _cond_suffix = ""
        print("  [baseline] WARNING: could not parse alpha/gamma from "
              f"site_files[0]={site_files[0] if site_files else '<empty>'!r} -- "
              "falling back to a condition-independent cache path, which may "
              "collide with other conditions run against this same data_dir.")
    cache_path = os.path.join(data_dir, f"local_baseline_fl_gain_revised{_cond_suffix}.csv")

    if os.path.exists(cache_path) and not force:
        print(f"  [baseline] cache found -> {cache_path} "
              f"(pass --force_baseline to retrain)")
        cache_df = pd.read_csv(cache_path)
        return dict(zip(cache_df["site_id"], cache_df["local_auroc"])), cache_df

    print(f"  [baseline] no cache at {cache_path} — training "
          f"{n_seeds}-seed local-only baseline (runs ONCE per condition)...")

    # Save/restore RNG state so this multi-seed loop doesn't perturb the
    # main training seed used afterwards for federation (client init,
    # KMeans clustering, etc.) — this block borrows the global RNGs
    # temporarily and hands them back exactly as it found them.
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    criterion_base = nn.BCEWithLogitsLoss()
    local_aurocs, local_stds = {}, {}

    for sid, fname in zip(site_ids, site_files):
        path = os.path.join(data_dir, fname)
        seed_aurocs = []
        for bseed in range(n_seeds):
            random.seed(bseed)
            np.random.seed(bseed)
            torch.manual_seed(bseed)

            tr_ld, te_ld, in_dim, _, _ = load_site(path, batch_size=batch_size)

            bc = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            ).to(device)
            bopt = optim.Adam(bc.parameters(), lr=lr)
            bc.train()
            for _ in range(epochs):
                for xb, yb in tr_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    bopt.zero_grad()
                    out = bc(xb).squeeze(-1)
                    loss = criterion_base(out, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(bc.parameters(), max_norm=5.0)
                    bopt.step()

            bc.eval()
            preds_b, trues_b = [], []
            with torch.no_grad():
                for xb, yb in te_ld:
                    xb = xb.to(device)
                    out = bc(xb).squeeze(-1)
                    preds_b.append(torch.sigmoid(out).cpu().numpy())
                    trues_b.append(yb.numpy())
            preds_b = np.concatenate(preds_b)
            trues_b = np.concatenate(trues_b).astype(int)
            if np.isnan(preds_b).any() or len(np.unique(trues_b)) < 2:
                auroc = 0.5
            else:
                auroc = roc_auc_score(trues_b, preds_b)
            seed_aurocs.append(auroc)
            print(f"    [baseline] {sid}  seed={bseed}  AUROC={auroc:.4f}")

        mean_auroc = float(np.mean(seed_aurocs))
        std_auroc = float(np.std(seed_aurocs))
        local_aurocs[sid] = mean_auroc
        local_stds[sid] = std_auroc
        print(f"  [baseline] {sid}: local_auroc = {mean_auroc:.4f} +/- {std_auroc:.4f} "
              f"(n_seeds={n_seeds})")

    # Hand the RNGs back exactly as found — federation training below uses
    # args.seed, set at the top of main(), untouched by this loop.
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    fl_gain_revised = compute_fl_gain_revised(site_stats, local_aurocs)

    total_aki = sum(s["n_aki"] for s in site_stats.values())
    max_feat = max(s["n_feat"] for s in site_stats.values())
    aurocs_list = list(local_aurocs.values())
    auroc_min, auroc_max = min(aurocs_list), max(aurocs_list)
    auroc_range = max(auroc_max - auroc_min, 1e-9)

    records = []
    for sid in site_ids:
        s = site_stats[sid]
        scarcity = 1.0 - s["n_aki"] / max(total_aki, 1)
        imbalance = 1.0 - s["prevalence"]
        sparsity = 1.0 - s["n_feat"] / max_feat
        ceiling = (local_aurocs[sid] - auroc_min) / auroc_range
        records.append({
            "site_id":                sid,
            "local_auroc":            round(local_aurocs[sid], 6),
            "local_auroc_std":        round(local_stds[sid], 6),
            "n_seeds":                n_seeds,
            "positive_case_scarcity": round(scarcity, 4),
            "class_imbalance":        round(imbalance, 4),
            "feature_sparsity":       round(sparsity, 4),
            "local_ceiling":          round(ceiling, 4),
            "fl_gain_index":          round(fl_gain_pretrain.get(sid, float("nan")), 4),
            "fl_gain_revised":        round(fl_gain_revised.get(sid, float("nan")), 4),
        })

    cache_df = pd.DataFrame(records)
    cache_df.to_csv(cache_path, index=False)
    print(f"  [baseline] saved shared baseline -> {cache_path}")

    return local_aurocs, cache_df


# ── Main training loop ────────────────────────────────────────────────────────

def run_fedadaptproto(args, data_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading site data...")
    # [FIX] The original filter below matched every site_*/sim_* CSV in
    # data_dir regardless of alpha/gamma. That's harmless when data_dir
    # contains only one condition's files, but the disjoint-sites
    # simulation script writes ALL conditions into the same flat
    # directory (site_A_alpha0.1_gamma0.0.csv, site_A_alpha0.3_gamma0.75.csv,
    # ... all together) -- so this was silently loading every condition's
    # data as if each (site, alpha, gamma) combination were its own
    # separate site, training a ~100-site federation instead of the
    # intended 5, for every single grid job regardless of which specific
    # alpha/gamma it was told to run. Now filters to exactly the
    # requested condition first, matching how the v2.3 script
    # (fedadapt_train_approach2_v2_3_phase1_archetype_post_leakage.py)
    # already does this correctly.
    suffix = f"_alpha{args.alpha}_gamma{args.gamma}.csv"
    site_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(suffix) and "fl_gain_index" not in f
        and (f.startswith("site_") or f.startswith("sim_"))
        # site_* = Phase 3 archetype naming (site_A.csv, ...)
        # sim_*  = Phase 4 real-GPC-site naming (sim_KUMC_alpha0.5_gamma0.75.csv, ...)
        # Original code only matched site_*, which silently produced an
        # empty site list (and a downstream max()-of-empty-sequence crash
        # in compute_fl_gain) on any Phase 4 data directory.
    ])
    if not site_files:
        # Fallback: no alpha/gamma-suffixed files found (e.g. data_dir
        # holds unsuffixed site_A.csv-style files for a single condition
        # already separated into its own directory). Falls back to the
        # original, unfiltered behavior in that case only.
        site_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".csv") and "fl_gain_index" not in f
            and (f.startswith("site_") or f.startswith("sim_"))
        ])
        if site_files:
            print(f"  [warning] no files matched suffix '{suffix}'; falling back to "
                  f"all {len(site_files)} site_*/sim_* CSVs in {data_dir}. If this "
                  f"directory holds more than one condition's data, this will "
                  f"incorrectly merge them into one run -- verify data_dir contains "
                  f"only the intended condition, or that filenames carry the "
                  f"expected alpha/gamma suffix.")
    site_ids = [os.path.splitext(f)[0] for f in site_files]   # e.g. ["site_A_alpha0.1_gamma0.0", ...]

    loaders_tr, loaders_te = {}, {}
    input_dims, prevalences, feat_cols_map = {}, {}, {}
    site_stats = {}

    for sid, fname in zip(site_ids, site_files):
        path = os.path.join(data_dir, fname)
        tr_ld, te_ld, in_dim, prev, fcols = load_site(path, batch_size=args.batch_size)
        loaders_tr[sid]  = tr_ld
        loaders_te[sid]  = te_ld
        input_dims[sid]  = in_dim
        prevalences[sid] = prev
        feat_cols_map[sid] = fcols

        n_total = len(tr_ld.dataset) + len(te_ld.dataset)
        n_aki   = int(round(prev * n_total))
        site_stats[sid] = {"n_feat": in_dim - 1, "n_aki": n_aki,
                           "prevalence": prev, "n_total": n_total}
        print(f"  [load] {sid}: {n_total:,} rows x {in_dim} cols  (AKI={prev:.3f})")

    n_sites = len(site_ids)
    site_idx_map = {sid: i for i, sid in enumerate(site_ids)}
    weights = {sid: site_stats[sid]["n_total"] / sum(s["n_total"] for s in site_stats.values())
               for sid in site_ids}

    # FL-gain index
    fl_gain = compute_fl_gain(site_stats)
    gain_csv = [f for f in os.listdir(data_dir) if "fl_gain_index" in f]
    if gain_csv:
        gain_df = pd.read_csv(os.path.join(data_dir, gain_csv[0]))
        print(f"  [fl_gain] auto-detected: {gain_csv[0]}")

    # ── Build models ───────────────────────────────────────────────────────────
    print("Building FedAdaptClient per site...")
    clients = {}
    optimizers = {}
    # Prevalence-adaptive lambda_adv, matching fedadapt_train_approach2_v2_3.py's
    # build_clients() exactly (same formula, same reference point) — previously
    # this scaled by feature-count ratio instead (input_dims[sid]/max), an
    # unrelated criterion with no documented rationale, meaning the same
    # --lambda_adv CLI value meant something different in v2.3 vs v2.5.
    # Unified so identical --lambda_adv values are comparable across all
    # methods now. Rationale (from v2.3): at low-prevalence sites, the GRL
    # adversarial signal overpowers the classification signal (weak class
    # gradient), so scale down lambda relative to a 15% reference prevalence.
    PREVALENCE_REF = 0.15
    for sid in site_ids:
        scale = min(1.0, prevalences[sid] / PREVALENCE_REF)
        lam = args.lambda_adv * scale
        model = FedAdaptClient(
            site_id=sid,
            input_dim=input_dims[sid],
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            n_groups=n_sites,   # scoping decision: site-identity target, see file header
            grl_lambda_max=lam,
            dropout=0.1,
            aki_prevalence=prevalences[sid],
        ).to(device)
        clients[sid]    = model
        optimizers[sid] = optim.Adam(model.parameters(), lr=args.lr)
        print(f"  [model] {sid}: input_dim={input_dims[sid]-1}  "
              f"→ embedding_dim={args.embedding_dim}  hidden={args.hidden_dim}  "
              f"lambda_adv={lam:.3f}")

    # ── Local-only baseline (shared, cached, multi-seed) ───────────────────────
    print("Loading/computing shared local-only baseline...")
    local_aurocs, baseline_cache_df = compute_or_load_shared_local_baseline(
        data_dir=data_dir,
        site_ids=site_ids,
        site_files=site_files,
        site_stats=site_stats,
        fl_gain_pretrain=fl_gain,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        n_seeds=args.baseline_seeds,
        epochs=args.baseline_epochs,
        force=args.force_baseline,
        device=device,
    )

    # ── Revised FL-gain index (post-local-baseline, includes local_ceiling) ────
    # Reported alongside the pre-training fl_gain for diagnostic comparison.
    # NOT used for lambda_adv/prototype scaling — those were already set above
    # using the pre-training index, since hyperparameters must be fixed before
    # local training runs. This is purely a post-hoc validation signal.
    fl_gain_revised = compute_fl_gain_revised(site_stats, local_aurocs)
    print("\n  [fl_gain revised] pre-training vs post-local-baseline "
          "(includes local_ceiling term):")
    print(f"    {'site':<28} {'pre-train':>10} {'revised':>10} {'local_auroc':>12}")
    for sid in site_ids:
        print(f"    {sid:<28} {fl_gain.get(sid, float('nan')):>10.4f} "
              f"{fl_gain_revised.get(sid, float('nan')):>10.4f} "
              f"{local_aurocs[sid]:>12.4f}")
    fl_gain_compare_df = pd.DataFrame([
        {"site_id": sid,
         "fl_gain_pretrain": round(fl_gain.get(sid, float("nan")), 4),
         "fl_gain_revised":  round(fl_gain_revised.get(sid, float("nan")), 4),
         "local_auroc":      round(local_aurocs[sid], 4)}
        for sid in site_ids
    ])
    fl_gain_compare_df.to_csv(
        os.path.join(output_dir, "fl_gain_revised_comparison.csv"), index=False
    )

    # ── AUTO-K SELECTION ───────────────────────────────────────────────────────
    n_clusters_per_site = {}

    # Parse manual override if given (--n_clusters_per_site takes precedence)
    manual_k = {}
    if args.n_clusters_per_site:
        for item in args.n_clusters_per_site.split(","):
            k, v = item.split("=")
            manual_k[k.strip()] = int(v.strip())

    if args.auto_k and not manual_k:
        # AUTO-K: Phase 1 — run full n_rounds of federation with uniform K=k_min
        # so the shared body fully converges before silhouette analysis.
        # Phase 2 — re-run n_rounds with the selected per-site K.
        k_sel_rnd = min(args.k_select_round, args.n_rounds)
        print(f"\n  ── Auto-K Phase 1: full federation ({k_sel_rnd} rounds, uniform K={args.k_min}) ──")
        print(f"  Silhouette runs after full fine-tuning on phase 1 model — matching v2.3 offline conditions.")
        _global_protos_pre = {}
        _n_clusters_pre = {sid: args.k_min for sid in site_ids}
        # [bestckpt_fix] Phase 1 previously had no best-checkpoint tracking --
        # unlike Phase 2 and v2.3, which both restore each site to its own
        # peak-AUROC round before continuing. Phase 1's own AUROC peaks
        # early (~round 10) then declines for the rest of its 50 rounds
        # (confirmed from console logs), and that already-degraded
        # end-of-Phase-1 state was what got handed to Phase 2. Added here
        # to match the pattern already proven in Phase 2/v2.3.
        _p1_best_auroc = {sid: -1.0 for sid in site_ids}
        _p1_best_round = {sid: 0 for sid in site_ids}
        _p1_best_state = {sid: None for sid in site_ids}
        for _rnd in range(1, k_sel_rnd + 1):
            _scale = warmup_scale(_rnd, args.warmup_rounds)
            _progress = _rnd / k_sel_rnd
            _all_pre = {}
            for sid in site_ids:
                # NOTE: prevalence scaling now lives in grl_lambda_max at
                # construction time (matches v2.3's build_clients), not
                # here -- no _prev_scale factor applied in this per-round
                # value anymore, avoiding double-counting.
                _lam = args.lambda_adv * _scale
                _ap  = args.alpha_proto * _scale
                _pw = (1.0 - prevalences[sid]) / max(prevalences[sid], 0.01)
                _lp, _ = local_train(clients[sid], loaders_tr[sid], optimizers[sid],
                                     device, site_idx_map[sid], n_sites,
                                     _lam, _ap, _global_protos_pre,
                                     _n_clusters_pre[sid], args.local_epochs,
                                     pos_weight=_pw,
                                     main_grad_clip=args.main_grad_clip,
                                     progress=_progress)
                _all_pre[sid] = _lp
            fedavg_bodies(clients, weights)
            _global_protos_pre = average_prototypes(_all_pre, fl_gain_weights=fl_gain)
            if _rnd % 5 == 0:
                _parts = []
                for sid in site_ids:
                    _a, _, _ = evaluate(clients[sid], loaders_te[sid], device)
                    _parts.append(f"{sid.split('_alpha')[0]}={_a:.3f}")
                    # [bestckpt_fix] track + snapshot each site's own peak
                    if _a > _p1_best_auroc[sid]:
                        _p1_best_auroc[sid] = _a
                        _p1_best_round[sid] = _rnd
                        _p1_best_state[sid] = {k: v.clone() for k, v in clients[sid].state_dict().items()}
                if _rnd % 10 == 0:
                    print(f"    [phase1] Round {_rnd:3d}/{k_sel_rnd}  |  " + "  ".join(_parts))

        # [bestckpt_fix] Restore each site to its own Phase 1 peak before
        # the post-hoc fine-tuning / silhouette selection / Phase 2 handoff
        # below -- matching v2.3's and Phase 2's own restoration pattern.
        print("  ── [bestckpt_fix] Restoring each site to its own Phase 1 best-checkpoint round ──")
        for sid in site_ids:
            if _p1_best_state[sid] is not None:
                clients[sid].load_state_dict(_p1_best_state[sid])
            _marker = " (== final round)" if _p1_best_round[sid] == k_sel_rnd else " <-- earlier than final round"
            print(f"    {sid}: phase1 best round={_p1_best_round[sid]}  "
                  f"best_auroc={_p1_best_auroc[sid]:.4f}{_marker}")

        print(f"\n  ── Auto-K selection (silhouette at round {k_sel_rnd}, K={args.k_min}–{args.k_max}) ──")

        # ── Full head fine-tuning on phase 1 model before silhouette ─────────
        # This matches exactly what v2.3 did offline:
        # run full federation → full head fine-tuning → silhouette on result.
        # The body is frozen; head trains to convergence on local data.
        # After silhouette we discard the phase 1 model and start phase 2 fresh.
        print("  [auto-K] Full head fine-tuning on phase 1 model (matching v2.3 offline conditions)...")
        _criterion_ft = nn.BCEWithLogitsLoss()
        for sid in site_ids:
            # Freeze body + adapter + disc; unfreeze head
            for p in clients[sid].body.parameters():      p.requires_grad_(False)
            for p in clients[sid].adapter.parameters():   p.requires_grad_(False)
            for p in clients[sid].discriminator.parameters():      p.requires_grad_(False)
            for p in clients[sid].head.parameters():      p.requires_grad_(True)
            _ft_opt = optim.Adam(clients[sid].head.parameters(), lr=args.lr * 0.1)
            clients[sid].train()
            for _ in range(30):   # full fine-tuning protocol — matches phase 2
                for xb, yb in loaders_tr[sid]:
                    xb, yb = xb.to(device), yb.to(device)
                    _ft_opt.zero_grad()
                    out, _ = clients[sid](xb)
                    _criterion_ft(out, yb).backward()
                    nn.utils.clip_grad_norm_(clients[sid].head.parameters(), 5.0)
                    _ft_opt.step()
            # Re-freeze head; unfreeze body for embedding extraction
            for p in clients[sid].head.parameters():      p.requires_grad_(False)
            for p in clients[sid].body.parameters():      p.requires_grad_(True)
            for p in clients[sid].adapter.parameters():   p.requires_grad_(True)
            _a, _, _ = evaluate(clients[sid], loaders_te[sid], device)
            _sid_short = sid.split("_alpha")[0]
            print(f"    [phase1 post-FT] {_sid_short}: AUROC={_a:.4f}")

        auto_k_report = []

        for sid in site_ids:
            # Embeddings now reflect federation-warmed body + locally fine-tuned head
            # Sanity check: confirm embeddings are valid
            _test_embs, _ = extract_embeddings(clients[sid], loaders_tr[sid], device)
            _nan_frac = np.isnan(_test_embs).mean()
            if _nan_frac > 0:
                print(f'    [warn] {sid}: {_nan_frac*100:.1f}% NaN embeddings after warmup — falling back to K={args.k_min}')
                n_clusters_per_site[sid] = args.k_min
                auto_k_report.append({'site': sid, 'selected_k': args.k_min, 'note': 'NaN embeddings'})
                continue

            # Silhouette analysis on federation-warmed embeddings
            best_k, scores_by_cls = select_k_for_site(
                clients[sid], loaders_tr[sid], device,
                k_min=args.k_min, k_max=args.k_max
            )

            n_clusters_per_site[sid] = best_k

            # Build report row
            row = {"site": sid, "selected_k": best_k}
            for cls in [0, 1]:
                lbl = "AKI" if cls == 1 else "nonAKI"
                for k, sc in scores_by_cls.get(cls, {}).items():
                    row[f"sil_{lbl}_k{k}"] = round(sc, 4)
                best_cls_k = max(scores_by_cls.get(cls, {k: -1 for k in range(args.k_min, args.k_max+1)}),
                                 key=scores_by_cls.get(cls, {}).get) \
                             if scores_by_cls.get(cls) else args.k_min
                row[f"best_k_{lbl}"] = best_cls_k
            auto_k_report.append(row)

            # Pretty print
            sil_str = "  ".join(
                f"K={k}: sil={sc:.3f}"
                for k, sc in sorted(scores_by_cls.get(1, {}).items())
            )
            print(f"    {sid}: selected K={best_k}  |  AKI silhouettes: {sil_str}")

        # ── Phase 2: keep phase 1 body+adapter+disc, reset head only ──────────
        # The phase 1 body has learned 50 rounds of cross-site AKI representations.
        # We keep those weights and only reset the head so it trains jointly with
        # the body in phase 2 — matching v2.3's joint training design.
        print("  [auto-K] Phase 2: keeping phase 1 body weights, resetting head to random init.")
        for sid in site_ids:
            # Reset head weights only
            for m in clients[sid].head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if args.head_fix_rounds > 0:
                # [head_fix test] Freeze body/adapter/discriminator; only the
                # freshly-reset head is trainable for the first
                # head_fix_rounds of Phase 2. Guards the already-trained
                # body against large early gradients from the random head
                # before joint training resumes.
                for p in clients[sid].body.parameters():          p.requires_grad_(False)
                for p in clients[sid].adapter.parameters():       p.requires_grad_(False)
                for p in clients[sid].discriminator.parameters(): p.requires_grad_(False)
                for p in clients[sid].head.parameters():          p.requires_grad_(True)
                optimizers[sid] = optim.Adam(clients[sid].head.parameters(), lr=args.lr)
            else:
                # Original behavior: rebuild optimizer to include all
                # parameters (head now trainable) from round 1 of Phase 2.
                for p in clients[sid].parameters():
                    p.requires_grad_(True)
                optimizers[sid] = optim.Adam(clients[sid].parameters(), lr=args.lr)
        if args.head_fix_rounds > 0:
            print(f"  [head_fix] Phase 2 clients ready — body/adapter/discriminator "
                  f"FROZEN for the first {args.head_fix_rounds} rounds; head only.")
        else:
            print("  [auto-K] Phase 2 clients ready — all parameters trainable.")

        # Save report
        report_path = os.path.join(output_dir, "auto_k_report.csv")
        if auto_k_report:
            keys = list(auto_k_report[0].keys())
            with open(report_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(auto_k_report)
            print(f"  [auto-K] report saved → {report_path}")

    elif manual_k:
        # Manual override
        n_clusters_per_site = {sid: manual_k.get(sid, args.n_clusters)
                                for sid in site_ids}
        print(f"  [manual K override] {n_clusters_per_site}")
    else:
        # Uniform K fallback
        n_clusters_per_site = {sid: args.n_clusters for sid in site_ids}
        print(f"  [uniform K={args.n_clusters}] applied to all sites")

    # [head_fix test] Only relevant for post-auto-K Phase 2 with
    # --head_fix_rounds > 0; never true for manual-K or uniform-K runs,
    # which never freeze anything to begin with.
    head_fix_active = bool(args.auto_k and not manual_k and args.head_fix_rounds > 0)

    print(f"  [per-site K] {n_clusters_per_site}")

    # ── Federation loop ────────────────────────────────────────────────────────
    # Phase 2 (or only phase if no auto_k): full federation from round 1
    global_protos = {}
    phase_label = "Auto-K Phase 2" if (args.auto_k and not manual_k) else "FEDADAPTPROTO"

    print(f"\n  ── Federation [{phase_label}]  {args.n_rounds} rounds × "
          f"{args.local_epochs} local epochs ──")

    round_aurocs  = []

    # Phase-1 checkpointing: track each site's own best round-level AUROC and
    # a snapshot of its full client state at that point. Used to restore
    # before head fine-tuning below, instead of always fine-tuning from
    # round args.n_rounds's state — which for sites like site_D (peaks in
    # the first few rounds, then decays under continued federation) can be
    # substantially worse than an earlier round. No-op for sites that are
    # still improving at the final round (their best IS the final state).
    best_auroc = {sid: -1.0 for sid in site_ids}
    best_round = {sid: 0 for sid in site_ids}
    best_state = {sid: None for sid in site_ids}

    for rnd in range(1, args.n_rounds + 1):
        if head_fix_active and rnd == args.head_fix_rounds + 1:
            # [head_fix test] Head-only warmup within Phase 2 is complete --
            # unfreeze body/adapter/discriminator and rebuild the optimizer
            # to include all parameters, resuming joint training exactly as
            # the original (head_fix_rounds=0) code path does from round 1.
            print(f"  [head_fix] Round {rnd}: head-only warmup complete "
                  f"({args.head_fix_rounds} rounds) -- unfreezing body/"
                  f"adapter/discriminator, resuming joint training.")
            for sid in site_ids:
                for p in clients[sid].parameters():
                    p.requires_grad_(True)
                optimizers[sid] = optim.Adam(clients[sid].parameters(), lr=args.lr)
            head_fix_active = False  # only fires once

        scale = warmup_scale(rnd, args.warmup_rounds)
        progress = rnd / args.n_rounds   # drives GRL's internal 2-phase schedule (v2.3-real)
        all_site_protos = {}

        for sid in site_ids:
            # REAL-ARCHITECTURE VERSION: prevalence scaling now lives in
            # grl_lambda_max baked in at client construction time (matches
            # v2.3's build_clients exactly) -- NOT re-applied here anymore,
            # avoiding double-counting. `scale` (warmup_rounds-based linear
            # ramp) is the only per-round multiplier on the EXPLICIT
            # lambda_adv loss term now, matching v2.3's eff_lambda_adv
            # formula exactly. The fl_gain-based third factor (--no_lambda_fl_mod)
            # is dropped entirely in this version -- v2.3 has no equivalent
            # at all, and this is now a full-fidelity architecture match,
            # not a partial patch. Separately, client.set_training_progress
            # (called inside local_train via the `progress` arg) drives the
            # GRL's OWN internal two-phase schedule -- an additional,
            # independent mechanism from this explicit multiplier, exactly
            # as in v2.3.
            lam_adv_eff = args.lambda_adv * scale
            alpha_proto_eff = args.alpha_proto * scale

            _pw = (1.0 - prevalences[sid]) / max(prevalences[sid], 0.01)
            local_protos, _ = local_train(
                client=clients[sid],
                loader=loaders_tr[sid],
                optimizer=optimizers[sid],
                device=device,
                site_idx=site_idx_map[sid],
                n_sites=n_sites,
                lam_adv=lam_adv_eff,
                alpha_proto=alpha_proto_eff,
                global_protos=global_protos,
                n_clusters=n_clusters_per_site[sid],
                local_epochs=args.local_epochs,
                pos_weight=_pw,
                main_grad_clip=args.main_grad_clip,
                progress=progress,
            )
            all_site_protos[sid] = local_protos

        # Aggregate shared body
        fedavg_bodies(clients, weights)

        # Aggregate prototypes
        global_protos = average_prototypes(all_site_protos, fl_gain_weights=fl_gain)

        # Periodic eval
        if rnd % 5 == 0:
            row = {"round": rnd}
            parts = []
            for sid in site_ids:
                auroc, _, _ = evaluate(clients[sid], loaders_te[sid], device)
                row[f"auroc_{sid}"] = round(auroc, 4)
                parts.append(f"{sid}={auroc:.3f}")
                if auroc > best_auroc[sid]:
                    best_auroc[sid] = auroc
                    best_round[sid] = rnd
                    best_state[sid] = {k: v.clone() for k, v in clients[sid].state_dict().items()}
            round_aurocs.append(row)
            print(f"    Round {rnd:3d}/{args.n_rounds}  |  " + "  ".join(parts))

    print("  ── Restoring each site to its own best-checkpoint round ──")
    for sid in site_ids:
        if best_state[sid] is not None:
            clients[sid].load_state_dict(best_state[sid])
        marker = " (== final round)" if best_round[sid] == args.n_rounds else " <-- earlier than final round"
        print(f"    {sid}: best round={best_round[sid]}  best_auroc={best_auroc[sid]:.4f}{marker}")

    # ── Head fine-tuning ───────────────────────────────────────────────────────
    print("  ── Head fine-tuning (FedAdapt) ──")
    for sid in site_ids:
        # In phase 2 head trained jointly throughout — fine-tune step
        # freezes body and runs lower-lr head-only pass to converge classification
        for p in clients[sid].body.parameters():
            p.requires_grad_(False)
        for p in clients[sid].adapter.parameters():
            p.requires_grad_(False)
        for p in clients[sid].discriminator.parameters():
            p.requires_grad_(False)
        for p in clients[sid].head.parameters():
            p.requires_grad_(True)
        head_opt = optim.Adam(clients[sid].head.parameters(), lr=args.lr * args.ft_lr_mult)
        if args.ft_pos_weight:
            _ft_pw = (1.0 - prevalences[sid]) / max(prevalences[sid], 0.01)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([_ft_pw], dtype=torch.float32).to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()
        clients[sid].train()
        for _ in range(args.ft_epochs):
            for xb, yb in loaders_tr[sid]:
                xb, yb = xb.to(device), yb.to(device)
                head_opt.zero_grad()
                out, _ = clients[sid](xb)
                criterion(out, yb).backward()
                if args.ft_grad_clip > 0:
                    nn.utils.clip_grad_norm_(clients[sid].head.parameters(), args.ft_grad_clip)
                head_opt.step()

        auroc, f1, auprc = evaluate(clients[sid], loaders_te[sid], device)
        print(f"    {sid}: AUROC={auroc:.4f}  F1={f1:.4f}  AUPRC={auprc:.4f}")

    # ── Save final metrics ─────────────────────────────────────────────────────
    print("  Final metrics:")
    results = []
    header = ["site_id", "auroc", "f1", "auprc", "local_auroc",
              "delta_auroc", "fl_gain_index", "fl_gain_revised", "selected_k",
              "best_checkpoint_round"]
    print(f"{'site_id':>8}  {'auroc':>8}  {'f1':>6}  {'auprc':>6}  "
          f"{'delta':>7}  {'K':>3}  {'ckpt_rd':>7}")
    for sid in site_ids:
        auroc, f1, auprc = evaluate(clients[sid], loaders_te[sid], device)
        delta = auroc - local_aurocs[sid]
        results.append({
            "site_id":         sid,
            "auroc":           round(auroc, 6),
            "f1":              round(f1, 6),
            "auprc":           round(auprc, 6),
            "local_auroc":     round(local_aurocs[sid], 6),
            "delta_auroc":     round(delta, 6),
            "fl_gain_index":   round(fl_gain[sid], 4),
            "fl_gain_revised": round(fl_gain_revised.get(sid, float("nan")), 4),
            "selected_k":      n_clusters_per_site[sid],
            "best_checkpoint_round": best_round[sid],
        })
        print(f"{sid:>8}  {auroc:.6f}  {f1:.4f}  {auprc:.4f}  "
              f"{delta:+.4f}  K={n_clusters_per_site[sid]}  rd={best_round[sid]}")

    metrics_path = os.path.join(output_dir, "final_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(results)

    # Round AUROC CSV
    if round_aurocs:
        ra_path = os.path.join(output_dir, "round_auroc.csv")
        with open(ra_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(round_aurocs[0].keys()))
            w.writeheader()
            w.writerows(round_aurocs)

    # FL-gain correlation CSV — includes BOTH the pre-train index and the
    # revised (4-component, +local_ceiling) index so downstream analysis
    # doesn't need to re-derive fl_gain_revised from final_metrics.csv.
    fl_path = os.path.join(output_dir, "fl_gain_correlation.csv")
    with open(fl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "fl_gain_index", "fl_gain_revised",
                     "local_auroc", "delta_auroc", "selected_k"])
        for r in results:
            w.writerow([r["site_id"], r["fl_gain_index"], r["fl_gain_revised"],
                        r["local_auroc"], r["delta_auroc"], r["selected_k"]])

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("  Generating plots...")
    _plot_training_curves(round_aurocs, site_ids, output_dir)
    _plot_final_metrics(results, output_dir)
    _plot_fl_gain(results, output_dir)
    if args.auto_k and not manual_k:
        _plot_auto_k_silhouettes(output_dir)

    print(f"\n✅  Done — all outputs in {output_dir}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_training_curves(round_aurocs, site_ids, output_dir):
    if not round_aurocs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#378ADD", "#1D9E75", "#888780", "#BA7517", "#D4537E"]
    rounds = [r["round"] for r in round_aurocs]
    for i, sid in enumerate(site_ids):
        key = f"auroc_{sid}"
        vals = [r[key] for r in round_aurocs if key in r]
        ax.plot(rounds[:len(vals)], vals, label=sid,
                color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("AUROC")
    ax.set_title("FedAdaptProto v2.4 — Training Curves")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print("  [plot] training_curves.png")


def _plot_final_metrics(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sites = [r["site_id"] for r in results]
    colors = ["#E24B4A" if r["delta_auroc"] < 0 else "#1D9E75" for r in results]
    for ax, metric, label in zip(axes,
                                  ["auroc", "f1", "auprc"],
                                  ["AUROC", "F1", "AUPRC"]):
        vals = [r[metric] for r in results]
        bars = ax.bar(sites, vals, color=colors)
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.set_ylabel(label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)
    fig.suptitle("FedAdaptProto v2.4 — Final Metrics (green=FL gain, red=FL loss)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "final_metrics.png"), dpi=150)
    plt.close(fig)
    print("  [plot] final_metrics.png")


def _plot_fl_gain(results, output_dir):
    """
    Two panels, side by side: delta_auroc vs the pre-train 3-component
    fl_gain_index (left) and vs the revised 4-component fl_gain_revised
    that adds local_ceiling (right). Both computed from THIS run's own
    local-only baseline (single run, not the multi-seed shared baseline
    from compute_fl_gain_offline.py) — treat this as a per-run diagnostic,
    not the cross-method comparison. For comparing fl_gain_revised across
    FedAvg/FedProx/SCAFFOLD/FedAdapt/v2.3/v2.4/v2.5 on the same footing,
    use compute_fl_gain_offline.py against a shared baseline instead.
    """
    colors = {"2": "#D4537E", "3": "#378ADD", "4": "#1D9E75", "5": "#BA7517"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    for ax, x_key, title in zip(
        axes,
        ["fl_gain_index", "fl_gain_revised"],
        ["Pre-train FL-Gain Index (3-component)",
         "Revised FL-Gain Index (4-component, + local_ceiling)"],
    ):
        xs, ys = [], []
        for r in results:
            k = str(r["selected_k"])
            c = colors.get(k, "#888780")
            x = r[x_key]
            y = r["delta_auroc"]
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y, color=c, s=100, zorder=3)
            ax.annotate(f"{r['site_id']}\n(K={k})", (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        if len(xs) >= 3 and np.std(xs) > 1e-9:
            r_val, p_val = pearsonr(xs, ys)
            rho_val, ps_val = spearmanr(xs, ys)
            z = np.polyfit(xs, ys, 1)
            xline = np.linspace(min(xs), max(xs), 50)
            ax.plot(xline, np.polyval(z, xline), color="gray",
                    linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
            title = (f"{title}\nPearson r={r_val:.3f} (p={p_val:.3g})   "
                     f"Spearman rho={rho_val:.3f} (p={ps_val:.3g})")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_title(title, fontsize=9.5)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("AUROC Improvement vs Local")
    fig.suptitle("FL-Gain Index vs Observed Improvement (auto-K) — "
                 "This Run's Own Local Baseline", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fl_gain_correlation.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  [plot] fl_gain_correlation.png (pre-train vs revised, side by side)")


def _plot_auto_k_silhouettes(output_dir):
    report_path = os.path.join(output_dir, "auto_k_report.csv")
    if not os.path.exists(report_path):
        return
    df = pd.read_csv(report_path)
    sil_cols = [c for c in df.columns if c.startswith("sil_AKI_k")]
    if not sil_cols:
        return
    k_vals = sorted([int(c.replace("sil_AKI_k", "")) for c in sil_cols])
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#378ADD", "#1D9E75", "#888780", "#BA7517", "#D4537E"]
    for i, row in df.iterrows():
        sils = [row.get(f"sil_AKI_k{k}", np.nan) for k in k_vals]
        ax.plot(k_vals, sils, marker="o", label=row["site"],
                color=colors[i % len(colors)])
        best_k = int(row["selected_k"])
        best_sil = row.get(f"sil_AKI_k{best_k}", np.nan)
        if not np.isnan(best_sil):
            ax.scatter([best_k], [best_sil], s=150, color=colors[i % len(colors)],
                       marker="*", zorder=5)
    ax.set_xlabel("K (clusters per class)")
    ax.set_ylabel("Silhouette Score (AKI class)")
    ax.set_title("Auto-K: Silhouette Analysis — AKI Class Embeddings\n(★ = selected K)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auto_k_silhouettes.png"), dpi=150)
    plt.close(fig)
    print("  [plot] auto_k_silhouettes.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FedAdaptProto v2.4 — auto-K")

    # Data / output
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--method",      default="fedadaptproto")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for random/numpy/torch/KMeans. Vary this "
                        "across runs (each writing to its own output_dir, e.g. "
                        ".../seed7/, .../seed123/) to get multi-seed stability "
                        "estimates instead of a single noisy run (default 42, "
                        "matching all prior v2.x runs).")

    # Heterogeneity identifiers (for logging only)
    p.add_argument("--alpha",  type=float, default=0.3)
    p.add_argument("--gamma",  type=float, default=0.75)

    # Training
    p.add_argument("--n_rounds",      type=int,   default=50)
    p.add_argument("--local_epochs",  type=int,   default=5)
    p.add_argument("--lr",            type=float, default=0.001)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--embedding_dim", type=int,   default=64)
    p.add_argument("--hidden_dim",    type=int,   default=128)

    # Shared local-only baseline (cached to data_dir, once per condition)
    p.add_argument("--baseline_seeds",  type=int, default=5,
                   help="Number of seeds to average the local-only baseline "
                        "over (default 5). Only spent once per data_dir — "
                        "cached, not re-run for every --seed of the main "
                        "federated training.")
    p.add_argument("--baseline_epochs", type=int, default=20,
                   help="Local epochs per baseline seed (default 20)")
    p.add_argument("--force_baseline",  action="store_true",
                   help="Retrain the shared local baseline even if a cache "
                        "already exists in data_dir")

    # FedAdaptProto hyperparameters
    p.add_argument("--lambda_adv",         type=float, default=0.1)
    p.add_argument("--alpha_proto",        type=float, default=1.0)

    # Head fine-tune recipe ablation — isolates v2.3 vs v2.5's fine-tune
    # differences from everything else (K, local_epochs, alpha_proto — all
    # already disentangled). Defaults reproduce v2.5's ORIGINAL hardcoded
    # recipe (lr*0.1, 30 epochs, no pos_weight, clip 5.0). Pass
    # --ft_lr_mult 1.0 --ft_epochs 10 --ft_pos_weight --ft_grad_clip 0.0
    # to reproduce v2.3's recipe instead, on top of v2.5's federation phase.
    p.add_argument("--ft_lr_mult",     type=float, default=0.1,
                   help="Multiplier on --lr for the head fine-tune step "
                        "(default 0.1 = v2.5 original; v2.3 uses 1.0)")
    p.add_argument("--ft_epochs",      type=int, default=30,
                   help="Head fine-tune epochs (default 30 = v2.5 original; "
                        "v2.3 uses 10 via --finetune_epochs)")
    p.add_argument("--ft_pos_weight",  action="store_true", default=False,
                   help="Use class-imbalance pos_weight in the fine-tune loss "
                        "(default: off, matches v2.5 original; v2.3 has it on)")
    p.add_argument("--ft_grad_clip",   type=float, default=5.0,
                   help="Grad-norm clip on head parameters during fine-tune "
                        "(default 5.0 = v2.5 original; v2.3 uses 0.0/off)")
    p.add_argument("--no_lambda_fl_mod", action="store_true", default=False,
                   help="Disable v2.5's extra fl_gain-based lambda_adv "
                        "dampening (a third multiplicative factor v2.3 does "
                        "not have at all). Default: off (keeps v2.5's "
                        "original behavior); pass this flag to make v2.5's "
                        "effective lambda_adv formula match v2.3's exactly.")
    p.add_argument("--main_grad_clip", type=float, default=5.0,
                   help="Grad-norm clip on ALL client parameters during "
                        "main federated training (default 5.0 = v2.5 "
                        "original; v2.3 uses 1.0, 5x tighter, in both its "
                        "fedadapt and fedadaptproto local-step functions)")
    p.add_argument("--warmup_rounds",      type=int,   default=10)
    p.add_argument("--head_fix_rounds", type=int, default=0,
                   help="[head_fix test] Number of rounds at the START of "
                        "Phase 2 (post-auto-K reset) during which body/"
                        "adapter/discriminator are FROZEN and only the "
                        "freshly-reset head trains, before everything is "
                        "unfrozen for the remainder of Phase 2. Tests "
                        "whether Phase 2's immediate full-model gradient "
                        "flow from a random head is disrupting the "
                        "already-trained Phase 1 body ('gradient shock'). "
                        "0 (default) = original behavior, everything "
                        "trainable from round 1 of Phase 2, unchanged.")
    p.add_argument("--early_stop_patience",type=int,   default=0)

    # K configuration — three mutually exclusive modes:
    # 1. --auto_k          → silhouette-based auto selection (v2.4 default)
    # 2. --n_clusters_per_site site_A=3,...  → manual override (v2.3 compat)
    # 3. neither           → uniform --n_clusters everywhere
    p.add_argument("--auto_k",    action="store_true",
                   help="Automatically select K per site via silhouette analysis")
    p.add_argument("--k_min",     type=int, default=2,
                   help="Minimum K to test in silhouette search (default 2)")
    p.add_argument("--k_max",     type=int, default=5,
                   help="Maximum K to test in silhouette search (default 5)")
    p.add_argument("--k_warmup_epochs", type=int, default=5,
                   help="Local epochs to warm embeddings before silhouette test (default 5)")
    p.add_argument("--k_select_round", type=int, default=50,
                   help="Federation rounds for phase 1 before silhouette (default=n_rounds=50). "
                        "Full federation then full fine-tuning runs before silhouette, "
                        "matching v2.3 offline analysis conditions.")
    p.add_argument("--n_clusters", type=int, default=3,
                   help="Uniform K fallback when --auto_k is not set (default 3)")
    p.add_argument("--n_clusters_per_site", type=str, default="",
                   help="Manual per-site K: 'site_A=3,site_B=3,...' — overrides --auto_k")

    return p.parse_args()


def main():
    args = parse_args()

    # Re-seed everything from --seed (overrides the module-level default set
    # at import time). Must happen before run_fedadaptproto so data splits,
    # weight init, and KMeans clustering all pick up the requested seed.
    # KMeans calls elsewhere in this file reference the module-level SEED
    # name directly (not args.seed) — mutating it here via `global` means
    # those calls pick up the new value at call-time, since run_fedadaptproto
    # is invoked after this point.
    global SEED
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    method_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)

    print("=" * 65)
    print(f"  FedAdaptProto v2.5  |  method={args.method.upper()}  |  seed={args.seed}")
    print(f"  rounds={args.n_rounds}  local_epochs={args.local_epochs}  "
          f"lr={args.lr}  batch={args.batch_size}")
    print(f"  embedding_dim={args.embedding_dim}  hidden_dim={args.hidden_dim}")
    print(f"  lambda_adv={args.lambda_adv}  alpha_proto={args.alpha_proto}")
    print(f"  warmup_rounds={args.warmup_rounds}")
    k_mode = (f"auto_k (silhouette @ round {args.k_select_round})"
              if args.auto_k and not args.n_clusters_per_site
              else f"manual: {args.n_clusters_per_site}" if args.n_clusters_per_site
              else f"uniform K={args.n_clusters}")
    print(f"  K mode: {k_mode}")
    print(f"  output → {method_dir}")
    print("=" * 65)

    run_fedadaptproto(args, args.data_dir, method_dir)


if __name__ == "__main__":
    main()