"""
fedadapt_train.py
=================
Unified training loop for federated AKI prediction experiments.

Supports four methods via --method flag:
    fedadapt  — proposed: input adapter + shared body + GRL + personal head
    fedavg    — FedAvg baseline: adapter + shared body + shared head (no GRL)
    fedprox   — FedProx baseline: fedavg + proximal penalty on body weights
    scaffold  — SCAFFOLD baseline: fedavg + control variates for client drift

All methods share:
    - SiteInputAdapter (local, never federated) — handles variable feature heterogeneity
      (Approach 2: prefix-based feature matching for stat-suffix columns)
    - SharedBody (federated via weighted FedAvg)
    - Same data loading, train/test split, evaluation metrics

FedAdapt additionally uses:
    - GRLGroupDiscriminator (local adversarial head, group-invariant embedding)
    - PersonalHead (local classifier, fine-tuned after federation)

OUTPUTS
-------
results/<method>/
    round_auroc.csv          — per-site AUROC at each federation round
    final_metrics.csv        — AUROC, F1, AUPRC per site after head fine-tuning
    fl_gain_correlation.png  — FL-gain index vs AUROC improvement (Phase III plot)
    training_curves.png      — per-site AUROC across rounds
    loss_curves.png          — per-site task + adv loss across rounds
    site_<X>_model.pt        — saved client model per site

USAGE
-----
# FedAdapt (proposed)
python fedadapt_train.py --data_dir ./phase2_sites/ --method fedadapt

# All baselines in one go (run sequentially)
for METHOD in fedavg fedprox scaffold fedadapt; do
    python fedadapt_train.py --data_dir ./phase2_sites/ --method $METHOD
done

KEY HYPERPARAMETERS
-------------------
--rounds        T   Federation rounds            (default 50)
--local_epochs  K   Local epochs per round       (default 5)
--finetune_epochs E Head fine-tune epochs        (default 10, FedAdapt only)
--lr            float Learning rate              (default 1e-3)
--batch_size    int  Mini-batch size             (default 256, CPU-safe)
--embedding_dim int  Shared embedding dim        (default 64, must match sim)
--hidden_dim    int  MLP hidden dim              (default 128)
--lambda_adv    float GRL adversarial weight     (default 0.5, FedAdapt only)
--mu            float FedProx proximal weight    (default 0.01)
--test_frac     float Local held-out test split  (default 0.20)
--seed          int  Random seed                 (default 42)
--fl_gain_csv   path FL-gain index CSV from sim  (auto-detected if present)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ── import model components ───────────────────────────────────────────────────
# Assumes fedadapt_model.py is in the same directory or on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))
from fedadapt_model_approach2 import (
    FedAdaptClient,
    FedAdaptServer,
    SharedBody,
    fedadapt_loss,
    fedprox_penalty,
    compute_local_prototypes,
    prototype_alignment_loss,
)

# ─── FEATURE GROUP DEFINITIONS (must match simulation) ────────────────────────

# APPROACH 2: prefix-based feature groups matching Approach 2 column naming
# e.g. "creatinine" matches creatinine_most_recent, creatinine_min etc.
# baseline_method excluded (string column)
FEATURE_GROUPS = {
    "renal":        [
        # hours_to_anchor removed: confirmed real, computed leakage-risk
        # column (see mimic_ftl_simulation_phase2_approach2.py for full
        # reasoning -- same asymmetric-anchor logic confirmed leaky for
        # Phase 4 in AKI_Anchor_Based_Approach2_aligned_features_PHASE4_
        # leakage_fixed.ipynb, Step 15B).
        "baseline_scr", "creatinine", "bun",
    ],
    "inflammatory": ["lactate", "wbc", "platelets"],
    "metabolic":    ["sodium", "potassium", "bicarbonate",
                     "hemoglobin", "glucose", "albumin", "bilirubin"],
    "hemodynamic":  ["sbp", "dbp", "heart_rate", "spo2",
                     "resp_rate", "temperature", "gcs_total"],
    "clinical":     ["admission_type", "gender", "age_at_admission",
                     "has_diabetes", "has_hypertension", "has_chf",
                     "has_sepsis", "has_liver_disease", "has_cancer",
                     "nephrotoxic_flag", "nephrotoxic_count", "n_distinct_meds"],
}
GROUP_INDEX = {g: i for i, g in enumerate(FEATURE_GROUPS)}
N_GROUPS    = len(FEATURE_GROUPS)   # 5


# ─── SEEDS ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # manual_seed alone does NOT guarantee determinism — several common ops
    # (including some used by the KMeans-based prototype clustering path)
    # remain non-deterministic without this. Investigated as the likely
    # cause of a same-seed-same-code reproducibility issue found earlier
    # this session, concentrated on site_C/D specifically (fedadaptproto's
    # KMeans clustering step). warn_only=True rather than a hard failure,
    # since forcing strict determinism can raise on an op with no
    # deterministic implementation available on some backends (e.g. MPS).
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

class SiteData:
    """
    Holds train/test tensors, DataLoaders, and metadata for one site.

    Group label assignment:
        Each sample is assigned the index of the dominant feature group —
        the group whose columns have the fewest missing values in that row.
        This gives the GRL discriminator a meaningful group signal even
        when sites have overlapping feature sets.
    """

    def __init__(
        self,
        site_id:    str,
        meta:       dict,
        df:         pd.DataFrame,
        label_col:  str,
        test_frac:  float,
        batch_size: int,
        seed:       int,
    ):
        self.site_id        = site_id
        self.meta           = meta
        self.feature_names  = meta["feature_names"]
        self.aki_prevalence = meta.get("aki_prevalence") or df[label_col].mean()
        self.unlabeled      = False   # site_F removed — all sites labeled
        self.n_samples      = len(df)

        # ── feature matrix ──────────────────────────────────────────────────
        feat_cols = [c for c in self.feature_names if c in df.columns]
        df_feat   = df[feat_cols].copy()

        # ── site-local median imputation ─────────────────────────────────────
        # Median is robust to right-skewed lab distributions (creatinine, BUN,
        # bilirubin, lactate). Binary flag columns (has_diabetes etc.) are
        # imputed with 0 (mode) since they are never NaN in practice.
        # All statistics are site-local — no global statistics shared (privacy).
        binary_cols  = [c for c in feat_cols if df_feat[c].dropna().isin([0, 1]).all()]
        numeric_cols = [c for c in feat_cols if c not in binary_cols]
        site_medians = df_feat[numeric_cols].median()
        df_feat[numeric_cols] = df_feat[numeric_cols].fillna(site_medians)
        df_feat[binary_cols]  = df_feat[binary_cols].fillna(0)
        X = df_feat.values.astype(np.float32)

        # ── site-local z-score standardisation ───────────────────────────────
        # Mean and std computed from this site's data only — privacy-preserving.
        # After standardisation, median-imputed values map to ~0 (neutral).
        mu  = X.mean(axis=0, keepdims=True)
        sig = X.std(axis=0, keepdims=True) + 1e-8
        X   = (X - mu) / sig

        # ── labels ──────────────────────────────────────────────────────────
        y = df[label_col].fillna(0).values.astype(np.float32)

        # ── group labels ─────────────────────────────────────────────────────
        g = self._assign_group_labels(df, feat_cols)

        # ── train / val / test split ────────────────────────────────────────
        # val split (carved out of train) used ONLY by FedAdapt(Proto) v2.1+
        # early-stopping; never seen by other methods, never used for final eval.
        rng   = np.random.default_rng(seed)
        idx   = rng.permutation(len(df))
        n_test = max(1, int(len(df) * test_frac))
        n_val  = max(1, int(len(df) * 0.10))     # 10% local val for early-stop
        test_idx = idx[:n_test]
        val_idx  = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        self.X_train = torch.tensor(X[train_idx])
        self.y_train = torch.tensor(y[train_idx])
        self.g_train = torch.tensor(g[train_idx], dtype=torch.long)
        self.X_val   = torch.tensor(X[val_idx])
        self.y_val   = torch.tensor(y[val_idx])
        self.X_test  = torch.tensor(X[test_idx])
        self.y_test  = torch.tensor(y[test_idx])

        # ── class-weighted sampler for imbalanced sites ──────────────────────
        if self.y_train.sum() > 0:
            n_pos  = self.y_train.sum().item()
            n_neg  = len(self.y_train) - n_pos
            w_pos  = len(self.y_train) / (2 * n_pos)
            w_neg  = len(self.y_train) / (2 * n_neg)
            sample_weights = torch.where(self.y_train == 1,
                                         torch.tensor(w_pos),
                                         torch.tensor(w_neg))
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                            replacement=True)
            self.train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train, self.g_train),
                batch_size=batch_size, sampler=sampler,
            )
        else:
            self.train_loader = DataLoader(
                TensorDataset(self.X_train, self.y_train, self.g_train),
                batch_size=batch_size, shuffle=True,
            )

        # ── pos_weight for BCE loss (handles class imbalance) ────────────────
        if self.y_train.sum() > 0:
            n_pos = self.y_train.sum().item()
            n_neg = len(self.y_train) - n_pos
            self.pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
        else:
            self.pos_weight = None

    def _assign_group_labels(self, df: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
        """
        Assign dominant feature group index to each row.
        APPROACH 2: uses prefix matching for group column resolution.
        """
        group_scores = np.zeros((len(df), N_GROUPS))

        for g_name, g_idx in GROUP_INDEX.items():
            # Prefix matching: col matches if it equals prefix or starts with prefix_
            cols_present = []
            for col in feat_cols:
                for prefix in FEATURE_GROUPS[g_name]:
                    if col == prefix or col.startswith(prefix + "_"):
                        cols_present.append(col)
                        break
            if not cols_present:
                continue
            sub = df[cols_present].fillna(0)
            group_scores[:, g_idx] = (sub != 0).mean(axis=1).values

        group_labels = group_scores.argmax(axis=1)
        return group_labels


def load_all_sites(
    data_dir:   Path,
    label_col:  str,
    alpha:      float,
    gamma:      float,
    test_frac:  float,
    batch_size: int,
    seed:       int,
) -> Dict[str, SiteData]:
    """
    Load all site CSVs and their adapter metadata JSONs from data_dir.
    Matches files by alpha/gamma suffix produced by mimic_ftl_simulation_phase2.py.
    """
    sites: Dict[str, SiteData] = {}

    for meta_path in sorted(data_dir.glob("*_adapter_meta.json")):
        site_id = meta_path.stem.replace("_adapter_meta", "")

        with open(meta_path) as f:
            meta = json.load(f)

        csv_path = data_dir / f"{site_id}_alpha{alpha}_gamma{gamma}.csv"
        if not csv_path.exists():
            # Try without alpha/gamma suffix (fallback)
            candidates = list(data_dir.glob(f"{site_id}*.csv"))
            if not candidates:
                print(f"  [skip] {site_id}: no CSV found")
                continue
            csv_path = candidates[0]

        df = pd.read_csv(csv_path)
        aki_rate = df[label_col].mean() if label_col in df.columns else float("nan")
        print(f"  [load] {site_id}: {len(df):,} rows × {df.shape[1]} cols  (AKI={aki_rate:.3f})")

        sites[site_id] = SiteData(
            site_id=site_id, meta=meta, df=df,
            label_col=label_col, test_frac=test_frac,
            batch_size=batch_size, seed=seed,
        )

    if not sites:
        raise FileNotFoundError(
            f"No site files found in {data_dir}. "
            f"Run mimic_ftl_simulation_phase2.py first."
        )

    return sites


# ─── CLIENT FACTORY ───────────────────────────────────────────────────────────

def build_clients(
    sites:         Dict[str, SiteData],
    embedding_dim: int,
    hidden_dim:    int,
    grl_lambda:    float,
    adaptive_lambda: bool = True,
) -> Dict[str, FedAdaptClient]:
    """
    Build one FedAdaptClient per site using adapter metadata.

    APPROACH 2 UPDATE: prevalence-adaptive lambda_adv.
    At low-prevalence sites, the GRL adversarial signal overpowers the
    classification signal (7.7% AKI at site_D has weak class gradient).
    Scale lambda_adv by site AKI prevalence relative to a reference (15%):

        effective_lambda = base_lambda * min(1.0, prevalence / 0.15)

    site_A (23.4%): lambda = base * 1.0   (full adversarial)
    site_B (11.0%): lambda = base * 0.73
    site_C ( 9.0%): lambda = base * 0.60
    site_D ( 7.7%): lambda = base * 0.51
    site_E ( 5.7%): lambda = base * 0.38  (weakest adversarial)
    """
    PREVALENCE_REF = 0.15   # reference prevalence — lambda scales linearly below this
    clients = {}
    for site_id, sd in sites.items():
        if adaptive_lambda:
            prevalence = float(sd.aki_prevalence)
            scale      = min(1.0, prevalence / PREVALENCE_REF)
            eff_lambda = grl_lambda * scale
        else:
            eff_lambda = grl_lambda

        clients[site_id] = FedAdaptClient.from_metadata(
            sd.meta,
            hidden_dim=hidden_dim,
            n_groups=N_GROUPS,
            grl_lambda_max=eff_lambda,
        )
        print(f"  [model] {site_id}: input_dim={sd.meta['input_dim']}  "
              f"→ embedding_dim={embedding_dim}  hidden={hidden_dim}  "
              f"lambda_adv={eff_lambda:.3f}")
    return clients


# ─── SCAFFOLD CONTROL VARIATES ────────────────────────────────────────────────

def init_scaffold_variates(
    clients: Dict[str, FedAdaptClient],
) -> Tuple[Dict[str, dict], dict]:
    """
    Initialise per-client (c_i) and server (c) control variates for SCAFFOLD.
    Both are zero-initialised state dicts matching SharedBody parameters.
    """
    zero_state = {
        k: torch.zeros_like(v)
        for k, v in next(iter(clients.values())).body.state_dict().items()
    }
    client_variates = {sid: deepcopy(zero_state) for sid in clients}
    server_variate  = deepcopy(zero_state)
    return client_variates, server_variate


# ─── GRADIENT-CONFLICT DIAGNOSTIC (site_D root-cause investigation) ───────────

def _flatten_state_dict(state_dict: dict) -> torch.Tensor:
    """Flatten a SharedBody state_dict into one 1D vector, in a stable key
    order (Python dicts preserve insertion order, and state_dict()'s key
    order is deterministic for a given model architecture), so vectors from
    different calls are directly comparable position-for-position."""
    return torch.cat([v.detach().float().flatten() for v in state_dict.values()])


def _cosine_similarity_state_dicts(delta_a: torch.Tensor, delta_b: torch.Tensor) -> float:
    return F.cosine_similarity(delta_a.unsqueeze(0), delta_b.unsqueeze(0)).item()


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_site(
    client:   FedAdaptClient,
    sd:       SiteData,
) -> Dict[str, float]:
    """
    Compute AUROC, F1, and AUPRC on the site's held-out test set.
    Returns zeros if no positive labels in test set.
    """
    if sd.y_test.sum() == 0:
        return {"auroc": 0.0, "f1": 0.0, "auprc": 0.0}

    client.eval()
    with torch.no_grad():
        logits, _ = client(sd.X_test)
        probs     = torch.sigmoid(logits).numpy()
        labels    = sd.y_test.numpy()

    try:
        from sklearn.metrics import (
            roc_auc_score, f1_score,
            average_precision_score, precision_recall_curve,
        )
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
        preds = (probs >= 0.5).astype(int)
        f1    = float(f1_score(labels, preds, zero_division=0))
    except Exception:
        auroc, auprc, f1 = 0.0, 0.0, 0.0

    return {"auroc": auroc, "f1": f1, "auprc": auprc}


def _local_val_auroc(client: FedAdaptClient, sd: SiteData) -> float:
    """
    Compute AUROC on the site's local validation split (v2.1 early-stopping).
    Returns -inf if no positive labels in the validation set (so any subsequent
    epoch with positives will trivially improve and update best_state).
    """
    if not hasattr(sd, "X_val") or sd.y_val.sum() == 0:
        return -float("inf")
    client.eval()
    with torch.no_grad():
        logits, _ = client(sd.X_val)
        probs     = torch.sigmoid(logits).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(sd.y_val.numpy(), probs))
    except Exception:
        return -float("inf")


# ─── v2.2: MULTI-CLUSTER PROTOTYPES ──────────────────────────────────────────
#
# Replaces single per-class centroid with K clusters per class. Lets sites with
# systematically different real subpopulations (e.g. site_A's sicker-than-typical
# AKI cohort vs site_E's milder-than-typical AKI cohort) align to *different*
# shared clusters rather than fighting toward a single averaged centroid.
#
# When --n_clusters 1 these helpers reduce to the legacy single-centroid
# behavior (one k-means cluster on each class equals the class mean), so the
# v2.2 path can fully replace the legacy single-centroid path with no
# behavioral change at K=1. All v2.2 logic lives in this script — no edits
# to fedadapt_model_approach2.py.

def compute_local_prototypes_multicluster(
    embeddings: Tensor,
    labels:     Tensor,
    n_clusters: int,
    rng:        np.random.Generator,
) -> Tuple[dict, int, int]:
    """
    Compute up to K cluster centroids per class via lightweight k-means
    on the local embeddings. Falls back to the class mean (K=1) when the
    class has fewer samples than K, or when n_clusters == 1.

    Returns:
        protos: {"aki": Tensor[k_pos, D], "non_aki": Tensor[k_neg, D]}
                — k_pos/k_neg may be < n_clusters if too few samples
        n_pos, n_neg
    """
    aki_mask     = labels.bool()
    non_aki_mask = ~aki_mask
    n_pos = int(aki_mask.sum().item())
    n_neg = int(non_aki_mask.sum().item())
    D     = embeddings.shape[1]

    def _kmeans(X: Tensor, k: int) -> Tensor:
        """Tiny CPU k-means; 10 iters, k-means++-style init."""
        if X.shape[0] == 0:
            return torch.zeros(1, D, device=X.device)
        if k <= 1 or X.shape[0] <= k:
            return X.mean(dim=0, keepdim=True)
        X_np = X.detach().cpu().numpy()
        # k-means++ init
        idx0 = int(rng.integers(X_np.shape[0]))
        centers = [X_np[idx0]]
        for _ in range(k - 1):
            d2 = np.min(
                np.array([np.sum((X_np - c) ** 2, axis=1) for c in centers]),
                axis=0,
            )
            s = d2.sum()
            if s <= 0:
                centers.append(X_np[int(rng.integers(X_np.shape[0]))])
                continue
            probs = d2 / s
            centers.append(X_np[int(rng.choice(X_np.shape[0], p=probs))])
        centers = np.stack(centers)
        for _ in range(10):
            dists = ((X_np[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assign = dists.argmin(axis=1)
            new_centers = np.stack([
                X_np[assign == j].mean(axis=0) if (assign == j).any() else centers[j]
                for j in range(centers.shape[0])
            ])
            if np.allclose(centers, new_centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        return torch.tensor(centers, dtype=X.dtype, device=X.device)

    aki_protos     = _kmeans(embeddings[aki_mask],     n_clusters)
    non_aki_protos = _kmeans(embeddings[non_aki_mask], n_clusters)

    return ({"aki": aki_protos.detach(), "non_aki": non_aki_protos.detach()},
            n_pos, n_neg)


def aggregate_prototypes_multicluster(
    client_protos: List[dict],
    client_n_pos:  List[int],
    client_n_neg:  List[int],
    n_clusters:    int,
    rng:           np.random.Generator,
) -> dict:
    """
    Pool every site's per-class cluster centers, weight each by sqrt(n_site)
    in its class (matching v1's sqrt-weighting), then k-means the pooled set
    down to n_clusters per class.

    Result: K shared clusters per class that reflect the diversity of real
    site subpopulations, not a single averaged centroid.
    """
    import math as _math

    def _pool_and_cluster(class_key: str, n_per_site: List[int]) -> Tensor:
        all_centers = []
        all_weights = []
        for proto_dict, n in zip(client_protos, n_per_site):
            centers = proto_dict[class_key]
            if centers.dim() == 1:
                centers = centers.unsqueeze(0)
            w = _math.sqrt(max(n, 1)) / max(centers.shape[0], 1)
            for c in centers:
                all_centers.append(c.detach().cpu().numpy())
                all_weights.append(w)
        if not all_centers:
            return torch.zeros(1, client_protos[0][class_key].shape[-1])
        X = np.stack(all_centers)
        w = np.asarray(all_weights)
        # weighted k-means on the pooled cluster centers
        k = min(n_clusters, X.shape[0])
        if k <= 1:
            return torch.tensor((X * w[:, None]).sum(axis=0) / w.sum()).unsqueeze(0).float()
        # k-means++ init weighted by w
        p0 = w / w.sum()
        idx0 = int(rng.choice(X.shape[0], p=p0))
        centers = [X[idx0]]
        for _ in range(k - 1):
            d2 = np.min(
                np.array([np.sum((X - c) ** 2, axis=1) for c in centers]),
                axis=0,
            )
            wd = d2 * w
            s = wd.sum()
            if s <= 0:
                centers.append(X[int(rng.integers(X.shape[0]))])
                continue
            centers.append(X[int(rng.choice(X.shape[0], p=wd / s))])
        centers = np.stack(centers)
        for _ in range(10):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            assign = dists.argmin(axis=1)
            new_centers = []
            for j in range(centers.shape[0]):
                mask = (assign == j)
                if mask.any():
                    cw = w[mask]
                    new_centers.append((X[mask] * cw[:, None]).sum(axis=0) / cw.sum())
                else:
                    new_centers.append(centers[j])
            new_centers = np.stack(new_centers)
            if np.allclose(centers, new_centers, atol=1e-6):
                centers = new_centers
                break
            centers = new_centers
        return torch.tensor(centers).float()

    return {
        "aki":     _pool_and_cluster("aki",     client_n_pos).detach(),
        "non_aki": _pool_and_cluster("non_aki", client_n_neg).detach(),
    }


def prototype_alignment_loss_multicluster(
    embeddings:    Tensor,
    labels:        Tensor,
    global_protos: dict,
) -> Tensor:
    """
    Pulls each local embedding toward its NEAREST global cluster of the same
    class. At K=1 reduces exactly to v1's single-centroid alignment.

    No within-class collapse: AKI samples are only ever pulled toward AKI
    clusters; non-AKI samples only toward non-AKI clusters.
    """
    aki_mask     = labels.bool()
    non_aki_mask = ~aki_mask

    aki_centers     = global_protos["aki"]      # [K_aki, D]
    non_aki_centers = global_protos["non_aki"]  # [K_neg, D]

    if aki_centers.dim() == 1:
        aki_centers = aki_centers.unsqueeze(0)
    if non_aki_centers.dim() == 1:
        non_aki_centers = non_aki_centers.unsqueeze(0)

    loss = torch.tensor(0.0, device=embeddings.device)
    n_terms = 0

    if aki_mask.any():
        emb_aki = embeddings[aki_mask]  # [n, D]
        d = torch.cdist(emb_aki, aki_centers.to(emb_aki.device))   # [n, K_aki]
        nearest = d.argmin(dim=1)                                  # [n]
        targets = aki_centers.to(emb_aki.device)[nearest]          # [n, D]
        loss = loss + F.mse_loss(emb_aki, targets)
        n_terms += 1

    if non_aki_mask.any():
        emb_neg = embeddings[non_aki_mask]
        d = torch.cdist(emb_neg, non_aki_centers.to(emb_neg.device))
        nearest = d.argmin(dim=1)
        targets = non_aki_centers.to(emb_neg.device)[nearest]
        loss = loss + F.mse_loss(emb_neg, targets)
        n_terms += 1

    return loss / max(n_terms, 1)


# ─── LOCAL TRAINING STEPS ─────────────────────────────────────────────────────

def local_step_fedadapt(
    client:     FedAdaptClient,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    lambda_adv: float,
    pos_weight: Optional[Tensor],
) -> Dict[str, float]:
    """One epoch of FedAdapt local training (task + GRL adversarial)."""
    client.train()
    totals = {"total": 0.0, "task": 0.0, "adv": 0.0}
    n = 0

    for x, y, g in loader:
        optimizer.zero_grad()
        task_logit, group_logits = client(x)
        total, task, adv, _ = fedadapt_loss(
            task_logit, y, group_logits, g,
            lambda_adv=lambda_adv, pos_weight=pos_weight,
        )
        total.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()

        b = len(y)
        totals["total"] += total.item() * b
        totals["task"]  += task.item()  * b
        totals["adv"]   += adv.item()   * b
        n += b

    return {k: v / max(n, 1) for k, v in totals.items()}


def local_step_fedadaptproto(
    client:        FedAdaptClient,
    loader:        DataLoader,
    optimizer:     torch.optim.Optimizer,
    lambda_adv:    float,
    pos_weight:    Optional[Tensor],
    global_protos: Optional[dict],
    alpha_proto:   float,
    multicluster:  bool = False,
) -> Dict[str, float]:
    """
    One epoch of FedAdapt-Proto local training: task + GRL adversarial
    + prototype alignment. Identical to local_step_fedadapt when
    global_protos is None or alpha_proto == 0 (falls back to plain
    FedAdapt loss for that batch/round, e.g. round 1 before any
    prototypes exist yet).

    v2.2: when multicluster=True, prototype alignment uses
    prototype_alignment_loss_multicluster (nearest-cluster MSE over K
    centers per class) instead of the legacy single-centroid loss baked
    into fedadapt_loss. At K=1 the multicluster loss reduces exactly to
    the legacy loss, so the two paths are equivalent for K=1; multicluster
    only diverges when n_clusters > 1.

    NOTE: a reliability-weighted variant (v2) was tried — down-weighting
    the prototype pull for sites with abundant local samples (theory:
    protect their already-strong signal from "dilution" by noisier
    sites). Tested at two tau values (50 and 3000); both REGRESSED every
    site, including site_D, which had been the standout success of this
    uniform-weight loss (20/20 wins vs SCAFFOLD, mean +0.031 margin).
    Reverted — uniform alpha_proto weighting (no per-site adjustment) is
    the strongest variant found.
    """
    client.train()
    totals = {"total": 0.0, "task": 0.0, "adv": 0.0, "proto": 0.0}
    n = 0

    use_proto = global_protos is not None and alpha_proto > 0.0

    for x, y, g in loader:
        optimizer.zero_grad()
        task_logit, group_logits, emb = client.forward_with_embedding(x)

        if use_proto and multicluster:
            # Compute task + adv via fedadapt_loss with no prototypes,
            # then add the multi-cluster alignment term separately.
            total_no_proto, task, adv, _ = fedadapt_loss(
                task_logit, y, group_logits, g,
                lambda_adv=lambda_adv, pos_weight=pos_weight,
                embeddings=None, global_protos=None, alpha_proto=0.0,
            )
            proto = prototype_alignment_loss_multicluster(
                emb, y, global_protos,
            )
            total = total_no_proto + alpha_proto * proto
        else:
            total, task, adv, proto = fedadapt_loss(
                task_logit, y, group_logits, g,
                lambda_adv=lambda_adv, pos_weight=pos_weight,
                embeddings=emb if use_proto else None,
                global_protos=global_protos if use_proto else None,
                alpha_proto=alpha_proto if use_proto else 0.0,
            )

        total.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()

        b = len(y)
        totals["total"] += total.item() * b
        totals["task"]  += task.item()  * b
        totals["adv"]   += adv.item()   * b
        totals["proto"] += proto.item() * b
        n += b

    return {k: v / max(n, 1) for k, v in totals.items()}


def local_step_fedavg(
    client:     FedAdaptClient,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    pos_weight: Optional[Tensor],
) -> Dict[str, float]:
    """One epoch of FedAvg local training (task loss only, no GRL)."""
    client.train()
    total_loss, n = 0.0, 0

    for x, y, _ in loader:
        optimizer.zero_grad()
        # For baselines: use encode() + head directly (bypasses GRL)
        emb    = client.encode(x)
        logit  = client.head(emb)
        loss   = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def local_step_fedprox(
    client:            FedAdaptClient,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    pos_weight:        Optional[Tensor],
    global_body_state: dict,
    mu:                float,
) -> Dict[str, float]:
    """One epoch of FedProx (FedAvg + proximal penalty on SharedBody)."""
    client.train()
    total_loss, n = 0.0, 0

    for x, y, _ in loader:
        optimizer.zero_grad()
        emb   = client.encode(x)
        logit = client.head(emb)
        task  = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        prox  = fedprox_penalty(client, global_body_state, mu=mu)
        loss  = task + prox
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def local_step_scaffold(
    client:          FedAdaptClient,
    loader:          DataLoader,
    optimizer:       torch.optim.Optimizer,
    pos_weight:      Optional[Tensor],
    c_i:             dict,
    c_server:        dict,
) -> Dict[str, float]:
    """
    One epoch of SCAFFOLD local training.
    After the standard gradient step, applies the control variate correction:
        w ← w - lr * (c_i - c_server)
    This corrects the client drift caused by heterogeneous local data.
    Note: sign is c_i - c_server (local minus global), not the reverse.
    """
    client.train()
    total_loss, n = 0.0, 0
    lr = optimizer.param_groups[0]["lr"]

    for x, y, _ in loader:
        optimizer.zero_grad()
        emb   = client.encode(x)
        logit = client.head(emb)
        loss  = F.binary_cross_entropy_with_logits(
            logit, y.float(), pos_weight=pos_weight
        )
        loss.backward()
        nn.utils.clip_grad_norm_(client.parameters(), max_norm=1.0)
        optimizer.step()

        # Apply control variate correction to SharedBody params only
        # SCAFFOLD correction: subtract (c_i - c_server) from gradient
        # i.e. w ← w - lr*(grad + c_i - c_server)
        # Equivalent to: param -= lr * (c_i - c_server) AFTER the grad step
        with torch.no_grad():
            for name, param in client.body.named_parameters():
                if name in c_server and name in c_i:
                    correction = c_i[name] - c_server[name]   # FIX: was c_server - c_i (sign flipped)
                    correction = correction.clamp(-1.0, 1.0)  # FIX: clip to prevent accumulation
                    param.data -= lr * correction

        total_loss += loss.item() * len(y)
        n += len(y)

    return {"total": total_loss / max(n, 1), "task": total_loss / max(n, 1), "adv": 0.0}


def update_scaffold_variates(
    client:         FedAdaptClient,
    c_i:            dict,
    c_server:       dict,
    global_body:    dict,
    local_body_pre: dict,
    local_epochs:   int,
    lr:             float,
) -> dict:
    """
    Compute updated client control variate c_i_new after local training.
    SCAFFOLD Option II:
        c_i_new = c_i - c_server + (w_global - w_local) / (K * lr)
    Returns delta = c_i_new - c_i for server aggregation.
    """
    c_i_new  = {}
    delta_c  = {}
    for key in c_i:
        w_g = global_body[key].float()
        w_l = client.body.state_dict()[key].float()
        c_i_new[key] = (c_i[key].float()
                        - c_server[key].float()
                        + (w_g - w_l) / (local_epochs * lr + 1e-8))
        delta_c[key] = c_i_new[key] - c_i[key]
    return c_i_new, delta_c


# ─── HEAD FINE-TUNING (FedAdapt only) ────────────────────────────────────────
# Ablation-ready: v2.3's original recipe was (lr, 10 epochs, pos_weight-on,
# no clip). v2.5 independently uses a different recipe (lr*0.1, 30 epochs,
# pos_weight-off, clip=5.0) with no documented rationale for the switch and
# no ablation isolating which change (if any) drives the difference. These
# four flags let the recipe be swept on top of v2.3's federation phase
# (held constant), instead of comparing two scripts that differ in the
# federation phase too.

def finetune_heads(
    clients:         Dict[str, FedAdaptClient],
    sites:           Dict[str, SiteData],
    finetune_epochs: int,
    lr:              float,
    use_pos_weight:  bool = True,
    grad_clip:       float = 0.0,
) -> None:
    """
    Freeze SharedBody. Fine-tune PersonalHead locally per site.
    FedBABU-style: body is fixed, only head weights update.

    use_pos_weight: if False, uses plain BCE (no class-imbalance weighting)
        — matches v2.5's final fine-tune recipe when False; matches v2.3's
        original behavior when True (default).
    grad_clip: if > 0, clips head-parameter gradient norm to this value
        after backward(), before the optimizer step. 0 (default) disables
        clipping — matches v2.3's original behavior. v2.5 uses 5.0.
    """
    print("\n  ── Head fine-tuning (FedAdapt) ──"
          f"  [lr={lr}  epochs={finetune_epochs}  "
          f"pos_weight={'on' if use_pos_weight else 'off'}  "
          f"grad_clip={grad_clip if grad_clip > 0 else 'off'}]")
    for site_id, client in clients.items():
        sd = sites[site_id]
        client.freeze_body()
        opt = torch.optim.Adam(client.head_parameters(), lr=lr)

        for epoch in range(finetune_epochs):
            client.train()
            for x, y, _ in sd.train_loader:
                opt.zero_grad()
                emb   = client.encode(x)
                logit = client.head(emb)
                pw    = sd.pos_weight if use_pos_weight else None
                loss  = F.binary_cross_entropy_with_logits(
                    logit, y.float(), pos_weight=pw
                )
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(client.head_parameters(), grad_clip)
                opt.step()

        client.unfreeze_body()
        metrics = evaluate_site(client, sd)

        print(f"    {site_id}: AUROC={metrics['auroc']:.4f}  "
              f"F1={metrics['f1']:.4f}  AUPRC={metrics['auprc']:.4f}")


# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_training_curves(
    round_records: List[dict],
    method:        str,
    output_dir:    Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df     = pd.DataFrame(round_records)
    sites  = [c for c in df.columns if c.startswith("auroc_")]
    labels = [c.replace("auroc_", "") for c in sites]

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Training curves — {method.upper()}", fontsize=12)

    ax = axes[0]
    for i, (col, lbl) in enumerate(zip(sites, labels)):
        ax.plot(df["round"], df[col], label=lbl, color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.set_xlabel("Federation round")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-site AUROC across rounds")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    loss_cols = [c for c in df.columns if c.startswith("task_loss_")]
    for i, col in enumerate(loss_cols):
        lbl = col.replace("task_loss_", "")
        ax2.plot(df["round"], df[col], label=lbl, color=COLORS[i % len(COLORS)],
                 linewidth=1.8)
    ax2.set_xlabel("Federation round")
    ax2.set_ylabel("Task loss")
    ax2.set_title("Per-site task loss across rounds")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = output_dir / "training_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


def plot_fl_gain_correlation(
    final_metrics:   pd.DataFrame,
    local_metrics:   Dict[str, float],
    fl_gain_csv:     Optional[Path],
    method:          str,
    output_dir:      Path,
) -> None:
    """
    Phase III validation figure:
    FL-gain index (x) vs AUROC improvement over local-only (y).

    Each point is one site. If the FL-gain index is a good predictor,
    we expect a positive correlation.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        print("  [skip] fl_gain_correlation: matplotlib/scipy not available")
        return

    if fl_gain_csv is None or not fl_gain_csv.exists():
        print("  [skip] fl_gain_correlation: fl_gain_csv not found")
        return

    fl_df = pd.read_csv(fl_gain_csv)

    rows = []
    for _, row in fl_df.iterrows():
        sid = row["site_id"]
        if sid not in final_metrics["site_id"].values:
            continue
        fed_auroc   = float(final_metrics.loc[final_metrics["site_id"] == sid, "auroc"].iloc[0])
        local_auroc = local_metrics.get(sid, 0.0)
        improvement = fed_auroc - local_auroc
        rows.append({
            "site_id":      sid,
            "fl_gain":      float(row["fl_gain_index"]),
            "local_auroc":  local_auroc,
            "fed_auroc":    fed_auroc,
            "improvement":  improvement,
            "role":         row.get("role", "unknown"),
        })

    if len(rows) < 3:
        print("  [skip] fl_gain_correlation: not enough labelled sites")
        return

    corr_df = pd.DataFrame(rows)

    ROLE_COLORS = {
        "contributor":            "#4C72B0",
        "conditional_benefitter": "#DD8452",
        "primary_benefitter":     "#C44E52",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"FL-gain index vs AUROC improvement — {method.upper()}\n"
        f"Phase III validation: does FL-gain index predict benefit from federation?",
        fontsize=11,
    )

    # ── Panel 1: FL-gain vs AUROC improvement ────────────────────────────────
    ax = axes[0]
    for _, row in corr_df.iterrows():
        color = ROLE_COLORS.get(row["role"], "#333333")
        ax.scatter(row["fl_gain"], row["improvement"],
                   color=color, s=120, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(row["site_id"], (row["fl_gain"], row["improvement"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Regression line
    x_vals = corr_df["fl_gain"].values
    y_vals = corr_df["improvement"].values
    if len(x_vals) >= 3:
        m, b   = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.6)
        try:
            r, p_val = pearsonr(x_vals, y_vals)
            rho, _   = spearmanr(x_vals, y_vals)
            ax.set_title(
                f"Pearson r={r:.3f}  Spearman ρ={rho:.3f}  (p={p_val:.3f})"
            )
        except Exception:
            ax.set_title("FL-gain vs AUROC improvement")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("FL-gain index")
    ax.set_ylabel("AUROC improvement (federated − local-only)")
    ax.grid(alpha=0.3)

    # ── Panel 2: side-by-side AUROC bar chart ────────────────────────────────
    ax2    = axes[1]
    n      = len(corr_df)
    x_pos  = np.arange(n)
    bw     = 0.35
    colors = [ROLE_COLORS.get(r, "#333") for r in corr_df["role"]]

    ax2.bar(x_pos - bw / 2, corr_df["local_auroc"], bw,
            label="Local-only", color="#AAAAAA", edgecolor="white")
    ax2.bar(x_pos + bw / 2, corr_df["fed_auroc"], bw,
            label=f"{method.upper()} (federated)", color=colors, edgecolor="white")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(corr_df["site_id"], rotation=15, ha="right")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Local-only vs federated AUROC per site")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Save correlation table
    corr_df.to_csv(output_dir / "fl_gain_correlation.csv", index=False)

    plt.tight_layout()
    p = output_dir / "fl_gain_correlation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


def plot_final_comparison(
    final_metrics: pd.DataFrame,
    method:        str,
    output_dir:    Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df     = final_metrics.copy()
    sites  = df["site_id"].tolist()
    x      = np.arange(len(sites))
    bw     = 0.25

    COLORS = {"auroc": "#4C72B0", "f1": "#DD8452", "auprc": "#55A868"}
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (metric, color) in enumerate(COLORS.items()):
        vals = df[metric].tolist()
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=metric.upper(),
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Final evaluation metrics — {method.upper()}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = output_dir / "final_metrics.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {p.name}")


# ─── LOCAL-ONLY BASELINE ─────────────────────────────────────────────────────

def run_local_only(
    sites:         Dict[str, SiteData],
    embedding_dim: int,
    hidden_dim:    int,
    local_epochs:  int,
    lr:            float,
) -> Dict[str, float]:
    """
    Train each site independently (no federation) for the same number of
    epochs as one federation round × T rounds. Used as the lower bound
    for the FL-gain correlation plot.
    """
    print("\n  ── Local-only baseline ──")
    local_aurocs: Dict[str, float] = {}

    for site_id, sd in sites.items():
        client = FedAdaptClient.from_metadata(
            sd.meta, hidden_dim=hidden_dim, n_groups=N_GROUPS,
        )
        opt = torch.optim.Adam(client.parameters(), lr=lr)

        for _ in range(local_epochs):
            for x, y, g in sd.train_loader:
                opt.zero_grad()
                emb   = client.encode(x)
                logit = client.head(emb)
                loss  = F.binary_cross_entropy_with_logits(
                    logit, y.float(), pos_weight=sd.pos_weight
                )
                loss.backward()
                opt.step()

        m = evaluate_site(client, sd)
        local_aurocs[site_id] = m["auroc"]
        print(f"    {site_id}: local AUROC={m['auroc']:.4f}")

    return local_aurocs


# ─── MAIN FEDERATION LOOP ─────────────────────────────────────────────────────

def run_federation(
    method:          str,
    sites:           Dict[str, SiteData],
    clients:         Dict[str, FedAdaptClient],
    args:            argparse.Namespace,
    output_dir:      Path,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Run T federation rounds for the chosen method.
    Returns (final_metrics_df, round_records).
    """
    server = FedAdaptServer(
        body=next(iter(clients.values())).body
    )
    print(f"  [DEBUG-GC] args.track_gradient_conflict = {args.track_gradient_conflict!r}  "
          f"(type={type(args.track_gradient_conflict).__name__})")
    server.broadcast(list(clients.values()))

    # v2.3: per-site K resolution. If --n_clusters_per_site is provided,
    # parse it; otherwise default every site to the global --n_clusters.
    # This lets you give site_E (uni-modal AKI structure per silhouette
    # diagnostic) K=2 while keeping site_A/B/C at K=3.
    per_site_k: Dict[str, int] = {}
    if getattr(args, "n_clusters_per_site", None):
        for pair in args.n_clusters_per_site.split(","):
            pair = pair.strip()
            if not pair: continue
            if "=" not in pair:
                raise ValueError(f"--n_clusters_per_site entries must be 'site_id=K': got {pair!r}")
            sid, kval = pair.split("=", 1)
            per_site_k[sid.strip()] = int(kval.strip())
        # any site not explicitly listed falls back to the global default
        for sid in clients:
            per_site_k.setdefault(sid, args.n_clusters)
        print(f"  [per-site K] {per_site_k}")
    else:
        for sid in clients:
            per_site_k[sid] = args.n_clusters

    # Per-client optimisers — all params for fedadapt, body+head for baselines
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for sid, client in clients.items():
        optimizers[sid] = torch.optim.Adam(client.parameters(), lr=args.lr)

    # SCAFFOLD variates
    if method == "scaffold":
        client_variates, server_variate = init_scaffold_variates(clients)

    # Global body snapshot for FedProx
    global_body_snapshot: dict = {}

    # Global prototypes for FedAdapt-Proto (None until round 1 completes)
    global_protos: Optional[dict] = None

    round_records: List[dict] = []

    # Phase-1 checkpointing: track each site's own best across-round AUROC
    # and a snapshot of its full client state at that point. Restored below
    # before head fine-tuning (fedadapt/fedadaptproto) or before final
    # evaluation (fedavg/fedprox/scaffold, which have no fine-tuning stage
    # at all — for those methods this checkpoint IS the only correction
    # available). No-op for sites still improving at the final round.
    ckpt_best_auroc: Dict[str, float] = {sid: -1.0 for sid in clients}
    ckpt_best_round: Dict[str, int]   = {sid: 0 for sid in clients}
    ckpt_best_state: Dict[str, dict]  = {sid: None for sid in clients}

    gradient_conflict_records: List[dict] = []

    print(f"\n  ── Federation [{method.upper()}]  {args.rounds} rounds × {args.local_epochs} local epochs ──")

    for rnd in range(1, args.rounds + 1):
        progress = rnd / args.rounds

        # ── warm-up scheduling for lambda_adv and alpha_proto ───────────────
        # Both auxiliary losses (GRL adversarial, prototype alignment) are
        # ramped from 0 -> target over the first --warmup_rounds rounds.
        # Lets the classifier converge on real task signal before the
        # auxiliary objectives start pulling the shared body around.
        # warmup_rounds=0 disables warm-up (legacy behavior).
        if args.warmup_rounds > 0 and rnd <= args.warmup_rounds:
            warmup_scale = rnd / float(args.warmup_rounds)
        else:
            warmup_scale = 1.0
        eff_lambda_adv  = args.lambda_adv  * warmup_scale
        eff_alpha_proto = args.alpha_proto * warmup_scale

        # ── broadcast ──────────────────────────────────────────────────────
        server.broadcast(list(clients.values()))
        if method == "fedprox":
            global_body_snapshot = deepcopy(
                next(iter(clients.values())).body.state_dict()
            )

        if args.track_gradient_conflict:
            pre_round_body_state = {
                k: v.detach().clone()
                for k, v in server.global_body.state_dict().items()
            }
            site_body_states_this_round: Dict[str, dict] = {}

        body_states:   List[dict]  = []
        sample_counts: List[float] = []
        delta_variates: Dict[str, dict] = {}

        # ── prototype pre-pass (FedAdapt-Proto only) ─────────────────────────
        # Computed BEFORE local training using current (pre-update) body, so
        # every client aligns toward prototypes that reflect the *start* of
        # this round's shared representation, not a stale previous-round one
        # and not a moving target updated mid-round.
        if method == "fedadaptproto":
            client_protos:  List[dict] = []
            client_n_pos:   List[int]  = []
            client_n_neg:   List[int]  = []
            for sid, client in clients.items():
                sd = sites[sid]
                client.eval()
                with torch.no_grad():
                    xs, ys = [], []
                    for x, y, _ in sd.train_loader:
                        xs.append(x)
                        ys.append(y)
                    x_all = torch.cat(xs, dim=0)
                    y_all = torch.cat(ys, dim=0)
                    emb_all = client.encode(x_all)
                # v2.3: per-site K. If --n_clusters_per_site override is given,
                # use it; otherwise fall back to the global --n_clusters value
                # so v2.2 behavior is preserved when no override is passed.
                k_site = per_site_k.get(sid, args.n_clusters)
                # Site-specific RNG seed: previously this was
                # args.seed + rnd only, identical across ALL sites within a
                # round, meaning every site's k-means++ initialization drew
                # the same relative random sequence — not truly independent
                # per-site randomness, even though it was still fully
                # deterministic run-to-run. site_idx (stable positional index
                # in the clients dict, since Python dicts preserve insertion
                # order) gives each site its own distinct, reproducible seed.
                site_idx = list(clients.keys()).index(sid)
                protos, n_pos, n_neg = (
                    compute_local_prototypes_multicluster(
                        emb_all, y_all,
                        n_clusters=k_site,
                        rng=np.random.default_rng(args.seed + rnd * 1000 + site_idx),
                    )
                    if k_site > 1
                    else compute_local_prototypes(emb_all, y_all)
                )
                client_protos.append(protos)
                client_n_pos.append(n_pos)
                client_n_neg.append(n_neg)

            # client_n_pos / client_n_neg feed server-side aggregation weights
            # (sqrt(n)-weighted, see FedAdaptServer.aggregate_prototypes) —
            # NOT used for per-client loss reweighting; that reliability-
            # weighted variant (v2) was tried and reverted, see
            # local_step_fedadaptproto docstring.
            # Global aggregation K = max of per-site Ks (so a site with K=2
            # contributes 2 centers into a 3-cluster global pool when other
            # sites use K=3). This is the cleanest pooling behavior: every
            # site's centers are still represented, but a uni-modal site
            # doesn't fabricate noise clusters into the shared space.
            k_global = max(per_site_k.values()) if per_site_k else args.n_clusters
            if k_global > 1:
                global_protos = aggregate_prototypes_multicluster(
                    client_protos, client_n_pos, client_n_neg,
                    n_clusters=k_global,
                    rng=np.random.default_rng(args.seed + rnd + 9999),
                )
            else:
                global_protos = server.aggregate_prototypes(
                    client_protos, client_n_pos, client_n_neg
                )

        # ── local training ─────────────────────────────────────────────────
        round_losses: Dict[str, dict] = {}
        for sid, client in clients.items():
            sd = sites[sid]
            client.set_training_progress(progress)
            opt = optimizers[sid]

            # v2.1: early-stopping per round for fedadapt/fedadaptproto.
            # Run a fixed schedule for other methods (unchanged from v1).
            use_early_stop = (
                args.early_stop_patience > 0
                and method in ("fedadapt", "fedadaptproto")
            )

            best_val_auc      = -float("inf")
            best_state        = None
            epochs_no_improve = 0

            for local_ep in range(args.local_epochs):
                if method == "fedadapt":
                    losses = local_step_fedadapt(
                        client, sd.train_loader, opt,
                        lambda_adv=eff_lambda_adv,
                        pos_weight=sd.pos_weight,
                    )
                elif method == "fedadaptproto":
                    losses = local_step_fedadaptproto(
                        client, sd.train_loader, opt,
                        lambda_adv=eff_lambda_adv,
                        pos_weight=sd.pos_weight,
                        global_protos=global_protos,
                        alpha_proto=eff_alpha_proto,
                        multicluster=(max(per_site_k.values()) > 1 if per_site_k else args.n_clusters > 1),
                    )
                elif method == "fedprox":
                    losses = local_step_fedprox(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                        global_body_state=global_body_snapshot,
                        mu=args.mu,
                    )
                elif method == "scaffold":
                    losses = local_step_scaffold(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                        c_i=client_variates[sid],
                        c_server=server_variate,
                    )
                else:  # fedavg
                    losses = local_step_fedavg(
                        client, sd.train_loader, opt,
                        pos_weight=sd.pos_weight,
                    )

                if use_early_stop:
                    val_auc = _local_val_auroc(client, sd)
                    if val_auc > best_val_auc + 1e-4:
                        best_val_auc = val_auc
                        best_state = {k: v.detach().cpu().clone()
                                       for k, v in client.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= args.early_stop_patience:
                            break

            # Restore best local checkpoint before contributing to global agg.
            if use_early_stop and best_state is not None:
                client.load_state_dict(best_state)

            round_losses[sid] = losses

            # SCAFFOLD: update client variate after local steps
            if method == "scaffold":
                c_i_new, delta_c = update_scaffold_variates(
                    client,
                    c_i=client_variates[sid],
                    c_server=server_variate,
                    global_body=server.global_body.state_dict(),
                    local_body_pre=global_body_snapshot,
                    local_epochs=args.local_epochs,
                    lr=args.lr,
                )
                client_variates[sid] = c_i_new
                delta_variates[sid]  = delta_c

            body_states.append(client.get_body_state())
            sample_counts.append(float(sd.n_samples))

            if args.track_gradient_conflict:
                site_body_states_this_round[sid] = {
                    k: v.detach().clone() for k, v in client.get_body_state().items()
                }

        # ── aggregate ──────────────────────────────────────────────────────
        server.aggregate(body_states, sample_counts)

        if args.track_gradient_conflict:
            post_agg_body_state = {
                k: v.detach().clone()
                for k, v in server.global_body.state_dict().items()
            }
            pre_vec = _flatten_state_dict(pre_round_body_state)
            agg_delta = _flatten_state_dict(post_agg_body_state) - pre_vec
            for sid_gc, site_state in site_body_states_this_round.items():
                site_delta = _flatten_state_dict(site_state) - pre_vec
                cos_sim = _cosine_similarity_state_dicts(site_delta, agg_delta)
                gradient_conflict_records.append({
                    "round": rnd,
                    "site_id": sid_gc,
                    "cosine_similarity_with_aggregate": round(cos_sim, 4),
                    "site_delta_norm": round(site_delta.norm().item(), 6),
                    "aggregate_delta_norm": round(agg_delta.norm().item(), 6),
                })
            if rnd == 1:
                print(f"  [DEBUG-GC] after round 1: "
                      f"site_body_states_this_round had {len(site_body_states_this_round)} sites, "
                      f"gradient_conflict_records now has {len(gradient_conflict_records)} rows")

        # SCAFFOLD: update server variate
        if method == "scaffold" and delta_variates:
            n_participants = len(delta_variates)
            for key in server_variate:
                server_variate[key] = server_variate[key] + sum(
                    dv[key] for dv in delta_variates.values()
                ) / n_participants

        # ── evaluate ───────────────────────────────────────────────────────
        server.broadcast(list(clients.values()))
        record: dict = {"round": rnd}
        for sid, client in clients.items():
            sd = sites[sid]
            m  = evaluate_site(client, sd)
            record[f"auroc_{sid}"]     = round(m["auroc"], 4)
            record[f"task_loss_{sid}"] = round(round_losses.get(sid, {}).get("task", 0.0), 4)
            record[f"adv_loss_{sid}"]  = round(round_losses.get(sid, {}).get("adv", 0.0), 4)

            if m["auroc"] > ckpt_best_auroc[sid]:
                ckpt_best_auroc[sid] = m["auroc"]
                ckpt_best_round[sid] = rnd
                ckpt_best_state[sid] = {k: v.detach().cpu().clone()
                                         for k, v in client.state_dict().items()}

        round_records.append(record)

        if rnd % max(1, args.rounds // 10) == 0 or rnd == args.rounds:
            aurocs = "  ".join(
                f"{sid}={record[f'auroc_{sid}']:.3f}"
                for sid in clients
            )
            print(f"    Round {rnd:3d}/{args.rounds}  |  {aurocs}")

    print("  ── Restoring each site to its own best-checkpoint round ──")
    for sid, client in clients.items():
        if ckpt_best_state[sid] is not None:
            client.load_state_dict(ckpt_best_state[sid])
        marker = " (== final round)" if ckpt_best_round[sid] == args.rounds else " <-- earlier than final round"
        print(f"    {sid}: best round={ckpt_best_round[sid]}  best_auroc={ckpt_best_auroc[sid]:.4f}{marker}")

    # ── head fine-tuning (FedAdapt and FedAdapt-Proto only) ────────────────
    if method in ("fedadapt", "fedadaptproto"):
        finetune_heads(clients, sites, args.finetune_epochs,
                       args.lr * args.ft_lr_mult,
                       use_pos_weight=args.ft_pos_weight,
                       grad_clip=args.ft_grad_clip)

    # ── final evaluation ───────────────────────────────────────────────────
    final_rows = []
    for sid, client in clients.items():
        m = evaluate_site(client, sites[sid])
        final_rows.append({
            "site_id": sid,
            "best_checkpoint_round": ckpt_best_round[sid],
            **m,
        })
    final_df = pd.DataFrame(final_rows)

    print(f"  [DEBUG-GC] final check: args.track_gradient_conflict="
          f"{args.track_gradient_conflict!r}, len(gradient_conflict_records)="
          f"{len(gradient_conflict_records)}")
    if args.track_gradient_conflict and gradient_conflict_records:
        pd.DataFrame(gradient_conflict_records).to_csv(
            output_dir / "gradient_conflict.csv", index=False
        )
        print(f"  [gradient-conflict] saved gradient_conflict.csv "
              f"({len(gradient_conflict_records)} rows)")

    return final_df, round_records


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--data_dir",       required=True,  help="Phase II simulation output dir")
    p.add_argument("--label",          default="AKI_label")
    p.add_argument("--alpha",          type=float, default=0.5,  help="Dirichlet α used in sim")
    p.add_argument("--gamma",          type=float, default=0.75, help="Gamma used in sim")
    p.add_argument("--fl_gain_csv",    default=None,   help="Path to fl_gain_index_*.csv")

    # Method
    p.add_argument("--method",         default="fedadapt",
                   choices=["fedadapt", "fedadaptproto", "fedavg", "fedprox", "scaffold"])

    # Federation
    p.add_argument("--rounds",         type=int,   default=50)
    p.add_argument("--local_epochs",   type=int,   default=1,    help="K local epochs per round")
    p.add_argument("--finetune_epochs",type=int,   default=30,   help="Head fine-tune epochs (FedAdapt)")

    # Head fine-tune recipe — DEFAULTS UPDATED to runD, the best of the
    # Tier 1 ablation (mean +0.0519 across 6 sites, best of runA-D, all
    # differences were small but this consistently edged out the others).
    # local_epochs default also changed to 1 (confirmed free improvement,
    # used in all of runA-D). To reproduce ORIGINAL v2.3 behavior instead
    # (pre-ablation, the actual v2.3 defaults before this update), pass
    # --local_epochs 5 --ft_lr_mult 1.0 --finetune_epochs 10
    # --ft_pos_weight --ft_grad_clip 0.0 explicitly.
    p.add_argument("--ft_lr_mult",     type=float, default=0.1,
                   help="Multiplier on --lr for the head fine-tune step "
                        "(default 0.1, runD/v2.5-style; original v2.3 was 1.0)")
    p.add_argument("--ft_pos_weight",  action="store_true", default=False,
                   help="Use class-imbalance pos_weight in the fine-tune loss "
                        "(default: off, runD/v2.5-style; original v2.3 was on)")
    p.add_argument("--ft_no_pos_weight", dest="ft_pos_weight", action="store_false",
                   help="(no-op with the new default — pos_weight is already off "
                        "by default. Kept for backward compatibility.)")
    p.add_argument("--ft_use_pos_weight", dest="ft_pos_weight", action="store_true",
                   help="Re-enable pos_weight (original v2.3 behavior)")
    p.add_argument("--ft_grad_clip",   type=float, default=5.0,
                   help="Grad-norm clip on head parameters during fine-tune "
                        "(default 5.0, runD/v2.5-style; original v2.3 was 0.0/off)")

    # Model
    p.add_argument("--embedding_dim",  type=int,   default=64)
    p.add_argument("--hidden_dim",     type=int,   default=128)

    # Optimisation
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--batch_size",     type=int,   default=256,  help="CPU-safe batch size")
    p.add_argument("--lambda_adv",      type=float, default=0.1,
                        help="GRL max weight (FedAdapt); was 0.5 which caused early collapse")
    p.add_argument("--adaptive_lambda", action="store_true", default=True,
                        help="Scale lambda_adv by site AKI prevalence (default: True). "
                             "Prevents GRL from overpowering classification at low-prevalence sites.")
    p.add_argument("--no_adaptive_lambda", dest="adaptive_lambda", action="store_false",
                        help="Disable prevalence-adaptive lambda_adv (use fixed lambda for all sites)")
    p.add_argument("--alpha_proto",    type=float, default=0.5,
                        help="Prototype alignment loss weight (FedAdapt-Proto only). "
                             "Cooperative term added to the loss — no gradient reversal. "
                             "0 disables prototype alignment (reduces to plain FedAdapt). "
                             "Uniform weighting across all sites — a reliability-weighted "
                             "variant (down-weighting abundant sites) was tried and reverted; "
                             "it regressed every site including site_D, the standout success "
                             "of this uniform-weight loss (20/20 wins vs SCAFFOLD).")
    p.add_argument("--mu",             type=float, default=0.01, help="Proximal weight (FedProx)")

    # Auxiliary-loss warm-up (FedAdapt / FedAdapt-Proto)
    p.add_argument("--warmup_rounds",  type=int,   default=10,
                        help="Linearly ramp lambda_adv and alpha_proto from 0 to "
                             "target value over the first N federation rounds. "
                             "Lets the classifier converge on real signal before "
                             "GRL/prototype objectives kick in. 0 disables warm-up "
                             "(legacy v1 behavior). Default 10 = 20%% of a 50-round run.")

    # v2.1: per-round early-stopping over local epochs (fedadapt/fedadaptproto)
    p.add_argument("--early_stop_patience", type=int, default=2,
                        help="Stop local epochs early if local-val AUROC fails to "
                             "improve for N consecutive epochs. 0 disables "
                             "(legacy v1 behavior — always run all local_epochs). "
                             "Restores best local checkpoint before contributing "
                             "to global aggregation. Applies to fedadapt/fedadaptproto only.")

    # Site_D root-cause diagnostic: per-round cosine similarity between each
    # site's local body-weight update and the resulting aggregate update.
    # Applies to ALL methods (the SharedBody aggregation mechanism is
    # identical across all of them) but intended primarily for fedavg first,
    # since it has no prototype/GRL/scaffold-variate complexity to confound
    # the read. Off by default — adds per-round compute + a new output file.
    p.add_argument("--track_gradient_conflict", action="store_true",
                        help="If set, save gradient_conflict.csv: per-round, "
                             "per-site cosine similarity between that site's "
                             "local body-weight delta and the resulting "
                             "aggregate delta. Low/negative similarity for a "
                             "site indicates its local update is fighting the "
                             "direction the aggregate actually moved in.")

    # v2.2: multi-cluster prototypes per class (FedAdapt-Proto only)
    p.add_argument("--n_clusters", type=int, default=1,
                        help="Number of prototype clusters per class. K=1 is the "
                             "legacy single-centroid behavior (v1/v2.0/v2.1). "
                             "K>1 enables per-class k-means clustering of "
                             "embeddings; each local sample aligns to its "
                             "nearest same-class global cluster instead of a "
                             "single averaged centroid. Lets sites with "
                             "systematically different real subpopulations "
                             "(e.g. ICU vs rural) align to different shared "
                             "clusters rather than fighting toward one mean.")

    # v2.3: per-site K override (FedAdapt-Proto only)
    p.add_argument("--n_clusters_per_site", type=str, default=None,
                        help="Per-site K override as comma-separated 'sid=K' pairs, "
                             "e.g. 'site_A=3,site_B=3,site_C=3,site_D=2,site_E=2'. "
                             "Sites not listed fall back to --n_clusters. Driven by "
                             "the silhouette diagnostic: site_E's AKI embeddings are "
                             "uni-modal (silhouette peaks at K=2, +0.243), so forcing "
                             "K=3 contributes a noise cluster to global aggregation. "
                             "Global aggregation K = max(per-site Ks); a site with "
                             "K=2 contributes 2 centers into the pooled space. "
                             "When unset, behavior is identical to v2.2.")

    # Misc
    p.add_argument("--test_frac",      type=float, default=0.20)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output_dir",     default="./results/",
                   help="Root output dir; method subdir created automatically")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FedAdapt training  |  method={args.method.upper()}")
    print(f"  rounds={args.rounds}  local_epochs={args.local_epochs}  "
          f"lr={args.lr}  batch={args.batch_size}")
    print(f"  embedding_dim={args.embedding_dim}  (DEBUG: confirms CLI arg received)")
    print(f"  output → {output_dir}")
    print(f"{'='*65}\n")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print("Loading site data...")
    sites = load_all_sites(
        data_dir, args.label, args.alpha, args.gamma,
        args.test_frac, args.batch_size, args.seed,
    )

    # ── 2. Auto-detect FL-gain CSV ───────────────────────────────────────────
    fl_gain_csv = None
    if args.fl_gain_csv:
        fl_gain_csv = Path(args.fl_gain_csv)
    else:
        candidates = list(data_dir.glob("fl_gain_index_*.csv"))
        if candidates:
            fl_gain_csv = candidates[0]
            print(f"  [fl_gain] auto-detected: {fl_gain_csv.name}")

    # ── 3. Local-only baseline (for correlation plot) ─────────────────────────
    print("\nRunning local-only baseline...")
    local_aurocs = run_local_only(
        sites, args.embedding_dim, args.hidden_dim,
        args.local_epochs * args.rounds, args.lr,
    )

    # ── 4. Build clients ─────────────────────────────────────────────────────
    print("\nBuilding FedAdaptClient per site...")
    clients = build_clients(sites, args.embedding_dim, args.hidden_dim, args.lambda_adv,
                            adaptive_lambda=args.adaptive_lambda)

    # ── 5. Federation ────────────────────────────────────────────────────────
    final_df, round_records = run_federation(
        method=args.method,
        sites=sites,
        clients=clients,
        args=args,
        output_dir=output_dir,
    )

    # ── 6. Save results ───────────────────────────────────────────────────────
    final_df.to_csv(output_dir / "final_metrics.csv", index=False)
    pd.DataFrame(round_records).to_csv(output_dir / "round_auroc.csv", index=False)
    print(f"\n  Final metrics:\n{final_df.to_string(index=False)}")

    # ── 7. Save models ────────────────────────────────────────────────────────
    for sid, client in clients.items():
        torch.save(client.state_dict(), output_dir / f"{sid}_model.pt")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_training_curves(round_records, args.method, output_dir)
    plot_final_comparison(final_df, args.method, output_dir)
    plot_fl_gain_correlation(
        final_df, local_aurocs, fl_gain_csv, args.method, output_dir
    )

    print(f"\n✅  Done — all outputs in {output_dir}/")


if __name__ == "__main__":
    main()