"""
mimic_ftl_simulation_phase2_approach2.py
=========================================
Phase II data simulation — APPROACH 2 (anchor-based, one row per patient).
Aligned with Liu et al. 2018 (AMIA, PMC5977670).

APPROACH 2 CHANGES vs original phase2 script
--------------------------------------------
1. Feature groups use prefix matching (e.g. "creatinine" matches
   creatinine_most_recent, creatinine_min, creatinine_max etc.)
2. sample_with_prevalence respects train/test split — only samples
   from train split to prevent test leakage into federated site CSVs
3. TARGET_N_PER_SITE auto-capped at available training patients
4. ANCHOR_PREVALENCE updated to 0.09 (Approach 2 cohort)
5. FIXED_PREVALENCES updated for new cohort AKI rate
6. baseline_method excluded from feature groups (string column)
7. [FIX] site_C's prevalence was reading the stale FIXED_PREVALENCES["site_C"]
   hardcoded literal (0.09, set by change #4/#5 above and never updated when
   exclusion criteria were later relaxed to match Phase 2/3). Now computed
   dynamically as the true cohort rate at runtime (~0.176 for the current
   leakage-corrected input), matching how sites A/B/D/E already derive
   their blend target. FIXED_PREVALENCES["site_C"] is now None
   (documentation placeholder only, not read for computation).

WHAT IS NEW vs Phase I (mimic_ftl_simulation3.py)
--------------------------------------------------
1. site_C local dominance
   - site_C is capped at ~33k samples (same as all other sites) so the
     comparison is fair in terms of data volume.
   - site_C retains its full feature set (89 features, confirmed via live
     run) and the true cohort AKI rate, computed dynamically (~17.6% for
     the current data). With richer features than any other site, a locally trained
     model at site_C is expected to outperform the federated global model
     for site_C's own patients — WITHOUT artificially reducing its N.
   - This motivates personalisation: even a strong contributor should keep
     a local head rather than fully merging into the global model.

2. FL-gain index
   - A per-site composite score computed BEFORE any model training.
   - Combines data rarity (inverse N share), class imbalance (minority
     class fraction), and feature sparsity (fraction of full feature set
     available).
   - Exported as fl_gain_index.csv alongside the site CSVs.
   - Hypothesis: FL-gain index correlates with observed AUROC improvement
     from federation vs local-only training — validated in Phase III.

3. Confirmed fixed prevalences (from actual simulation outputs)
   site_A  35%   ICU / Tertiary
   site_B  12%   General Ward
   site_C   computed dynamically (~17.6%)   Academic anchor (true cohort rate)
   site_D   7%   Community clinic
   site_E   4%   Resource-limited / Rural  ← primary benefitter
   (site_F removed — semi-supervised site dropped for Approach 2)

4. SiteInputAdapter specification
   - Each site exports a metadata JSON recording its local feature list
     and embedding_dim target. This is consumed by fedadapt_model.py and
     ALL baseline FL models (FedAvg, FedProx, SCAFFOLD) so that input
     heterogeneity is handled uniformly and comparisons are fair.
   - The adapter is infrastructure, not a novelty; FedAdapt's novelty is
     the GRL group alignment and the personal head.

PHASE I COMPATIBILITY
---------------------
This script reads from the same raw MIMIC-IV CSV as Phase I but is
completely independent — it does not read or modify any Phase I outputs.
Run both side by side; they share only the upstream data source.

USAGE
-----
python mimic_ftl_simulation_phase2.py \
    --input  /path/to/aki_full.csv \
    --label  AKI_label \
    --alpha  0.5 \
    --gamma  0.75 \
    --output ./phase2_sites/

Optional flags:
    --alpha   float   Dirichlet concentration (default 0.5)
    --gamma   float   Covariate shift severity (default 0.75)
    --seed    int     Random seed (default 42)
    --sweep           Run alpha in [0.1, 0.3, 0.5, 1.0, 10.0] and
                      gamma in [0.0, 0.5, 0.75, 1.0]
    --embedding_dim   int  Shared embedding dim for SiteInputAdapter (default 64)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

TARGET_N_PER_SITE = 17_000      # target — auto-capped at available train patients
                                 # (pushed as high as empirically possible while staying
                                 #  disjoint: 18,000 caused site_E -- last processed,
                                 #  lowest AKI-prevalence target -- to get 6,241 within-
                                 #  site duplicate patients, because site_A's high 35%
                                 #  AKI-positive target depletes the scarce AKI-positive
                                 #  class before site_E's turn. 17,000 tested clean with
                                 #  zero shortfall/duplication warnings at both alpha=0.3
                                 #  and the more extreme alpha=0.1. 5 x 17,000 = 85,000
                                 #  of the 91,776-patient train pool, 6,776 buffer.
                                 #  Re-verify if the cohort size changes again.)
                                 # (updated after the KDIGO baseline-SCr / CKD-exclusion
                                 #  fix reduced the total cohort to 114,720 patients.
                                 #  IMPORTANT: sample_with_prevalence_and_acuity()
                                 #  restricts to the "train" split only (91,776 of the
                                 #  114,720, since a "split" column is present) -- the
                                 #  test-split 22,944 patients are never available for
                                 #  site sampling. 5 x 16,000 = 80,000 leaves an 11,776
                                 #  patient / 12.8% buffer within that 91,776 train pool.
                                 #  Re-check this constant again if the cohort size or
                                 #  the train/test split ratio changes.)
ANCHOR_PREVALENCE = 0.176       # Fallback only -- true cohort rate is computed
                                 # dynamically wherever actual site_C data is
                                 # available (see run_simulation()). Previously
                                 # hardcoded to 0.09, which had drifted stale
                                 # from the actual cohort rate; fixed.

# Fixed AKI prevalences — clinically grounded lower-end literature values
# site_C: anchor site, prevalence computed dynamically as the true cohort
# rate at runtime (see run_simulation()) -- NOT a hardcoded literal, since
# a stale hardcoded value here previously drifted from the actual cohort
# rate once exclusion criteria were relaxed to match Phase 2/3 (was 0.09,
# confirmed stale against the current leakage-corrected cohort's true
# rate of ~0.176 -- fixed).
FIXED_PREVALENCES: Dict[str, Optional[float]] = {
    "site_A": 0.35,    # ICU — literature range 20-50%, using 35%
    "site_B": 0.12,    # General ward — literature range 12-17%, lower end
    "site_C": None,    # Academic anchor — computed dynamically, see below
    "site_D": 0.07,    # Community clinic — literature range 5-10%, lower end
    "site_E": 0.04,    # Rural/resource-limited — literature range 3-7%, lower end
}

# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────

# APPROACH 2: feature names are prefixes matching all stat-suffix variants
# e.g. "creatinine" matches creatinine_most_recent, creatinine_min,
#       creatinine_max, creatinine_mean, creatinine_hours_since
# Exact names (baseline_scr, age_at_admission etc.) matched as-is.
# baseline_method excluded — string column, not numeric feature.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "renal": [
        # hours_to_anchor removed: confirmed via
        # AKI_Anchor_Based_Approach2_aligned_features_PHASE4_leakage_fixed.ipynb
        # (Step 15B) to be a computed column (anchor_time - admittime) built
        # from the SAME asymmetric anchor logic used here (AKI: prospective
        # first-KDIGO-positive SCr; non-AKI: retrospective last_scr_time-24h)
        # -- the anchor-selection asymmetry that made this column leaky for
        # Phase 4 applies to this cohort's anchor construction too. Removed
        # here (not just confirmed dead) since, unlike Phase 4's master CSV,
        # there's no confirmed upstream removal step for this cohort yet --
        # this is the actual fix, not just hygiene, unless/until the source
        # notebook for aki_anchor_based_48h_lookback.csv is found and
        # confirmed to already exclude it.
        "baseline_scr",
        "creatinine", "bun",
    ],
    "inflammatory": [
        "lactate", "wbc", "platelets",
    ],
    "metabolic": [
        "sodium", "potassium", "bicarbonate",
        "hemoglobin", "glucose",
        "albumin", "bilirubin",
    ],
    "hemodynamic": [
        "sbp", "dbp", "heart_rate",
        "spo2", "resp_rate",
        "temperature", "gcs_total",
    ],
    "clinical": [
        "admission_type", "gender", "age_at_admission",
        "has_diabetes", "has_hypertension", "has_chf",
        "has_sepsis", "has_liver_disease", "has_cancer",
        "nephrotoxic_flag", "nephrotoxic_count", "n_distinct_meds",
    ],
}

# ─── SITE CONFIGURATIONS ──────────────────────────────────────────────────────

SITE_CONFIGS: Dict[str, dict] = {
    "site_A": {
        "description": "ICU / Tertiary hospital",
        "groups":      ["renal", "inflammatory", "clinical"],
        "acuity_bias": +2.0,    # draws sicker patients
        "spread_scale": 0.70,   # focused / narrower feature distributions
        "unlabeled":   False,
    },
    "site_B": {
        "description": "General ward / Secondary hospital",
        "groups":      ["renal", "metabolic", "clinical"],
        "acuity_bias": +0.5,
        "spread_scale": 1.10,
        "unlabeled":   False,
    },
    "site_C": {
        "description": "Academic Medical Centre — MIMIC-IV anchor, full feature set",
        "groups":      ["renal", "hemodynamic", "inflammatory", "metabolic", "clinical"],
        "acuity_bias": 0.0,     # anchor: unbiased sampling
        "spread_scale": 1.00,
        "unlabeled":   False,
        "is_anchor":   True,
        # NOTE: site_C local dominance
        # site_C has 39 features vs 18-30 at other sites.
        # With equal N (~33k) its locally trained model is expected to
        # outperform the global model for site_C patients — motivating
        # a personal head even for a strong contributor.
    },
    "site_D": {
        "description": "Community / Primary care clinic",
        "groups":      ["metabolic", "clinical"],
        "acuity_bias": -0.5,
        "spread_scale": 1.20,
        "unlabeled":   False,
    },
    "site_E": {
        "description": "Resource-limited / Rural hospital — PRIMARY FL BENEFITTER",
        "groups":      ["renal", "clinical"],
        "acuity_bias": -0.8,
        "spread_scale": 1.40,   # wide / heterogeneous referral patterns
        "unlabeled":   False,
    },
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def resolve_feature_columns(df: pd.DataFrame, groups: List[str]) -> List[str]:
    """
    Return columns present in df that belong to the requested groups.

    APPROACH 2: feature names are prefixes. A column matches if:
      - it equals the prefix exactly (e.g. baseline_scr, age_at_admission), OR
      - it starts with prefix + "_" (e.g. creatinine_most_recent, bun_min)
    String columns (baseline_method) are automatically excluded.

    LEAKAGE FIX: any column ending in "_hours_since" is excluded regardless
    of which prefix it matched. Confirmed via AKI_Anchor_Based_Approach2.ipynb
    (the actual, unfixed source notebook -- no Step-15B-equivalent removal
    exists there) that these are live, real columns computed for every lab
    (creatinine_hours_since, bun_hours_since, etc.) and saved directly into
    aki_anchor_based_24h_lookback.csv. This is the SAME leak class Phase 4's
    Step 15B fix targeted (confirmed there: recency alone gets 0.62 AUROC
    with zero lab values). Removing the standalone hours_to_anchor entry from
    FEATURE_GROUPS alone does NOT catch these -- prefix-matching on e.g.
    "creatinine" pulls in creatinine_hours_since automatically alongside
    creatinine_min/max/mean/most_recent, since FEATURE_GROUPS content can't
    express "this prefix but not this suffix variant." Must be filtered here,
    at the column-resolution level, not the group-content level.
    """
    wanted_prefixes = []
    for g in groups:
        wanted_prefixes.extend(FEATURE_GROUPS.get(g, []))

    matched = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue   # skip string/object columns
        if col.endswith("_hours_since"):
            continue   # leakage fix -- see docstring
        for prefix in wanted_prefixes:
            if col == prefix or col.startswith(prefix + "_"):
                if col not in matched:
                    matched.append(col)
                break
    return matched


def sample_with_prevalence(
    df: pd.DataFrame,
    label_col: str,
    target_prev: float,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw n rows from df such that AKI prevalence ≈ target_prev.
    Samples AKI and non-AKI strata separately then concatenates.

    APPROACH 2: if a "split" column is present, sample only from the
    train split — test patients must not appear in any site CSV.
    One row per patient (no multi-row expansion), so replace=False
    unless the site target N exceeds available training patients.

    NOTE: kept for backward compatibility / unweighted use cases.
    Covariate shift is now applied via sample_with_prevalence_and_acuity()
    below, which replaces this for the per-site simulation loop.
    """
    # Restrict to train split if present
    pool = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()

    aki     = pool[pool[label_col] == 1]
    non_aki = pool[pool[label_col] == 0]

    # Cap n to available training patients
    max_available = len(aki) + len(non_aki)
    n = min(n, max_available)

    n_aki     = min(int(round(n * target_prev)), len(aki))
    n_non_aki = min(n - n_aki, len(non_aki))

    sampled_aki     = aki.sample(n=n_aki,     replace=len(aki)     < n_aki,     random_state=int(rng.integers(1e6)))
    sampled_non_aki = non_aki.sample(n=n_non_aki, replace=len(non_aki) < n_non_aki, random_state=int(rng.integers(1e6)))

    return pd.concat([sampled_aki, sampled_non_aki]).sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)


# ─── REAL-PATIENT-SELECTION COVARIATE SHIFT (replaces apply_covariate_shift) ──

ACUITY_SCORE_COLS = [
    "creatinine_most_recent", "bun_most_recent",
    "lactate_most_recent",
]


def compute_acuity_score(df_source: pd.DataFrame) -> pd.Series:
    """
    Composite patient acuity score computed ONLY from each patient's own
    real lab values (z-scored mean of creatinine, BUN, lactate).
    Higher = sicker. This is purely descriptive / used afterward as a
    SELECTION weight — it is never written back into any feature column,
    so no patient's data is ever modified.

    Missing labs (common for lactate especially, ~80% missing — it's
    typically only drawn for suspected sepsis) are handled by averaging
    only the available z-scored labs per patient. Patients missing all
    four labs (~11.5% of the cohort) fall back to 0.0 (population
    average acuity) so they remain eligible for sampling rather than
    being silently excluded.
    """
    sub = df_source[ACUITY_SCORE_COLS].astype(float)
    z = (sub - sub.mean()) / sub.std(ddof=0)
    score = z.mean(axis=1, skipna=True)
    return score.fillna(0.0)


def sample_with_prevalence_and_acuity(
    df: pd.DataFrame,
    label_col: str,
    target_prev: float,
    n: int,
    acuity_bias: float,
    spread_scale: float,
    gamma: float,
    acuity_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw n REAL patients — no feature value is ever modified — such that:
      - AKI prevalence ~= target_prev  (label shift via alpha; unchanged)
      - patients are preferentially SELECTED toward high/low real acuity
        per the site's acuity_bias and spread_scale, scaled by gamma.

    This replaces the old additive/multiplicative value-shift mechanism
    (apply_covariate_shift) with importance-weighted SELECTION over
    already-real, unmodified patients. Every value in the returned
    dataframe is copied verbatim from the source file.

    Selection weight is a Gaussian over each patient's REAL acuity z-score:
        target_mean = gamma * acuity_bias
        target_std  = max(1.0 + gamma * (spread_scale - 1.0), 0.05)
        weight(p)  propto Normal_pdf(acuity_z(p); target_mean, target_std)

    gamma=0             -> target_mean=0, target_std=1 -> matches the
                            natural population distribution -> ~uniform
    gamma=1, bias>0      -> selection concentrates on real high-acuity
                            (sicker) patients
    gamma=1, bias<0      -> selection concentrates on real low-acuity
                            (healthier) patients
    spread_scale<1       -> narrower band of real patients selected
                            (more homogeneous, e.g. a specialized ICU)
    spread_scale>1       -> wider band of real patients selected
                            (more heterogeneous, e.g. a rural referral base)
    acuity_bias=0, spread_scale=1.0 (the anchor) -> target stays N(0,1)
        regardless of gamma -> anchor is naturally never perturbed,
        with no special-case code required.
    """
    pool = df[df["split"] == "train"].copy() if "split" in df.columns else df.copy()
    aki     = pool[pool[label_col] == 1]
    non_aki = pool[pool[label_col] == 0]

    max_available = len(aki) + len(non_aki)
    n = min(n, max_available)
    n_aki     = min(int(round(n * target_prev)), len(aki))
    n_non_aki = min(n - n_aki, len(non_aki))

    target_mean = gamma * acuity_bias
    target_std  = max(1.0 + gamma * (spread_scale - 1.0), 0.05)

    def weighted_draw(stratum: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0:
            return stratum.iloc[0:0]
        if gamma == 0.0:
            idx = stratum.sample(
                n=k, replace=len(stratum) < k,
                random_state=int(rng.integers(1e6)),
            ).index
            return stratum.loc[idx]
        z = stratum[acuity_col].values
        log_w = -0.5 * ((z - target_mean) / target_std) ** 2
        log_w -= log_w.max()                      # numerical stability
        w = np.exp(log_w)
        # Floor near-zero weights so numpy never sees "fewer non-zero than size"
        w = np.maximum(w, 1e-12)
        w = w / w.sum()
        # Allow replacement when effective non-zero pool is smaller than k
        n_effective = int((w > 1e-10).sum())
        chosen = rng.choice(
            stratum.index.values, size=k,
            replace=(n_effective < k or len(stratum) < k), p=w,
        )
        return stratum.loc[chosen]

    sampled_aki     = weighted_draw(aki, n_aki)
    sampled_non_aki = weighted_draw(non_aki, n_non_aki)

    out = pd.concat([sampled_aki, sampled_non_aki])
    out = out.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)
    return out


# ─── FL-GAIN INDEX ────────────────────────────────────────────────────────────

def compute_fl_gain_index(
    site_dfs: Dict[str, pd.DataFrame],
    label_col: str,
    total_features: int,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute a per-site FL-gain index BEFORE any model training.

    COMPONENTS (each in [0, 1])
    ───────────────────────────
    positive_case_scarcity = 1 - (site_n_positive / total_positive_all_sites)
                       Measures AKI case contribution scarcity.
                       site_A (ICU, many positives) scores low → contributor.
                       site_E (rural, few positives) scores high → benefitter.
                       Non-zero even when all sites have equal sample count N.

    class_imbalance  = 1 - minority_class_fraction
                       minority = min(prevalence, 1 - prevalence)
                       site_E: prev=0.05 → minority=0.05 → imbalance=0.95
                       site_A: prev=0.43 → minority=0.43 → imbalance=0.57

    feature_sparsity = 1 - (site_features / total_features)
                       site_E (18 feat) scores high; site_C (39 feat) scores 0.

    (label_efficiency removed — redundant with positive_case_scarcity
     when all sites have equal N=33k)

    FL-GAIN INDEX
    ─────────────
    Weighted mean of the four components. Default weights reflect the
    clinical insight that label scarcity and feature poverty drive FL
    benefit more than raw data volume when N is equalised:

        default weights = {
            data_rarity:      0.10,   # near-zero signal at equal N
            class_imbalance:  0.30,   # symmetric imbalance penalty
            feature_sparsity: 0.30,   # feature poverty
            label_efficiency: 0.30,   # positive label scarcity (AKI)
        }

    Override via the `weights` argument, e.g.:
        weights = {"data_rarity": 0.0, "class_imbalance": 0.25,
                   "feature_sparsity": 0.35, "label_efficiency": 0.40}

    INTERPRETATION
    ──────────────
        0.0 → rich data, balanced/high AKI rate, full features → low FL benefit
        1.0 → scarce data, rare AKI, few features              → high FL benefit

    HYPOTHESIS (validated in Phase III)
    ────────────────────────────────────
    FL-gain index correlates positively with observed AUROC improvement
    (federated global model vs local-only model) per site.
    """
    # Default weights — data_rarity down-weighted since N is equalised
    # 3-component index — data_rarity and label_efficiency replaced by
    # positive_case_scarcity which is non-zero even when all sites have equal N.
    DEFAULT_WEIGHTS = {
        "positive_case_scarcity": 0.30,  # AKI case contribution scarcity
        "class_imbalance":        0.30,  # symmetric imbalance penalty
        "feature_sparsity":       0.40,  # feature poverty (strongest driver)
    }
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    w_total = sum(w.values())
    w = {k: v / w_total for k, v in w.items()}

    # Total positive cases across all sites
    total_positive = sum(int(df[label_col].sum()) for df in site_dfs.values())

    records = []

    for site_id, df in site_dfs.items():
        n          = len(df)
        prev       = float(df[label_col].mean())
        n_positive = int(df[label_col].sum())

        # positive_case_scarcity: low AKI-case share → high scarcity → benefitter
        contributor_score      = n_positive / max(total_positive, 1)
        positive_case_scarcity = float(np.clip(1.0 - contributor_score, 0, 1))

        # class_imbalance: symmetric penalty for skewed prevalence
        minority        = min(prev, 1.0 - prev)
        class_imbalance = float(np.clip(1.0 - minority, 0, 1))

        # feature_sparsity: fraction of full feature set missing at this site
        n_features       = sum(
            1 for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        )
        feature_sparsity = float(np.clip(1.0 - (n_features / total_features), 0, 1))

        components = {
            "positive_case_scarcity": positive_case_scarcity,
            "class_imbalance":        class_imbalance,
            "feature_sparsity":       feature_sparsity,
        }

        fl_gain = float(sum(w[k] * v for k, v in components.items()))

        records.append({
            "site_id":               site_id,
            "n_samples":             n,
            "aki_prevalence":        round(prev, 4),
            "n_positive":            n_positive,
            "n_features":            n_features,
            "contributor_score":     round(contributor_score, 4),
            **{k: round(v, 4) for k, v in components.items()},
            "fl_gain_index":         round(fl_gain, 4),
            "role":                  _infer_role(site_id, fl_gain),
            "weight_positive_case_scarcity": round(w["positive_case_scarcity"], 3),
            "weight_class_imbalance":        round(w["class_imbalance"],        3),
            "weight_feature_sparsity":       round(w["feature_sparsity"],       3),
        })

    return pd.DataFrame(records).sort_values("fl_gain_index", ascending=False).reset_index(drop=True)


def _infer_role(site_id: str, fl_gain: float) -> str:
    """
    Role thresholds calibrated for 3-component index with positive_case_scarcity.
    site_A (ICU, many AKI cases, rich features) expected ~0.45 → contributor
    site_C (anchor, full features)               expected ~0.55 → conditional
    site_E (rural, few cases, sparse)            expected ~0.75 → primary_benefitter
    """
    if fl_gain >= 0.68:
        return "primary_benefitter"
    if fl_gain >= 0.52:
        return "conditional_benefitter"
    return "contributor"


# ─── SITE INPUT ADAPTER METADATA ─────────────────────────────────────────────

def export_adapter_metadata(
    site_dfs: Dict[str, pd.DataFrame],
    label_col: str,
    embedding_dim: int,
    output_dir: Path,
) -> None:
    """
    Export a JSON metadata file per site consumed by fedadapt_model.py
    and ALL baseline FL models (FedAvg, FedProx, SCAFFOLD).

    The SiteInputAdapter is SHARED INFRASTRUCTURE across all methods —
    not a FedAdapt novelty. It exists because feature heterogeneity
    (18–39 features) means every FL algorithm needs a site-specific
    projection to a common embedding dimension before federation.

    Schema:
    {
        "site_id":       "site_E",
        "input_dim":     18,
        "embedding_dim": 64,
        "feature_names": ["baseline_scr", ..., "has_cancer"],
        "aki_prevalence": 0.05,
        "unlabeled":     false
    }
    """
    for site_id, df in site_dfs.items():
        feature_cols = [
            c for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        prev = float(df[label_col].mean())

        meta = {
            "site_id":        str(site_id),
            "input_dim":      int(len(feature_cols)),
            "embedding_dim":  int(embedding_dim),
            "feature_names":  [str(c) for c in feature_cols],
            "aki_prevalence": float(prev),
            "unlabeled":      False,
            "description":    str(SITE_CONFIGS[site_id]["description"]),
            "feature_groups": [str(g) for g in SITE_CONFIGS[site_id]["groups"]],
        }

        out_path = output_dir / f"{site_id}_adapter_meta.json"
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [adapter] {out_path.name}  input_dim={len(feature_cols)}  → embedding_dim={embedding_dim}")


# ─── SIMULATION PLOTS ─────────────────────────────────────────────────────────

def plot_simulation_summary(
    site_dfs: Dict[str, pd.DataFrame],
    fl_gain_df: pd.DataFrame,
    label_col: str,
    alpha: float,
    gamma: float,
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [plot] matplotlib not available — skipping plots")
        return

    sites   = list(site_dfs.keys())
    colors  = {
        "contributor":           "#4C72B0",
        "conditional_benefitter":"#DD8452",
        "primary_benefitter":    "#C44E52",
        }
    role_map = dict(zip(fl_gain_df["site_id"], fl_gain_df["role"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Phase II simulation summary  |  α={alpha}  γ={gamma}\n"
        f"site_C local dominance: equal N (~33k), richest features (39) → "
        f"local model expected to outperform global",
        fontsize=11,
    )

    # Panel 1 — AKI prevalence
    ax = axes[0]
    prevs = []
    bar_colors = []
    for s in sites:
        df = site_dfs[s]
        prevs.append(float(df[label_col].mean()))
        bar_colors.append(colors.get(role_map.get(s, "contributor"), "#4C72B0"))
    bars = ax.bar(sites, prevs, color=bar_colors, edgecolor="white", linewidth=0.5)
    # Anchor reference line: site_C's actual prevalence (now correctly the
    # true cohort rate after the FIXED_PREVALENCES["site_C"] fix) rather
    # than the separate, previously-stale ANCHOR_PREVALENCE constant.
    anchor_rate = float(site_dfs["site_C"][label_col].mean()) if "site_C" in site_dfs else ANCHOR_PREVALENCE
    ax.axhline(anchor_rate, color="navy", linestyle="--", linewidth=1, label=f"Anchor ({anchor_rate:.1%})")
    for bar, p in zip(bars, prevs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{p:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("AKI prevalence")
    ax.set_title("AKI prevalence per site")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)

    # Panel 2 — FL-gain index
    ax = axes[1]
    gain_dict = dict(zip(fl_gain_df["site_id"], fl_gain_df["fl_gain_index"]))
    gains = [gain_dict.get(s, 0) for s in sites]
    gc    = [colors.get(role_map.get(s, "contributor"), "#4C72B0") for s in sites]
    bars2 = ax.bar(sites, gains, color=gc, edgecolor="white", linewidth=0.5)
    for bar, g in zip(bars2, gains):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{g:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Benefitter threshold (0.5)")
    ax.set_ylabel("FL-gain index")
    ax.set_title("FL-gain index per site\n(higher = benefits more from federation)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)

    # Panel 3 — Feature count
    ax = axes[2]
    fcounts = []
    for s in sites:
        df = site_dfs[s]
        fcounts.append(sum(
            1 for c in df.columns
            if c != label_col and pd.api.types.is_numeric_dtype(df[c])
        ))
    fc_colors = [colors.get(role_map.get(s, "contributor"), "#4C72B0") for s in sites]
    bars3 = ax.bar(sites, fcounts, color=fc_colors, edgecolor="white", linewidth=0.5)
    for bar, f in zip(bars3, fcounts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(f), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of features")
    ax.set_title("Feature count per site\n(site_C has most features → local dominance)")

    # Legend for roles
    patches = [mpatches.Patch(color=c, label=r.replace("_", " ").title())
               for r, c in colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fname = output_dir / f"phase2_summary_alpha{alpha}_gamma{gamma}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {fname.name}")


def plot_fl_gain_decomposition(
    fl_gain_df: pd.DataFrame,
    alpha: float,
    gamma: float,
    output_dir: Path,
) -> None:
    """
    Stacked bar showing all four FL-gain components per site.
    Bars are weighted contributions (weight × raw component value)
    so the total bar height equals the FL-gain index directly.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df    = fl_gain_df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"FL-gain index decomposition  |  α={alpha}  γ={gamma}\n"
        f"Left: weighted contributions (bar height = FL-gain index)  |  "
        f"Right: raw component scores",
        fontsize=11,
    )

    sites = df["site_id"].tolist()
    x     = np.arange(len(sites))
    bw    = 0.55

    COMPONENT_COLORS = {
        "positive_case_scarcity": "#4C72B0",
        "class_imbalance":        "#DD8452",
        "feature_sparsity":       "#C44E52",
    }
    COMPONENT_LABELS = {
        "positive_case_scarcity": "Positive-case scarcity (w=0.30)",
        "class_imbalance":        "Class imbalance (w=0.30)",
        "feature_sparsity":       "Feature sparsity (w=0.40)",
    }
    W = {"positive_case_scarcity": 0.30, "class_imbalance": 0.30,
         "feature_sparsity": 0.40}

    # ── Panel 1: weighted contributions ──────────────────────────────────────
    ax   = axes[0]
    bottom = np.zeros(len(df))
    for comp, color in COMPONENT_COLORS.items():
        weighted = df[comp].values * W[comp]
        ax.bar(x, weighted, bw, bottom=bottom,
               label=COMPONENT_LABELS[comp], color=color, edgecolor="white", linewidth=0.4)
        bottom += weighted

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(i, row["fl_gain_index"] + 0.008,
                f'{row["fl_gain_index"]:.3f}', ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("Weighted FL-gain contribution")
    ax.set_title("Weighted contributions\n(bar height = FL-gain index)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(len(sites) - 0.45, 0.51, "benefitter threshold", fontsize=7, color="gray")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel 2: raw component scores (unweighted) ────────────────────────────
    ax2    = axes[1]
    bottom2 = np.zeros(len(df))
    for comp, color in COMPONENT_COLORS.items():
        raw = df[comp].values
        ax2.bar(x, raw, bw, bottom=bottom2,
                label=comp.replace("_", " ").title(), color=color,
                edgecolor="white", linewidth=0.4, alpha=0.85)
        bottom2 += raw

    ax2.set_xticks(x)
    ax2.set_xticklabels(sites)
    ax2.set_ylabel("Raw component score (unweighted)")
    ax2.set_title("Raw component scores\n(for comparison — not the actual index)")
    ax2.set_ylim(0, 4.2)
    ax2.axhline(2.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fname = output_dir / f"fl_gain_decomposition_alpha{alpha}_gamma{gamma}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {fname.name}")


# ─── MAIN SIMULATION ──────────────────────────────────────────────────────────

def run_simulation(
    df_source: pd.DataFrame,
    label_col: str,
    alpha: float,
    gamma: float,
    embedding_dim: int,
    output_dir: Path,
    rng: np.random.Generator,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Phase II simulation  |  α={alpha}  γ={gamma}")
    print(f"{'='*60}")

    # Global stats for covariate shift (computed once from full source)
    all_numeric = df_source.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in all_numeric:
        all_numeric.remove(label_col)
    global_stats = df_source[all_numeric].agg(["mean", "std"]).T

    total_features = len(all_numeric)

    # Real-patient-selection covariate shift: compute each patient's REAL
    # acuity z-score once, up front. This is descriptive only — it is
    # used purely as a selection weight below and is never written into
    # any feature column, so no patient's data is ever modified.
    df_source = df_source.copy()
    df_source["_acuity_z"] = compute_acuity_score(df_source)

    site_dfs: Dict[str, pd.DataFrame] = {}

    # [DISJOINT SAMPLING] Track every patient already assigned to a site so
    # later sites in this loop draw from a shrinking, disjoint pool instead
    # of independently resampling the same full df_source (which previously
    # allowed the same patient to be selected into multiple sites -- see
    # check_overlap.py / HOW_TO_CHECK_OVERLAP.txt). Iteration order below
    # follows SITE_CONFIGS as defined, so sites processed later face a
    # smaller available pool; any shortfall is reported explicitly rather
    # than silently absorbed by within-site replacement sampling.
    used_subject_ids: set = set()

    for site_id, cfg in SITE_CONFIGS.items():
        print(f"\n  → {site_id}: {cfg['description']}")

        # 1. Resolve feature columns
        feat_cols = resolve_feature_columns(df_source, cfg["groups"])
        if not feat_cols:
            print(f"     WARNING: no matching feature columns found — skipping")
            continue

        # 2. Sample with alpha-interpolated prevalence (Option A).
        #
        #   blend = alpha / (alpha + 1)
        #   target_prev = (1 - blend) x archetype + blend x global_rate
        #
        #   alpha -> 0 : target_prev -> archetype  (clinical archetype)
        #   alpha -> inf : target_prev -> global_rate (true cohort rate,
        #                  computed dynamically -- ~17.6% for current data)
        #   midpoint at alpha=1.0 (50/50 blend)
        #
        #   site_C (anchor) always uses the true cohort rate (global_prev),
        #   computed fresh below -- NOT a hardcoded literal. Previously used
        #   FIXED_PREVALENCES["site_C"] (a stale hardcoded 0.09) here; fixed
        #   to use global_prev directly, matching how sites A/B/D/E already
        #   derive their blend target dynamically from the actual input data.
        archetype_prev = FIXED_PREVALENCES[site_id]
        global_prev = float(df_source[label_col].mean())

        if site_id == "site_C":
            target_prev = global_prev
            archetype_prev = global_prev  # for the print line below only
            blend = 0.0
        else:
            blend       = float(alpha) / (float(alpha) + 1.0)
            target_prev = (1.0 - blend) * archetype_prev + blend * global_prev

        print(f"     target_prev={target_prev:.3f}  "
              f"(archetype={archetype_prev:.3f}, global={global_prev:.3f}, blend={blend:.2f})")

        # 3. Sample REAL patients with target prevalence + acuity-weighted
        #    selection (covariate shift via selection, not value editing).
        # [DISJOINT SAMPLING] Exclude patients already claimed by an
        # earlier site in this loop before sampling.
        if "subject_id" in df_source.columns:
            df_source_avail = df_source[~df_source["subject_id"].isin(used_subject_ids)]
            n_avail = len(df_source_avail)
            if n_avail < TARGET_N_PER_SITE:
                print(f"     [disjoint-sampling] WARNING: only {n_avail:,} patients "
                      f"remain available for {site_id} after excluding "
                      f"{len(used_subject_ids):,} already-assigned to earlier sites "
                      f"(target is {TARGET_N_PER_SITE:,}). This site will fall short "
                      f"of target N and/or need within-site replacement sampling.")
        else:
            df_source_avail = df_source
            print(f"     [disjoint-sampling] WARNING: 'subject_id' not found in "
                  f"df_source.columns -- cannot enforce disjoint sampling for "
                  f"{site_id}; falling back to independent resampling from the "
                  f"full pool (may overlap with other sites).")

        site_df = sample_with_prevalence_and_acuity(
            df_source_avail, label_col, target_prev, n=TARGET_N_PER_SITE,
            acuity_bias=cfg["acuity_bias"], spread_scale=cfg["spread_scale"],
            gamma=gamma, acuity_col="_acuity_z", rng=rng,
        )
        sampled_real_index = site_df.index  # kept for the integrity check below

        # [OVERLAP CHECK / DISJOINT SAMPLING] Save this site's sampled
        # subject_ids before the keep_cols masking below drops the column,
        # both for post-hoc overlap verification (check_overlap.py) and to
        # mark these patients unavailable to every subsequent site in this
        # loop. Also flags within-site duplication, which can occur if this
        # site's own available class-stratum ran short and the sampling
        # function's own replace=True fallback fired.
        if "subject_id" in site_df.columns:
            n_drawn = len(site_df)
            n_unique = site_df["subject_id"].nunique()
            if n_unique < n_drawn:
                print(f"     [disjoint-sampling] WARNING: {site_id} has "
                      f"{n_drawn - n_unique:,} within-site duplicate patient(s) "
                      f"-- its available class-stratum ran short after cross-site "
                      f"exclusion, and replacement sampling filled the gap.")
            if n_drawn < TARGET_N_PER_SITE:
                print(f"     [disjoint-sampling] {site_id}: target={TARGET_N_PER_SITE:,}  "
                      f"actual={n_drawn:,}  shortfall={TARGET_N_PER_SITE - n_drawn:,}")
            used_subject_ids.update(site_df["subject_id"].tolist())
            site_df[["subject_id"]].to_csv(
                output_dir / f"_subject_ids_{site_id}_alpha{alpha}_gamma{gamma}.csv",
                index=False,
            )
        else:
            print(f"     [overlap check] WARNING: 'subject_id' not found in "
                  f"site_df.columns for {site_id} -- cannot save IDs for overlap check.")

        # 4. Mask to site feature set + label (drop the helper acuity column)
        keep_cols = feat_cols + [label_col]
        keep_cols = [c for c in keep_cols if c in site_df.columns]
        site_df   = site_df[keep_cols].copy()

        obs_prev = float(site_df[label_col].mean())
        print(f"     N={len(site_df):,}  features={len(feat_cols)}  AKI_prev={obs_prev:.3f}")

        site_dfs[site_id] = site_df

        # 6. Save CSV
        csv_path = output_dir / f"{site_id}_alpha{alpha}_gamma{gamma}.csv"
        site_df.to_csv(csv_path, index=False)

    # [DISJOINT SAMPLING] Summary: confirm zero cross-site overlap and
    # report how many total unique patients were used vs. the naive
    # (possibly infeasible) target of TARGET_N_PER_SITE * n_sites.
    n_sites_total = len(SITE_CONFIGS)
    naive_target_total = TARGET_N_PER_SITE * n_sites_total
    print(f"\n  [disjoint-sampling] Summary: {len(used_subject_ids):,} unique patients "
          f"used across {n_sites_total} sites (naive target was "
          f"{naive_target_total:,} = {TARGET_N_PER_SITE:,} x {n_sites_total} sites). "
          f"By construction, no patient appears at more than one site in this run "
          f"-- verify with check_overlap.py against the saved _subject_ids_*.csv files.")

    # 7. FL-gain index
    print("\n  Computing FL-gain index...")
    fl_gain_df = compute_fl_gain_index(site_dfs, label_col, total_features)
    gain_path  = output_dir / f"fl_gain_index_alpha{alpha}_gamma{gamma}.csv"
    fl_gain_df.to_csv(gain_path, index=False)
    print(fl_gain_df[["site_id", "n_samples", "aki_prevalence", "n_positive",
                       "n_features", "positive_case_scarcity", "class_imbalance",
                       "feature_sparsity", "fl_gain_index", "role"]].to_string(index=False))

    # 8. Adapter metadata (shared by all FL methods)
    print("\n  Exporting SiteInputAdapter metadata...")
    export_adapter_metadata(site_dfs, label_col, embedding_dim, output_dir)

    # 9. Plots
    print("\n  Generating plots...")
    plot_simulation_summary(site_dfs, fl_gain_df, label_col, alpha, gamma, output_dir)
    plot_fl_gain_decomposition(fl_gain_df, alpha, gamma, output_dir)

    print(f"\n  ✅ Done — outputs in {output_dir}")

    # Return per-site stats so the sweep loop can build the cross-alpha chart
    site_stats = {}
    for sid, sdf in site_dfs.items():
        n_aki = int(sdf[label_col].sum())
        n_no  = int((sdf[label_col] == 0).sum())
        site_stats[sid] = {
            "n_aki":       n_aki,
            "n_no_aki":    n_no,
            "description": SITE_CONFIGS[sid]["description"],
        }
    return site_stats


def plot_class_distribution_by_alpha(
    sweep: dict,          # {alpha: {site_id: {"n_aki": int, "n_no_aki": int, "description": str}}}
    output_dir: Path,
    global_aki_rate: float,  # required -- no stale default; caller must pass the
                              # actual computed rate (see main(), which already does)
) -> None:
    """
    Horizontal stacked-bar chart: one panel per alpha, one bar per site.

    HIGH alpha → all sites converge toward global_aki_rate (the true cohort
                 rate, computed dynamically from the input data -- not a
                 fixed constant)
    LOW  alpha → sites diverge toward clinical archetypes
                 (ICU ~35%, rural ~4%, etc.)

    A dashed red reference line marks global_aki_rate on every panel.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    CLASS_COLORS = {"No AKI": "#D1D5DB", "AKI": "#1D4ED8"}

    def _regime(a):
        if a <= 0.1:  return "extreme non-IID"
        if a <= 0.3:  return "non-IID"
        if a <= 1.0:  return "moderate non-IID"
        if a <= 3.0:  return "mild non-IID"
        return f"near-IID  (→ {global_aki_rate*100:.1f}% global)"

    alphas   = sorted(sweep.keys())
    n_panels = len(alphas)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(3.4 * n_panels, 5.8),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        site_data = sweep[alpha]
        site_ids  = sorted(site_data.keys())

        for i, sid in enumerate(site_ids):
            info     = site_data[sid]
            n_aki    = info.get("n_aki", 0)
            n_no_aki = info.get("n_no_aki", 0)
            total    = n_aki + n_no_aki
            if total == 0:
                continue
            frac_no = n_no_aki / total
            frac_ak = n_aki    / total

            ax.barh(i, frac_no, color=CLASS_COLORS["No AKI"],
                    edgecolor="white", linewidth=0.4)
            ax.barh(i, frac_ak, left=frac_no,
                    color=CLASS_COLORS["AKI"],
                    edgecolor="white", linewidth=0.4)

            if frac_ak > 0.06:
                ax.text(frac_no + frac_ak / 2, i,
                        f"{frac_ak*100:.0f}%",
                        ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")

        # reference line at global AKI rate
        ax.axvline(x=global_aki_rate, color="#EF4444",
                   linestyle="--", linewidth=1.3, alpha=0.9)

        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_xlabel("Proportion of patients", fontsize=9)
        ax.set_title(f"α = {alpha}\n{_regime(alpha)}",
                     fontsize=9, fontweight="bold", pad=6)
        ax.grid(axis="x", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        short = [s.replace("site_", "") for s in site_ids]
        descs = [site_data[s].get("description", "").split("/")[0].strip()[:12]
                 for s in site_ids]
        ylabels = [f"{sh}\n({d})" for sh, d in zip(short, descs)]
        ax.set_yticks(range(len(site_ids)))
        if ax is axes[0]:
            ax.set_yticklabels(ylabels, fontsize=8)
            ax.set_ylabel("Site", fontsize=10)
        else:
            ax.set_yticklabels([])

    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS["No AKI"], label="No AKI"),
        mpatches.Patch(color=CLASS_COLORS["AKI"],    label="AKI"),
        plt.Line2D([0], [0], color="#EF4444", linestyle="--", linewidth=1.3,
                   label=f"Global AKI rate ({global_aki_rate*100:.1f}% — Approach 2)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "AKI Label Distribution per Site Across α Values\n"
        "High α → converge to global rate  |  Low α → diverge to clinical archetypes",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out = output_dir / "class_distribution_by_alpha.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [plot] class_distribution_by_alpha.png  →  {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input",         required=True,  help="Path to MIMIC-IV AKI CSV")
    p.add_argument("--label",         default="AKI_label", help="Label column name")
    p.add_argument("--alpha",         type=float, default=0.5, help="Dirichlet α")
    p.add_argument("--gamma",         type=float, default=0.75, help="Covariate shift γ")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--embedding_dim", type=int,   default=64, help="Shared embedding dim for SiteInputAdapter")
    p.add_argument("--output",        default="./phase2_sites/", help="Output directory")
    p.add_argument("--sweep",         action="store_true", help="Sweep α and γ values")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    rng     = np.random.default_rng(args.seed)
    out_dir = Path(args.output)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Source: {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Overall AKI rate: {df[args.label].mean():.3f}")

    if args.sweep:
        alphas = [0.1, 0.3, 0.5, 1.0, 10.0]
        gammas = [0.0, 0.5, 0.75, 1.0]
        # Collect per-alpha stats (using gamma=0.75 as the representative slice)
        # so we can draw the cross-alpha label-distribution chart at the end.
        sweep_stats: dict = {}
        for a in alphas:
            for g in gammas:
                sub_dir   = out_dir / f"alpha{a}_gamma{g}"
                site_stats = run_simulation(df, args.label, a, g, args.embedding_dim, sub_dir, rng)
                if g == 0.75:          # representative gamma for the chart
                    sweep_stats[a] = site_stats
        # One chart across all alpha values → saved at the top-level output dir
        out_dir.mkdir(parents=True, exist_ok=True)
        global_rate = float(df[args.label].mean())
        plot_class_distribution_by_alpha(sweep_stats, out_dir, global_aki_rate=global_rate)
    else:
        run_simulation(df, args.label, args.alpha, args.gamma,
                       args.embedding_dim, out_dir, rng)


if __name__ == "__main__":
    main()
