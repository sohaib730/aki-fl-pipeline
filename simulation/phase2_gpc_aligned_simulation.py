"""
mimic_ftl_simulation_phase4_gpc_aligned.py
=========================================
Phase IV data simulation — GPC-ALIGNED, 6-SITE ARCHITECTURE. Builds on
Phase III (mimic_ftl_simulation_phase3_aligned.py) but restructures the
site architecture to directly mirror the real GPC network's structure,
not just its feature *content*.

PHASE IV CHANGES vs Phase III
-------------------------------
1. All 6 real GPC sites represented (UTSW, UPITT, MCW, KUMC, UofU, UIOWA)
   — Phase III's 5-site architecture required dropping one real site;
   Phase IV adds sim_UIOWA as a 6th simulated site so none are dropped.
2. Uniform feature architecture, not varying richness per site. Every
   site now gets the IDENTICAL "shared98_core" group (18 MIMIC-IV columns
   confirmed matched to GPC's 19 real shared98 fields) + "gpc_dx_universal"
   (70 universal ICD-9 diagnosis codes from GPC's shared98 spec). This
   replaces Phase III's varying-richness archetypes (e.g. the old
   sim_MCW "anchor" with all 8 feature groups / 159 features while other
   sites had far fewer) — every real GPC site has the same shared98
   baseline, so Phase IV's simulated sites now do too.
3. Site-specific diagnosis codes, not just site-specific feature GROUPS.
   Each site additionally gets its own confirmed real-GPC site-specific
   ICD-9 codes (SITE_SPECIFIC_DX_COLUMNS): sim_UTSW 67, sim_MCW 56,
   sim_KUMC 32, sim_UofU 31, sim_UPITT 24, sim_UIOWA 6 codes — genuine
   inter-site heterogeneity from the real network, layered on top of the
   uniform shared98 baseline. Requires the cohort notebook's Section 5C
   to have been run (extracts dx_site_* columns via BigQuery) — degrades
   gracefully with a warning, not a crash, if those columns are absent.
4. GPC-REALISTIC (narrow) label distribution — NOT Phase III's wide,
   deliberately-heterogeneous prevalence sweep (6-40% simulated spread).
   Uses each site's own REAL, confirmed GPC prevalence rate as its
   archetype target:
       sim_UTSW   14.85%   sim_UPITT  13.90%   sim_MCW    13.41%
       sim_KUMC   12.91%   sim_UofU    9.99%   sim_UIOWA  13.42%
   All 6 target values are directly confirmed from the real GPC network's
   own per-site RF feature-importance export manifests (n_rows/n_positive
   counts), not estimated or re-scaled. Observed spread after alpha-blend
   sampling: ~10.8%-14.1%, matching real GPC's tight clustering — a
   deliberate contrast with Phase III's wide simulated range.
5. No architecturally-special "anchor" site. Phase III's sim_MCW had a
   zero-blend special case (always hit its exact target prevalence,
   bypassing alpha-interpolation). Removed in Phase IV — every site uses
   the same alpha-blended targeting, since no real GPC site is
   architecturally privileged over the others.

PHASE IV IS A SEPARATE, PARALLEL MODE, NOT A REPLACEMENT
------------------------------------------------------------
Phase III's wide-heterogeneity sweep (6-40% prevalence, varying feature
richness per site) remains the tool for the extremes-benefit / sweet-spot
/ ranking-table findings from that phase — those depend on having wide
prevalence variance to study. Phase IV is for testing whether findings
generalize to GPC's actual, much narrower real-world heterogeneity
structure. Run both, don't treat one as superseding the other.

INHERITED FROM PHASE III / PHASE II / APPROACH 2
-------------------------------------------------
1. Feature groups use prefix matching (e.g. "creatinine" matches
   creatinine_most_recent, creatinine_min, creatinine_max etc.)
2. sample_with_prevalence respects train/test split — only samples
   from train split to prevent test leakage into federated site CSVs
3. TARGET_N_PER_SITE auto-capped at available training patients
4. baseline_method excluded from feature groups (string column)
5. FL-gain index: a per-site composite score computed before any model
   training, combining positive case scarcity, class imbalance, and
   feature sparsity — exported as fl_gain_index.csv. Hypothesis (FL-gain
   index correlates with observed federated-vs-local AUROC improvement)
   was validated in Phase III; not yet re-tested against Phase IV's
   narrower prevalence structure.
6. SiteInputAdapter specification — each site exports a metadata JSON
   recording its local feature list and embedding_dim target, consumed
   uniformly by fedadapt_model.py and all baseline FL models.

USAGE
-----
python mimic_ftl_simulation_phase4_gpc_aligned.py \\
    --input  /path/to/aki_anchor_based_24h_lookback_aligned_features.csv \\
    --label  AKI_label \\
    --alpha  0.5 \\
    --gamma  0.75 \\
    --output ./phase4_gpc_aligned_sites/

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

TARGET_N_PER_SITE = 14_000      # target — auto-capped at available train patients
                                 # (pushed as high as empirically possible while staying
                                 #  disjoint: 15,000 caused sim_UIOWA -- last processed --
                                 #  to fall short by 1,138 patients. 14,000 tested clean
                                 #  with zero shortfall/duplication warnings across all 3
                                 #  conditions (alpha=0.0/0.5/1.0). 6 x 14,000 = 84,000 of
                                 #  the 91,776-patient train pool, 7,776 buffer. Re-verify
                                 #  if the cohort size changes again.)
                                 # (updated after the KDIGO baseline-SCr / CKD-exclusion
                                 #  fix reduced the total cohort to 114,720 patients.
                                 #  IMPORTANT: sample_with_prevalence() restricts to the
                                 #  "train" split only (91,776 of the 114,720, since a
                                 #  "split" column is present) -- the test-split 22,944
                                 #  patients are never available for site sampling.
                                 #  6 x 13,000 = 78,000 leaves a 13,776 patient / 15.0%
                                 #  buffer within that 91,776 train pool. Re-check this
                                 #  constant again if the cohort size or the train/test
                                 #  split ratio changes.)
ANCHOR_PREVALENCE = 0.1327      # GPC network pooled AKI rate (confirmed from real
                                 # per-site GPC data: KUMC/MCW/UIOWA/UPITT/UTSW/UofU)

# GPC-REALISTIC prevalences — this is a NEW, additional condition, not a
# replacement for the original FIXED_PREVALENCES sweep above. Uses 5 of the
# 6 real GPC network sites' actual confirmed AKI rates directly (dropping
# UIOWA, 13.42%, since it's nearly identical to MCW, 13.41% -- redundant
# for a 5-site simulation). Spread is deliberately much narrower than the
# original 6-43% simulated range, matching real GPC's tight clustering.
# Run this ALONGSIDE the original wide-heterogeneity sweep, not instead of
# it -- narrowing prevalence everywhere would remove the variance the
# extremes-benefit finding (sim_UTSW/E vs. B/C/D) depends on.
FIXED_PREVALENCES: Dict[str, Optional[float]] = {
    "sim_UTSW": 0.1485,  # ~ UTSW (highest of the 6 real GPC sites)
    "sim_UPITT": 0.1390,  # ~ UPITT
    "sim_MCW": 0.1341,  # ~ MCW
    "sim_KUMC": 0.1291,  # ~ KUMC
    "sim_UofU": 0.0999,  # ~ UofU (lowest of the 6 real GPC sites)
    "sim_UIOWA": 0.1342,  # ~ UIOWA (confirmed from manifest: N=100,415, prevalence=0.134213)
}

# Site-specific (non-shared98) GPC diagnosis codes, mapped from each real
# GPC site's own confirmed ICD-9 category codes (see the cohort notebook's
# "5C. Site-Specific GPC Diagnosis Codes" section for extraction). These
# supplement the universal 70 shared98 DX codes (all sites get those via
# the normal "clinical" or a dedicated DX feature group) with genuine,
# site-specific inter-site heterogeneity from the real GPC network.
# Column names must match the notebook's dx_site_{code} convention.
SITE_SPECIFIC_DX_COLUMNS: Dict[str, List[str]] = {
    "sim_UTSW": [f"dx_site_{c}" for c in [
        '038', '112', '197', '198', '268', '274', '279', '284', '348', '402', '404', '412',
        '433', '434', '437', '440', '443', '453', '459', '477', '478', '486', '492', '511',
        '514', '515', '516', '519', '536', '553', '562', '568', '569', '573', '574', '578',
        '592', '596', '600', '682', '707', '721', '722', '728', '783', '794', '998', 'E93',
        'V01', 'V13', 'V14', 'V16', 'V17', 'V42', 'V44', 'V46', 'V53', 'V64', 'V65', 'V66',
        'V67', 'V70', 'V71', 'V73', 'V76', 'V87', 'V88',
    ]],
    "sim_UPITT": [f"dx_site_{c}" for c in [
        '268', '402', '404', '412', '443', '486', '511', '553', '562', '600', '682', '707',
        '783', 'E03', 'E84', 'E87', 'E88', 'V13', 'V14', 'V17', 'V46', 'V67', 'V70', 'V76',
    ]],
    "sim_MCW": [f"dx_site_{c}" for c in [
        '038', '112', '197', '198', '238', '268', '274', '277', '284', '289', '309', '348',
        '362', '366', '402', '404', '412', '440', '443', '453', '459', '486', '511', '553',
        '568', '569', '573', '578', '596', '600', '682', '707', '721', '722', '728', '783',
        '791', '792', '794', '796', '998', 'E93', 'E94', 'V13', 'V14', 'V42', 'V46', 'V53',
        'V54', 'V59', 'V65', 'V67', 'V68', 'V70', 'V76', 'V87',
    ]],
    "sim_KUMC": [f"dx_site_{c}" for c in [
        '038', '268', '279', '284', '309', '348', '402', '404', '440', '443', '453', '486',
        '511', '536', '568', '569', '572', '573', '577', '682', '722', '783', '794', 'E93',
        'V14', 'V42', 'V46', 'V67', 'V70', 'V71', 'V73', 'V76',
    ]],
    "sim_UofU": [f"dx_site_{c}" for c in [
        '038', '197', '198', '277', '284', '289', '348', '402', '412', '453', '486', '511',
        '569', '573', '722', '728', '783', '794', 'E93', 'V13', 'V14', 'V16', 'V17', 'V42',
        'V46', 'V53', 'V54', 'V67', 'V70', 'V71', 'V87',
    ]],
    "sim_UIOWA": [f"dx_site_{c}" for c in [
        '443', '794', 'E93', 'V13', 'V42', 'V87',
    ]],
}

# Site-specific LAB features -- confirmed already extracted in the cohort
# notebook's Step 8 (LAB_ITEMIDS dict, real verified MIMIC-IV itemids, not
# new BigQuery work) but never wired into shared98_core. Cross-referenced
# directly against GPC's real *_dedup_rf_feature_list.csv per-site LAB rows
# (feature_class=='LAB', in_shared98=='no'): all 3 are exact clinical-name
# matches -- "Bicarbonate", "RBC Distribution Width", "Lymphocyte Percent"
# -- appearing in exactly these 5 sites and absent from UofU's real list,
# which independently matches the earlier mimic_gpc_feature_alignment.csv
# finding that UofU is missing rdw/lymphocyte_pct/bicarbonate/sodium from
# the 24 originally-mapped GPC features. Column names use the SAME base
# names as LAB_ITEMIDS keys (bicarbonate, rdw, lymphocyte_pct), so
# resolve_feature_columns' existing prefix-matching picks up their
# _min/_max/_mean/_most_recent/_hours_since stat columns automatically --
# no extraction changes needed, this data is already sitting in the cohort
# CSV. A larger pool of additional site-specific labs (71 unique GPC terms
# found in the real per-site lists, only 3 of which were already extracted)
# remains unmapped -- see mimic_gpc_feature_alignment.csv and the module
# docstring note below for the path to add the rest, which DOES require new
# notebook/BigQuery work, unlike this dict.
# Site-specific LAB features. First 3 columns per site (bicarbonate, rdw,
# lymphocyte_pct) were already extracted (Step 8's original LAB_ITEMIDS,
# real verified itemids) -- zero new BigQuery work. The rest require the
# notebook's NEW Section 5D (LIKE-pattern label matching against
# d_labitems, verification-first -- see notebook comments) to have been
# run; this dict degrades gracefully (same pattern as DX columns) if that
# hasn't happened yet -- df_source just won't have those columns, and
# resolve_columns_by_prefix silently returns fewer matches, no crash.
SITE_SPECIFIC_LAB_COLUMNS: Dict[str, List[str]] = {
    "sim_KUMC": ["bicarbonate", "rdw", "lymphocyte_pct", "alkaline_phosphatase", "alt", "ast", "band_neutrophils", "creatine_kinase", "esr", "ferritin", "free_t4", "hematocrit", "inr", "mch", "mchc_rbc", "mcv_rbc", "metamyelocytes_leukocytes", "neutrophil_percent", "non_hdl_cholesterol", "oxygen_saturation", "po2_blda", "urine_ph", "urine_sodium"],
    "sim_MCW": ["bicarbonate", "rdw", "lymphocyte_pct", "activated_clotting_time", "alkaline_phosphatase", "alt", "ast", "co2_bldv_scnc", "creatine_kinase", "eosinophil_bld_manual", "eosinophil_nfr_bld", "est_average_glucose_bld_ghb_est_mcnc", "ferritin", "hematocrit", "imm_granulocytes_nfr_bld", "inr", "ldh_serpl_l_to_p_ccnc", "lymphocytes_nfr_bld", "mch", "mchc_rbc", "mcv_rbc", "monocytes_nfr_bld", "neutrophil_percent", "neutrophils_bld", "non_hdl_cholesterol", "oxygen_saturation", "po2_blda", "pt_bld", "urate", "urine_ph", "urine_sodium"],
    "sim_UIOWA": ["bicarbonate", "rdw", "lymphocyte_pct", "alkaline_phosphatase", "alt", "amylase", "anion_gap_serpl_calculated_3ions_scnc", "aptt_bld", "ast", "band_neutrophils", "base_excess", "base_excess_blda_calc_scnc", "ca_i_bld_mcnc", "co2_bldv_scnc", "creatine_kinase", "eosinophil_bld_manual", "eosinophil_nfr_bld", "esr", "est_average_glucose_bld_ghb_est_mcnc", "ethanol", "ferritin", "free_t4", "gamma_gt", "hematocrit", "imm_granulocytes_nfr_bld", "inr", "lymphocytes_nfr_bld", "mch", "mchc_rbc", "mcv_rbc", "metamyelocytes_leukocytes", "monocytes_nfr_bld", "myelocytes_leukocytes", "neutrophil_percent", "neutrophils_bld", "neuts_seg_bld", "non_hdl_cholesterol", "nrbc_bld_rto", "osmolality", "pco2_temp_adj_blda", "ph_temp_adj_blda", "pmv_bld_rees_ecker", "po2_blda", "po2_temp_adj_bldv", "prealbumin", "pt_bld", "tsh_serpl_dl_0_05_miu_l_acnc", "urine_osmolality", "urine_ph", "urine_sodium", "urine_specific_gravity", "variant_lymphs_bld_manual"],
    "sim_UPITT": ["bicarbonate", "rdw", "lymphocyte_pct", "alkaline_phosphatase", "alt", "ast", "base_excess_blda_calc_scnc", "eosinophil_nfr_bld", "esr", "ferritin", "hematocrit", "inr", "mch", "mchc_rbc", "mcv_rbc", "monocytes_nfr_bld", "osmolality", "oxygen_saturation", "po2_blda", "prealbumin", "urine_ph", "urine_sodium"],
    "sim_UTSW": ["bicarbonate", "rdw", "lymphocyte_pct", "activated_clotting_time", "alkaline_phosphatase", "alt", "amylase", "anion_gap_serpl_calculated_3ions_scnc", "aptt_bld", "ast", "base_excess_blda_calc_scnc", "creatine_kinase", "eosinophil_bld_manual", "ferritin", "free_t4", "haptoglobin", "imm_granulocytes_nfr_bld", "mch", "mchc_rbc", "mcv_rbc", "metamyelocytes_leukocytes", "myelocytes_leukocytes", "non_hdl_cholesterol", "oxygen_saturation", "po2_blda", "prealbumin", "reticulocyte_percent", "urate", "urine_osmolality", "urine_protein", "urine_sodium"],
    "sim_UofU": ["albumin_serpl_elph_mcnc", "alkaline_phosphatase", "alt", "ast", "band_neutrophils", "creatine_kinase", "est_average_glucose_bld_ghb_est_mcnc", "hematocrit", "inr", "ldh_serpl_l_to_p_ccnc", "neutrophils_nfr_fld", "non_hdl_cholesterol", "prealbumin", "urate", "urine_ph", "variant_lymphocytes"],
}

# ─── FEATURE GROUPS ───────────────────────────────────────────────────────────

# GPC-ALIGNED: feature names are prefixes matching all stat-suffix variants
# e.g. "creatinine" matches creatinine_most_recent, creatinine_min,
#       creatinine_max, creatinine_mean, creatinine_hours_since
# Exact names (baseline_scr, age_at_admission etc.) matched as-is.
# baseline_method excluded — string column, not numeric feature.
#
# Groups mirror GPC RF feature list `feature_class` / `clinical_meaning`
# categories (see mimic_gpc_feature_alignment.csv) so that site-level
# feature masking produces feature-sparsity conditions directly
# comparable to GPC's 6-site feature coverage.
FEATURE_GROUPS: Dict[str, List[str]] = {
    # ── GPC-ALIGNED 6-SITE MODE ─────────────────────────────────────────
    # "shared98_core": the 18 MIMIC-IV features confirmed matched against
    # GPC's real shared98 spec (19 GPC fields -> 18 distinct MIMIC columns,
    # since "Bilirubin direct" and "Direct bilirubin" are 2 separate GPC
    # entries mapping to the same bilirubin_dir column). ALL 6 sites in
    # this mode get this group -- it's the universal baseline, mirroring
    # what every real GPC site has in common.
    "shared98_core": [
        "age_at_admission", "gender", "dbp", "sbp", "bmi",
        "creatinine", "bun", "glucose", "calcium", "chloride", "potassium",
        "magnesium", "lactate", "phosphate", "bilirubin", "bilirubin_dir",
        "total_protein", "basophils_pct",
    ],
    # "gpc_dx_universal": the 70 universal ICD-9 shared98 diagnosis flags
    # (notebook Section 5B, columns named dx_{code}). ALL 6 sites get this.
    # NOTE: explicit code list, NOT a "dx" wildcard prefix -- a wildcard
    # would also match dx_site_{code} columns (site-specific DX, section
    # 5C), silently giving every site all 86 site-specific codes too and
    # defeating the point of site-specific differentiation.
    "gpc_dx_universal": [f"dx_{c}" for c in [
        '041', '244', '250', '263', '272', '275', '276', '278', '280', '285', '287', '288',
        '296', '300', '305', '311', '327', '338', '357', '401', '403', '414', '416', '424',
        '425', '426', '427', '428', '429', '458', '491', '493', '496', '518', '530', '564',
        '571', '584', '585', '593', '599', '715', '719', '724', '729', '733', '780', '781',
        '782', '784', '785', '786', '787', '788', '789', '790', '793', '799', '995', '996',
        'V05', 'V10', 'V12', 'V15', 'V43', 'V45', 'V49', 'V58', 'V72', 'V85',
    ]],
    "renal": [
        "baseline_scr", "hours_to_anchor",
        "creatinine", "bun",
    ],
    "metabolic_panel": [
        # GPC-aligned: Sodium, Potassium, Chloride, Bicarbonate, Calcium,
        # Phosphate, Magnesium, Glucose — all in GPC shared98 (mean rank
        # 4.7 - 78.5 across 6 GPC sites)
        "sodium", "potassium", "chloride", "bicarbonate",
        "calcium", "phosphate", "magnesium", "glucose",
    ],
    "hepatic": [
        # GPC-aligned: Albumin, Protein Mass, Bilirubin total/direct
        "albumin", "total_protein", "bilirubin", "bilirubin_dir",
    ],
    "hematology": [
        # wbc, platelets, hemoglobin retained (no direct GPC match found
        # for wbc/platelets, but standard CBC labs — kept as MIMIC-IV
        # clinical signal). rdw/basophils_pct/lymphocyte_pct are
        # GPC-aligned (mean rank 5.2 - 70.5).
        "wbc", "hemoglobin", "platelets",
        "rdw", "basophils_pct", "lymphocyte_pct",
    ],
    "inflammatory": [
        "lactate",
    ],
    "vitals_gpc": [
        # GPC-aligned: maps directly to GPC VITAL_TIME class
        # (SYSTOLIC, DIASTOLIC, ORIGINAL_BMI)
        "sbp", "dbp", "bmi",
    ],
    "vitals_mimic": [
        # MIMIC-IV only — no GPC RF-list equivalent found, retained as
        # local enrichment (does not claim GPC comparability)
        "heart_rate", "resp_rate", "spo2", "temperature", "gcs_total",
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
    # acuity_bias / spread_scale below are derived from real GPC data
    # (site_distribution_summary.csv), replacing earlier unvalidated
    # placeholders carried over from the archetype (site_A..F) design.
    #
    # acuity_bias: z-scored composite of AGE + Creatinine + BUN + Lactate + WBC
    #   (all LOINC-identified per Zijian's email; all +1 direction -- increase
    #   with sickness). KUMC's WBC excluded (unrecoverable units, per Zijian);
    #   all other sites use the full 5-feature set (UTSW's WBC unit-conversion
    #   bug was confirmed recoverable by Zijian and corrected before scoring).
    # spread_scale: mean per-site std ratio (site std / cross-site mean std)
    #   across the 40 shared continuous features common to all 6 sites.
    #   Real range is 0.985-1.028 -- sites are far more homogeneous in spread
    #   than the old 0.70-1.40 placeholder assumed.
    "sim_UTSW": {
        "description": "GPC-aligned site ~ UTSW",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": -0.11,
        "spread_scale": 0.985,
        "unlabeled":   False,
    },
    "sim_UPITT": {
        "description": "GPC-aligned site ~ UPITT",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": +0.80,
        "spread_scale": 0.992,
        "unlabeled":   False,
    },
    "sim_MCW": {
        "description": "GPC-aligned site ~ MCW",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": -0.03,
        "spread_scale": 1.028,
        "unlabeled":   False,
    },
    "sim_KUMC": {
        "description": "GPC-aligned site ~ KUMC",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": -0.33,
        "spread_scale": 0.988,
        "unlabeled":   False,
    },
    "sim_UofU": {
        "description": "GPC-aligned site ~ UofU",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": +0.13,
        "spread_scale": 1.019,
        "unlabeled":   False,
    },
    "sim_UIOWA": {
        "description": "GPC-aligned site ~ UIOWA",
        "groups":      ["shared98_core", "gpc_dx_universal"],
        "acuity_bias": -0.53,
        "spread_scale": 0.987,
        "unlabeled":   False,
    },
}
# All 6 sites get the SAME base groups (shared98_core + gpc_dx_universal) --
# this is the deliberate architecture change for GPC-aligned mode: unlike
# the original archetype-based simulation (where feature RICHNESS varied
# site-to-site, e.g. sim_MCW/site_C as the "anchor" with all 8 groups),
# every real GPC site has the same shared98 baseline. Differentiation here
# comes from (a) each site's own prevalence target (FIXED_PREVALENCES) and
# (b) each site's own unique DX codes (SITE_SPECIFIC_DX_COLUMNS, added in
# the main loop below, not via "groups"). acuity_bias/spread_scale values
# are carried over from the original archetypes as reasonable placeholders
# for now -- not re-derived from real GPC data, since GPC's feature-list
# exports don't include acuity/spread information, only feature presence
# and importance rankings.

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def resolve_feature_columns(df: pd.DataFrame, groups: List[str]) -> List[str]:
    """
    Return columns present in df that belong to the requested groups.

    APPROACH 2: feature names are prefixes. A column matches if:
      - it equals the prefix exactly (e.g. baseline_scr, age_at_admission), OR
      - it starts with prefix + "_" (e.g. creatinine_most_recent, bun_min)
    String columns (baseline_method) are automatically excluded.
    """
    wanted_prefixes = []
    for g in groups:
        wanted_prefixes.extend(FEATURE_GROUPS.get(g, []))

    return resolve_columns_by_prefix(df, wanted_prefixes)


def resolve_columns_by_prefix(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    """
    Same matching logic as resolve_feature_columns, but takes raw base-name
    prefixes directly instead of FEATURE_GROUPS keys. Used for
    SITE_SPECIFIC_LAB_COLUMNS, which holds actual column-name prefixes
    (e.g. "bicarbonate") rather than group names to look up.
    """
    matched = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue   # skip string/object columns
        for prefix in prefixes:
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


def apply_covariate_shift(
    df: pd.DataFrame,
    feature_cols: List[str],
    global_stats: pd.DataFrame,
    acuity_bias: float,
    spread_scale: float,
    gamma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Apply two-mechanism covariate shift scaled by gamma.

    Mechanism A — acuity-driven mean shift (z-score units):
        delta_z ~ N(acuity_bias * gamma, 0.1)

    Mechanism B — spread perturbation:
        effective_scale = 1.0 + gamma * (spread_scale - 1.0)
    """
    df = df.copy()
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric_cols or gamma == 0.0:
        return df

    # Draw per-site delta_z once (coherent shift across all features)
    delta_z        = rng.normal(acuity_bias * gamma, 0.1)
    eff_scale      = 1.0 + gamma * (spread_scale - 1.0)

    for col in numeric_cols:
        if col not in global_stats.index:
            continue
        mu  = global_stats.loc[col, "mean"]
        sig = global_stats.loc[col, "std"]
        if sig == 0 or np.isnan(sig):
            continue
        # Shift in raw units: delta_z standard deviations
        df[col] = df[col] + delta_z * sig
        # Scale spread around new mean
        new_mean   = mu + delta_z * sig
        df[col]    = new_mean + eff_scale * (df[col] - new_mean)

    return df


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
                       sim_UTSW (ICU, many positives) scores low → contributor.
                       sim_UofU (rural, few positives) scores high → benefitter.
                       Non-zero even when all sites have equal sample count N.

    class_imbalance  = 1 - minority_class_fraction
                       minority = min(prevalence, 1 - prevalence)
                       sim_UofU: prev=0.05 → minority=0.05 → imbalance=0.95
                       sim_UTSW: prev=0.43 → minority=0.43 → imbalance=0.57

    feature_sparsity = 1 - (site_features / total_features)
                       sim_UofU (18 feat) scores high; sim_MCW (39 feat) scores 0.

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
    sim_UTSW (ICU, many AKI cases, rich features) expected ~0.45 → contributor
    sim_MCW (anchor, full features)               expected ~0.55 → conditional
    sim_UofU (rural, few cases, sparse)            expected ~0.75 → primary_benefitter
    """
    if fl_gain >= 0.65:
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
        "site_id":       "sim_UofU",
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
        f"Phase IV simulation summary — 6 real GPC sites, uniform shared98 baseline  |  α={alpha}  γ={gamma}",
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
    ax.axhline(ANCHOR_PREVALENCE, color="navy", linestyle="--", linewidth=1, label=f"Anchor ({ANCHOR_PREVALENCE:.1%})")
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
    ax.set_title("Feature count per site\n(uniform shared98 baseline + each site's own DX codes)")

    # Legend for roles
    patches = [mpatches.Patch(color=c, label=r.replace("_", " ").title())
               for r, c in colors.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fname = output_dir / f"phase4_summary_alpha{alpha}_gamma{gamma}.png"
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
    print(f"Phase IV simulation (GPC-aligned, 6-site, narrow prevalence)  |  α={alpha}  γ={gamma}")
    print(f"{'='*60}")

    # Global stats for covariate shift (computed once from full source)
    all_numeric = df_source.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in all_numeric:
        all_numeric.remove(label_col)
    global_stats = df_source[all_numeric].agg(["mean", "std"]).T

    total_features = len(all_numeric)

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

        # 1b. Add this site's own site-specific GPC diagnosis codes, if any
        # exist in df_source (only present when the notebook's Section 5C
        # extraction has been run) — matches this simulated site's real
        # GPC counterpart's genuine site-specific DX heterogeneity.
        site_dx_cols = SITE_SPECIFIC_DX_COLUMNS.get(site_id, [])
        available_site_dx = [c for c in site_dx_cols if c in df_source.columns]
        if available_site_dx:
            feat_cols = feat_cols + [c for c in available_site_dx if c not in feat_cols]
            print(f"     + {len(available_site_dx)} site-specific DX codes "
                  f"(of {len(site_dx_cols)} expected)")
        elif site_dx_cols:
            print(f"     WARNING: {len(site_dx_cols)} site-specific DX codes expected "
                  f"but none found in data — run notebook Section 5C first")

        # 1c. Add this site's own site-specific LAB features, if the base
        # columns exist in df_source (they already do -- see
        # SITE_SPECIFIC_LAB_COLUMNS docstring, Step 8's LAB_ITEMIDS already
        # extracts bicarbonate/rdw/lymphocyte_pct). Uses the same prefix
        # matching as shared98_core, just applied to a site-specific base
        # name list instead of a universal one.
        site_lab_base = SITE_SPECIFIC_LAB_COLUMNS.get(site_id, [])
        site_lab_cols = resolve_columns_by_prefix(df_source, site_lab_base)
        available_site_lab = [c for c in site_lab_cols if c not in feat_cols]
        if available_site_lab:
            feat_cols = feat_cols + available_site_lab
            print(f"     + {len(available_site_lab)} site-specific LAB columns "
                  f"({', '.join(site_lab_base)})")
        elif site_lab_base:
            print(f"     WARNING: site-specific LAB base names {site_lab_base} "
                  f"expected but no matching columns found in data")

        if not feat_cols:
            print(f"     WARNING: no matching feature columns found — skipping")
            continue

        # 2. Sample with alpha-interpolated prevalence (Option A).
        #
        #   blend = alpha / (alpha + 1)
        #   target_prev = (1 - blend) x archetype + blend x global_rate
        #
        #   alpha -> 0 : target_prev -> archetype  (clinical archetype)
        #   alpha -> inf : target_prev -> global_rate (~12.6%, IID baseline)
        #   midpoint at alpha=1.0 (50/50 blend)
        #
        #   All 6 sites use the same alpha-blended logic in this mode --
        #   no "anchor" exception (unlike the original archetype-based
        #   simulation), since every real GPC site is architecturally
        #   equivalent here (same shared98_core baseline).
        archetype_prev = FIXED_PREVALENCES[site_id]
        global_prev = float(df_source[label_col].mean())

        blend       = float(alpha) / (float(alpha) + 1.0)
        target_prev = (1.0 - blend) * archetype_prev + blend * global_prev

        print(f"     target_prev={target_prev:.3f}  "
              f"(archetype={archetype_prev:.3f}, global={global_prev:.3f}, blend={blend:.2f})")

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

        site_df = sample_with_prevalence(
            df_source_avail, label_col, target_prev,
            n=TARGET_N_PER_SITE, rng=rng,
        )

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

        # 3. Mask to site feature set + label
        keep_cols = feat_cols + [label_col]
        keep_cols = [c for c in keep_cols if c in site_df.columns]
        site_df   = site_df[keep_cols].copy()

        # 4. Apply covariate shift
        site_df = apply_covariate_shift(
            site_df, feat_cols, global_stats,
            acuity_bias=cfg["acuity_bias"],
            spread_scale=cfg["spread_scale"],
            gamma=gamma, rng=rng,
        )

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
    global_aki_rate: float = 0.125,
) -> None:
    """
    Horizontal stacked-bar chart: one panel per alpha, one bar per site.

    HIGH alpha → all sites converge toward global_aki_rate (~12.6 %)
    LOW  alpha → sites diverge toward clinical archetypes
                 (ICU ~45-50 %, rural ~5-9 %, etc.)

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
                             sharey=True)
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
    p.add_argument("--output",        default="./phase4_gpc_aligned_sites/", help="Output directory")
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
        gammas = [0.75]  # only gamma=0.75 is ever used for the cross-alpha chart below;
                          # the other gamma values were computed but discarded, wasting
                          # significant compute and disk for no effect on this figure.
        # Collect per-alpha stats (using gamma=0.75 as the representative slice)
        # so we can draw the cross-alpha label-distribution chart at the end.
        sweep_stats: dict = {}
        for a in alphas:
            for g in gammas:
                sub_dir   = out_dir / f"alpha{a}_gamma{g}"
                site_stats = run_simulation(df, args.label, a, g, args.embedding_dim, sub_dir, rng)
                if g == 0.75:          # representative gamma for the chart
                    sweep_stats[a] = site_stats
                # Only the returned site_stats dict (prevalence/counts) is needed for
                # the final chart -- delete the large per-site CSVs immediately rather
                # than accumulate all conditions' full datasets on disk.
                import glob as _glob, os as _os
                for _f in _glob.glob(str(sub_dir / "*.csv")):
                    _os.remove(_f)
        # One chart across all alpha values → saved at the top-level output dir
        out_dir.mkdir(parents=True, exist_ok=True)
        global_rate = float(df[args.label].mean())
        plot_class_distribution_by_alpha(sweep_stats, out_dir, global_aki_rate=global_rate)
    else:
        run_simulation(df, args.label, args.alpha, args.gamma,
                       args.embedding_dim, out_dir, rng)


if __name__ == "__main__":
    main()