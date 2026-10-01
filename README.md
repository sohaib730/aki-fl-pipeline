# aki-fl-pipeline

Federated-learning experiments for acute kidney injury (AKI) prediction on
MIMIC-IV. The pipeline has four stages, run in order:

1. **Get MIMIC-IV access** (credentialed PhysioNet + Google BigQuery)
2. **Preprocess** — two Colab notebooks build the Phase 1 and Phase 2 master cohorts
3. **Simulate disjoint sites** — split each cohort into non-overlapping synthetic hospitals
4. **Train** — one bash script per phase runs the full federated training grid (Code will be uploaded soon after manuscript submission)

Everything lives at the repo root on purpose: the shell scripts call the
Python files by bare name and expect the master CSVs in the working
directory. Don't move files into subfolders without editing the scripts.


---

## Stage 1 — Gain access to MIMIC-IV v3.1

Data: <https://physionet.org/content/mimiciv/3.1/>

MIMIC-IV is restricted-access. You need all of the following before the
notebooks in Stage 2 will run:

1. A PhysioNet account, **credentialed** (identity verification plus the
   CITI "Data or Specimens Only Research" training). Credentialing is
   reviewed by a human and can take a few days to a few weeks.
2. Sign the data use agreement for the MIMIC-IV v3.1 project page.
3. Under your PhysioNet profile → **Cloud**, add the Google account you
   will use in Colab. This is what grants that account read access to the
   public BigQuery dataset `physionet-data.mimiciv_3_1`.
4. Your own Google Cloud project with the BigQuery API enabled. The
   dataset is hosted by PhysioNet, but every query *runs* under your
   project, so your account needs the `bigquery.jobs.create` permission
   there (the BigQuery Job User or BigQuery User role). If you see
   `User does not have bigquery.jobs.create permission in project ...`,
   that role is what's missing.

The notebooks query BigQuery directly; nothing is downloaded from
PhysioNet by hand.

---

## Stage 2 — Preprocess with the Phase 1 and Phase 2 notebooks

Open in Google Colab and run top to bottom (each notebook authenticates
with `google.colab.auth`, then queries BigQuery):

| Notebook | Produces | Shape |
|---|---|---|
| `phase1_archetype_cohort.ipynb` | `aki_anchor_based_24h_lookback.csv` | 94 columns |
| `phase2_gpc_aligned_cohort.ipynb` | `aki_anchor_based_24h_lookback_aligned_features.csv` | 490 columns, ~144 MB |

Before running, edit the configuration cell near the top of each notebook:

```python
PROJECT_ID = 'your-gcp-project-id'          # ← your own GCP project
DATASET    = 'physionet-data.mimiciv_3_1'   # leave as is
```

Download the resulting CSV from Colab and place it in this directory.
Both notebooks produce the **same 114,720 patients with the same
per-patient train/test assignment** (91,776 train / 22,944 test). Confirm
that before going further:

```bash
python3 record_train_test_numbers.py \
  aki_anchor_based_24h_lookback.csv \
  aki_anchor_based_24h_lookback_aligned_features.csv
```

### How Phase 1 differs from Phase 2

The two notebooks share the same skeleton — anchor-based AKI labelling
following Liu et al. 2018 (first KDIGO-positive serum creatinine as the
anchor for AKI patients, features collected only up to `anchor − 24h`),
the same 3-tier KDIGO baseline-SCr hierarchy with CKD exclusion, the same
age 18–64 restriction, and the same leakage fixes (`hours_since` /
`hours_to_anchor` dropped). Where they diverge is *which features get
extracted*, and that difference exists because the two cohorts feed two
different simulation designs downstream.

**Phase 1 (clinical-archetype cohort)** extracts a compact, general
feature set: demographics, comorbidities, a 12-lab panel (albumin,
bicarbonate, bilirubin, BUN, creatinine, glucose, hemoglobin, lactate,
platelets, potassium, sodium, WBC), core vitals (no BMI) and medication
exposure. It is meant to be carved into five *archetypal* hospital types
(ICU, general ward, academic, community, rural) that differ in how many
features they record and in AKI prevalence.

**Phase 2 (GPC-aligned cohort)** extends the same extraction so that
MIMIC-IV can stand in for six *real* Greater Plains Collaborative sites
(KUMC, MCW, UIOWA, UPITT, UTSW, UofU). It adds the labs and vitals that
appear in those sites' random-forest feature-importance lists but were
missing from Phase 1 (calcium, chloride, phosphate, magnesium, total
protein, direct bilirubin, RDW, basophil %, lymphocyte %, BMI, plus 59
site-specific lab terms mapped by label text), and it adds ICD-9
diagnostic features: 70 codes universal to all six GPC sites and a
site-specific set layered on top. That is why it grows from 94 to 490
columns. At training time Phase 2 also drops the vitals that have no
counterpart in GPC production tables (heart rate, respiratory rate,
temperature, SpO2, GCS) — this is baked into the training scripts, not a
step you run.

Short version: same patients, different feature views.

---

## Stage 3 — Create disjoint site data for Phase 1 and Phase 2

Each master CSV is split into simulated hospital sites. The simulation
scripts sample **without cross-site overlap** — no patient appears at
more than one site within a condition — and draw only from the `train`
split. Two knobs control heterogeneity: `--alpha` (Dirichlet label skew)
and `--gamma` (covariate shift).

Files involved:

| File | Role |
|---|---|
| `phase1_archetype_simulation.py` | 5 sites × 17,000 patients (A ICU 35 %, B ward 12 %, C academic ≈ pooled, D community 7 %, E rural 4 % AKI prevalence) |
| `phase2_gpc_aligned_simulation.py` | 6 sites × 14,000 patients (`sim_KUMC`, `sim_MCW`, `sim_UIOWA`, `sim_UPITT`, `sim_UTSW`, `sim_UofU`) with GPC-derived acuity bias per site |
| `check_overlap.py` | verifies zero cross-site patient overlap from the `_subject_ids_*.csv` files each run writes |
| `HOW_TO_CHECK_OVERLAP.txt` | background on the overlap check |
| `run_disjoint_sites_data_gen.sh` | **the entry point** — smoke test, full grid, overlap verification, both phases |

Run it once, on its own, before anything else:

```bash
chmod +x run_disjoint_sites_data_gen.sh
./run_disjoint_sites_data_gen.sh
```

What it does, in order: a single-condition smoke test for each phase,
`check_overlap.py` on both, then the full grid — 20 conditions for
Phase 1 (α ∈ {0.1, 0.3, 0.5, 1.0, 10.0} × γ ∈ {0.0, 0.5, 0.75, 1.0}) and
3 conditions for Phase 2 (α/γ = 0.0/0.0, 0.5/0.75, 1.0/1.0) — and a final
overlap check. Output lands in `./phase1_data_disjoint/` and
`./phase2_data_disjoint/`, with alpha/gamma encoded in each filename
(e.g. `site_A_alpha0.3_gamma0.75.csv`, `sim_KUMC_alpha0.5_gamma0.75.csv`).

If you'd rather run a single condition by hand:

```bash
python3 phase1_archetype_simulation.py \
  --input aki_anchor_based_24h_lookback.csv --label AKI_label \
  --alpha 0.3 --gamma 0.75 --seed 42 --output ./phase1_data_disjoint/

python3 phase2_gpc_aligned_simulation.py \
  --input aki_anchor_based_24h_lookback_aligned_features.csv --label AKI_label \
  --alpha 0.5 --gamma 0.75 --seed 42 --output ./phase2_data_disjoint/

python3 check_overlap.py ./phase1_data_disjoint/
python3 check_overlap.py ./phase2_data_disjoint/
```

Two things to watch. First, `check_overlap.py` should report zero
overlap for every site pair; if it doesn't, stop. Second, watch the
generation log for `[disjoint-sampling]` shortfall or within-site
duplication warnings. `TARGET_N_PER_SITE` (17,000 for Phase 1, 14,000 for
Phase 2, set inside each simulation script) is the largest value that
runs clean against the 91,776-patient train pool — if your cohort size
differs from 114,720, re-check that constant before trusting the output.

---

## Stage 4 — Run training

All training scripts import `fedadapt_model_approach2.py` (shared model
definitions), so keep it alongside them. Each grid script skips a job
whose output already exists, so an interrupted run can simply be
restarted. Run the scripts one at a time, not concurrently — concurrent
runs have produced silent `--data_dir`/`--alpha`/`--gamma` misreads.

Methods compared: `fedadaptproto` (the proposed method; v2.3 = manual
K=2 clustering, v2.5 = automatic K selection with best-checkpoint
restoration), `fedadapt`, `fedavg`, `fedprox`, `scaffold`.

### Phase 1 (clinical-archetype)

```bash
chmod +x run_phase1_grid_v23.sh run_phase1_grid_v25.sh
./run_phase1_grid_v23.sh   # 5 methods × 20 conditions × 3 seeds = 300 jobs
./run_phase1_grid_v25.sh   # fedadaptproto v2.5 × 20 conditions × 3 seeds = 60 jobs
```

Uses `phase1_archetype_train_v23.py` and `phase1_archetype_train_v25.py`;
reads `./phase1_data_disjoint/`; writes `./results_phase1_grid_v23/` and
`./results_phase1_grid_v25/`. An optional tag argument
(`./run_phase1_grid_v25.sh postfix`) suffixes the output folder so runs
from different pipeline states stay separate on disk. `local_epochs=1`
for this cohort.

### Phase 2 (GPC-aligned)

```bash
chmod +x run_phase2_training.sh
./run_phase2_training.sh   # 9 (v2.3) + 9 (v2.5) + 36 (baselines) = 54 jobs
```

Uses `phase2_gpc_aligned_train_v23.py` and `phase2_gpc_aligned_train_v25.py`;
reads `./phase2_data_disjoint/`; writes `./results_phase2_training/`.
`local_epochs=3` for v2.3 and the baselines; v2.5 uses `local_epochs=1`
so the v2.3-vs-v2.5 comparison isolates the clustering strategy.

Each job writes `fl_gain_correlation.csv` (per-site ΔAUROC of federated
vs. local-only training) under its output directory — that file is also
what the resume-skip check looks for.

---

## Files in the repository

**Preprocessing (Stage 2)**

- `phase1_archetype_cohort.ipynb` — builds `aki_anchor_based_24h_lookback.csv`
- `phase2_gpc_aligned_cohort.ipynb` — builds `aki_anchor_based_24h_lookback_aligned_features.csv`
- `record_train_test_numbers.py` — verifies the train/test split and that both CSVs share the same patient population

**Simulation (Stage 3)**

- `phase1_archetype_simulation.py`, `phase2_gpc_aligned_simulation.py`
- `run_disjoint_sites_data_gen.sh` — entry point for both phases
- `check_overlap.py`, `HOW_TO_CHECK_OVERLAP.txt`

**Training (Stage 4)**
The code will be uploaded soon after manuscript submission
- `phase1_archetype_train_v23.py`, `phase1_archetype_train_v25.py`
- `phase2_gpc_aligned_train_v23.py`, `phase2_gpc_aligned_train_v25.py`
- `fedadapt_model_approach2.py` — shared model definitions
- `run_phase1_grid_v23.sh`, `run_phase1_grid_v25.sh`, `run_phase2_training.sh`

**Housekeeping**

- `requirements.txt` — `torch`, `pandas`, `numpy`, `scikit-learn` for the scripts; `google-cloud-bigquery`, `db-dtypes` for the notebooks
- `.gitignore` — keeps MIMIC-derived CSVs, data folders and results out of git
- `LICENSE` — GPL-2.0, inherited from the source repository

**Not in the repository:** the two master CSVs and everything derived
from them. Generate them yourself in Stage 2.

---

## Quick start (after Stage 1 and Stage 2)

```bash
pip install -r requirements.txt
ls aki_anchor_based_24h_lookback.csv aki_anchor_based_24h_lookback_aligned_features.csv
chmod +x *.sh
./run_disjoint_sites_data_gen.sh
./run_phase1_grid_v23.sh
./run_phase1_grid_v25.sh
./run_phase2_training.sh
```
