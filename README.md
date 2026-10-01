# Source-linked provenance audit of the Heart Failure Prediction Dataset

Code and record-level provenance accompanying the manuscript *"Outcome-encoding value completion in a widely used heart-disease benchmark: a source-linked audit of the 918-patient Heart Failure Prediction Dataset"*.

## Data (not redistributed here)
1. **Distributed file** – `heart.csv` from Kaggle, *Heart Failure Prediction Dataset* (fedesoriano, 2021), doi:10.34740/KAGGLE/DS/2162210.
2. **Source files** – UCI Heart Disease repository (doi:10.24432/C52P4X):
   `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`.

Scripts look for these files under `/kaggle/input` (any sub-folder) or the working directory, and otherwise download the UCI files from `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/`.

## Coding map (UCI -> distributed file)
| Variable | UCI | Distributed |
|---|---|---|
| sex | 1 / 0 | M / F |
| cp | 1, 2, 3, 4 | TA, ATA, NAP, ASY |
| restecg | 0, 1, 2 | Normal, ST, LVH |
| exang | 1 / 0 | Y / N |
| slope | 1, 2, 3 | Up, Flat, Down |
| num | 0 / 1-4 | HeartDisease 0 / 1 |
| missing | `?` | (no missing values) |

## Scripts (run in order; each is self-contained and can be pasted into one notebook cell)
| Script | Purpose | Output |
|---|---|---|
| `00_feature_tiers_distributed.ipynb` | Feature-tier analysis on the file as distributed (Core-6/8/9/Extended-11, six classifiers, robustness, calibration) | `HeartSense_V3_Results_Bundle.zip` |
| `01_leave_one_cohort_out.py` | Record linkage + leave-one-cohort-out validation | `HeartSense_LOCO_Results.zip` |
| `02_provenance_sensitivity.py` | Value provenance, ST_Slope x outcome, distributed vs provenance-corrected (random CV + LOCO) | `HeartSense_STSlope_Check.zip` |
| `03_export_provenance_flags.py` | Record-level provenance flags (Supplementary File S1, v1) | `HeartSense_Provenance.zip` |
| `04_revision_analyses.py` | Outcome-blinded linkage, ambiguous-match and zero-coding sensitivity, per-predictor audit, ST_Slope-only demonstration, Oldpeak/ST_Slope ablation, repeated CV, calibration, LOCO with CIs, Supplementary File S1 v2, fold IDs, package versions | `HeartSense_Revision.zip` |

## Record linkage
Cost of a (distributed, source) pair = number of disagreeing variables. A source `?` is compatible with any value; for Cholesterol and RestingBP a distributed 0 is compatible with source `?` or 0. Primary linkage includes the outcome (penalty 100); the outcome-blinded linkage uses the 11 predictors only and uses the outcome for validation. Optimal one-to-one assignment: `scipy.optimize.linear_sum_assignment`.

## Provenance definitions
- **source_recorded** – value present in the UCI processed file.
- **completed** – `?` in the UCI file, value present in `heart.csv`.
- **source_zero_coded** – 0 in the UCI file for Cholesterol / RestingBP (treated as not recorded in the primary analysis; sensitivity analysis treats it as recorded).

## Reproducibility
Random seed 42 (repeated CV seeds 1000-1009). Fold assignments are written to `fold_ids_seed42.csv`; package versions to `environment_versions.json`.

## Released data (`data/`)
- `S1_provenance.csv` – for each of the 918 records: linked UCI cohort, file and row; linkage diagnostics (primary and outcome-blinded); and, for every predictor, the source value, the distributed value and its provenance status.
- `S1_data_dictionary.csv` – description of every column.
- `fold_ids_seed42.csv` – cross-validation fold of each record (seed 42).

The source data themselves are not redistributed here; obtain `heart.csv` from Kaggle and the UCI files from the UCI repository (see above).

## Quick start
```bash
pip install -r requirements.txt
# place heart.csv and the four processed.*.data files in the working directory (or /kaggle/input)
python scripts/04_revision_analyses.py      # main audit and re-analysis -> HeartSense_Revision.zip
```
Exact package versions used for the reported results are listed in `environment_versions.json`.

## Citation
If you use this code or the provenance file, please cite the accompanying article (citation to be added on publication).

## Licence
Code: MIT (see `LICENSE`). Derived provenance data in `data/`: CC BY 4.0, consistent with the UCI Heart Disease dataset licence.
