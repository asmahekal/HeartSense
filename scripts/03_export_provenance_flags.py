# ======================================================================
# HeartSense — STANDALONE: Provenance flags export (no model training, ~1 minute)
# ======================================================================
# Runs on its own (no other cell needed). Runs as a script (python <file>) or pasted into a single notebook cell.
# Kaggle: attach dataset "fedesoriano/heart-failure-prediction" and set
# Settings -> Internet -> ON (to download the 4 original UCI files).
# Output: HeartSense_Provenance.zip  
# ======================================================================

import os, sys, json, random, shutil, warnings, subprocess, importlib
from pathlib import Path

warnings.filterwarnings("ignore")

REQ = {
    "numpy":"numpy",
    "pandas":"pandas",
    "sklearn":"scikit-learn",
    "scipy":"scipy",
    "matplotlib":"matplotlib",
    "catboost":"catboost",
    "lightgbm":"lightgbm",
    "xgboost":"xgboost",
    "statsmodels":"statsmodels",
}
missing=[]
for m,p in REQ.items():
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(p)
if missing:
    print("Installing:", missing)
    subprocess.check_call([sys.executable,"-m","pip","install","-q"]+missing)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve

# -----------------------
# 1) Configuration
# -----------------------
SEED=42
OUTER_FOLDS=5
BOOTSTRAP_REPS=2000
ROBUST_REPEATS=5

np.random.seed(SEED)
random.seed(SEED)

TARGET="HeartDisease"

FEATURE_TIERS = {
    "Core_6": [
        "Age","Sex","ChestPainType","RestingBP","MaxHR","ExerciseAngina"
    ],
    "Core_8": [
        "Age","Sex","ChestPainType","RestingBP","MaxHR","ExerciseAngina",
        "Cholesterol","FastingBS"
    ],
    "Core_9": [
        "Age","Sex","ChestPainType","RestingBP","MaxHR","ExerciseAngina",
        "Cholesterol","FastingBS","RestingECG"
    ],
    "Extended_11": [
        "Age","Sex","ChestPainType","RestingBP","MaxHR","ExerciseAngina",
        "Cholesterol","FastingBS","RestingECG","Oldpeak","ST_Slope"
    ]
}

NUMERIC_ALL = ["Age","RestingBP","MaxHR","Cholesterol","Oldpeak"]
CATEGORICAL_ALL = [
    "Sex","ChestPainType","ExerciseAngina","FastingBS","RestingECG","ST_Slope"
]

WORKDIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

# -----------------------
# 2) Dataset discovery
# -----------------------
def locate_heart_csv():
    roots=[Path.cwd()]
    if Path("/kaggle/input").exists():
        roots.insert(0,Path("/kaggle/input"))

    expected=set(FEATURE_TIERS["Extended_11"]+[TARGET])

    for root in roots:
        for p in root.rglob("heart.csv"):
            try:
                sample=pd.read_csv(p,nrows=5)
                if expected.issubset(sample.columns):
                    full=pd.read_csv(p)
                    if len(full)==918:
                        return p
            except Exception:
                pass

    # KaggleHub fallback
    try:
        import kagglehub
    except Exception:
        try:
            subprocess.check_call([sys.executable,"-m","pip","install","-q","kagglehub"])
            import kagglehub
        except Exception:
            kagglehub=None

    if kagglehub is not None:
        folder=Path(kagglehub.dataset_download("fedesoriano/heart-failure-prediction"))
        for p in folder.rglob("heart.csv"):
            sample=pd.read_csv(p,nrows=5)
            if expected.issubset(sample.columns):
                return p

    raise FileNotFoundError(
        "heart.csv not found. Attach the Kaggle Heart Failure Prediction Dataset (fedesoriano)."
    )

DATA_PATH=locate_heart_csv()
raw=pd.read_csv(DATA_PATH)
df=raw.drop_duplicates().reset_index(drop=True)

if TARGET not in df.columns:
    raise ValueError("Target column HeartDisease not found.")

y=df[TARGET].astype(int).copy()

DATA_PATH = locate_heart_csv()
df = pd.read_csv(DATA_PATH).drop_duplicates().reset_index(drop=True)
y = df[TARGET].astype(int).copy()
print("Loaded", DATA_PATH, "rows:", len(df))

# -----------------------
# 3) Helpers
# -----------------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore",sparse=False)

def build_preprocessor(features):
    num=[c for c in features if c in NUMERIC_ALL]
    cat=[c for c in features if c in CATEGORICAL_ALL]

    num_pipe=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipe=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",make_ohe())
    ])

    return ColumnTransformer([
        ("num",num_pipe,num),
        ("cat",cat_pipe,cat)
    ])

def make_models(features):
    prep=build_preprocessor(features)
    return {
        "Logistic_Regression":Pipeline([
            ("prep",clone(prep)),
            ("model",LogisticRegression(max_iter=2500,random_state=SEED))
        ]),
        "Gradient_Boosting":Pipeline([
            ("prep",clone(prep)),
            ("model",GradientBoostingClassifier(
                n_estimators=250,learning_rate=0.03,max_depth=2,random_state=SEED
            ))
        ]),
        "Random_Forest":Pipeline([
            ("prep",clone(prep)),
            ("model",RandomForestClassifier(
                n_estimators=600,min_samples_leaf=2,max_features="sqrt",
                random_state=SEED,n_jobs=-1
            ))
        ]),
        "XGBoost":Pipeline([
            ("prep",clone(prep)),
            ("model",XGBClassifier(
                n_estimators=450,max_depth=3,learning_rate=0.03,
                subsample=0.9,colsample_bytree=0.9,
                reg_lambda=1.0,eval_metric="logloss",
                random_state=SEED,n_jobs=-1
            ))
        ]),
        "LightGBM":Pipeline([
            ("prep",clone(prep)),
            ("model",LGBMClassifier(
                n_estimators=500,learning_rate=0.025,num_leaves=15,
                min_child_samples=20,reg_lambda=1.0,
                random_state=SEED,n_jobs=-1,verbosity=-1
            ))
        ]),
    }

CATBOOST_PARAMS=dict(
    iterations=700,
    depth=6,
    learning_rate=0.035,
    l2_leaf_reg=1.0,
    random_strength=0.05,
    bagging_temperature=1.0,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=0,
    random_seed=SEED,
    allow_writing_files=False
)

def cb_prep(train_df, apply_df, features):
    num=[c for c in features if c in NUMERIC_ALL]
    cat=[c for c in features if c in CATEGORICAL_ALL]
    tr=train_df.copy()
    ap=apply_df.copy()

    for c in num:
        med=pd.to_numeric(tr[c],errors="coerce").median()
        tr[c]=pd.to_numeric(tr[c],errors="coerce").fillna(med)
        ap[c]=pd.to_numeric(ap[c],errors="coerce").fillna(med)

    for c in cat:
        mode=tr[c].astype("string").dropna().mode()
        fill=str(mode.iloc[0]) if len(mode) else "Missing"
        tr[c]=tr[c].astype("string").fillna(fill).astype(str)
        ap[c]=ap[c].astype("string").fillna(fill).astype(str)

    return tr,ap,cat

def metric_dict(ytrue,prob,threshold=0.5):
    prob=np.asarray(prob,dtype=float)
    pred=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(ytrue,pred,labels=[0,1]).ravel()
    return {
        "Accuracy":accuracy_score(ytrue,pred),
        "Precision_PPV":precision_score(ytrue,pred,zero_division=0),
        "Sensitivity_Recall":recall_score(ytrue,pred,zero_division=0),
        "Specificity":tn/(tn+fp) if tn+fp else np.nan,
        "NPV":tn/(tn+fn) if tn+fn else np.nan,
        "F1":f1_score(ytrue,pred,zero_division=0),
        "Balanced_Accuracy":balanced_accuracy_score(ytrue,pred),
        "MCC":matthews_corrcoef(ytrue,pred),
        "ROC_AUC":roc_auc_score(ytrue,prob),
        "PR_AUC":average_precision_score(ytrue,prob),
        "Brier":brier_score_loss(ytrue,prob)
    }

def inject_missingness(frame, fraction, rng):
    out=frame.copy()
    mask=rng.random(out.shape)<fraction
    out=out.mask(mask)
    return out

def inject_numeric_noise(frame, train_frame, features, rng, scale_fraction=0.05):
    out=frame.copy()
    for c in [x for x in features if x in NUMERIC_ALL]:
        sd=pd.to_numeric(train_frame[c],errors="coerce").std()
        if np.isfinite(sd) and sd>0:
            noise=rng.normal(0,scale_fraction*sd,size=len(out))
            out[c]=pd.to_numeric(out[c],errors="coerce")+noise
    return out

def bootstrap_ci(ytrue,prob,reps=BOOTSTRAP_REPS,seed=SEED):
    ytrue=np.asarray(ytrue)
    prob=np.asarray(prob)
    rng=np.random.default_rng(seed)
    keys=list(metric_dict(ytrue,prob).keys())
    store={k:[] for k in keys}
    for _ in range(reps):
        idx=rng.integers(0,len(ytrue),len(ytrue))
        if len(np.unique(ytrue[idx]))<2:
            continue
        m=metric_dict(ytrue[idx],prob[idx])
        for k,v in m.items():
            if np.isfinite(v):
                store[k].append(v)
    point=metric_dict(ytrue,prob)
    rows=[]
    for k in keys:
        vals=np.asarray(store[k])
        rows.append({
            "Metric":k,
            "Estimate":point[k],
            "CI95_Low":np.percentile(vals,2.5),
            "CI95_High":np.percentile(vals,97.5)
        })
    return pd.DataFrame(rows)


import io, urllib.request
from scipy.optimize import linear_sum_assignment

LOCO_ROOT = WORKDIR / "HeartSense_Provenance"
if LOCO_ROOT.exists():
    shutil.rmtree(LOCO_ROOT)
LOCO_ROOT.mkdir(parents=True)

# ----------------------------------------------------------------------
# 1) Download the original source cohorts
# ----------------------------------------------------------------------
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
SOURCES = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "LongBeachVA": "processed.va.data",
}
UCI_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
            "exang", "oldpeak", "slope", "ca", "thal", "num"]

def load_source(fname):
    local = [p for p in Path("/kaggle/input").rglob(fname)] if Path("/kaggle/input").exists() else []
    if local:
        raw_txt = Path(local[0]).read_text()
    else:
        with urllib.request.urlopen(UCI_BASE + fname, timeout=60) as r:
            raw_txt = r.read().decode("utf-8", errors="ignore")
    d = pd.read_csv(io.StringIO(raw_txt), header=None, names=UCI_COLS, na_values="?")
    return d

src_frames = []
for cohort, fname in SOURCES.items():
    d = load_source(fname)
    d["Cohort"] = cohort
    d["SourceRow"] = np.arange(len(d))
    src_frames.append(d)
    print(f"{cohort}: {len(d)} source records")
src = pd.concat(src_frames, ignore_index=True)

# Map UCI coding -> Kaggle coding
src_k = pd.DataFrame({
    "Age": src["age"],
    "Sex": src["sex"].map({1: "M", 0: "F"}),
    "ChestPainType": src["cp"].map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
    "RestingBP": src["trestbps"],
    "Cholesterol": src["chol"],
    "FastingBS": src["fbs"],
    "RestingECG": src["restecg"].map({0: "Normal", 1: "ST", 2: "LVH"}),
    "MaxHR": src["thalach"],
    "ExerciseAngina": src["exang"].map({1: "Y", 0: "N"}),
    "Oldpeak": src["oldpeak"],
    "ST_Slope": src["slope"].map({1: "Up", 2: "Flat", 3: "Down"}),
    "HeartDisease": (src["num"] > 0).astype(int),
})

# ----------------------------------------------------------------------
# 2) One-to-one record linkage (Kaggle row -> source record)
# ----------------------------------------------------------------------
LINK_COLS = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
             "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope", "HeartDisease"]

K = df[LINK_COLS].reset_index(drop=True)
cost = np.zeros((len(K), len(src_k)), dtype=float)

for c in LINK_COLS:
    kv = K[c].to_numpy()
    sv = src_k[c].to_numpy()
    s_missing = pd.isna(sv)
    if c in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS", "HeartDisease"]:
        kf = pd.to_numeric(pd.Series(kv), errors="coerce").to_numpy(dtype=float)
        sf = pd.to_numeric(pd.Series(sv), errors="coerce").to_numpy(dtype=float)
        mism = np.abs(kf[:, None] - sf[None, :]) > 1e-6
        # Kaggle stores unknown cholesterol / BP as 0 -> treat 0 as compatible with unknown
        if c in ["Cholesterol", "RestingBP"]:
            mism &= ~((kf[:, None] == 0) & (np.isnan(sf)[None, :] | (sf[None, :] == 0)))
    else:
        mism = kv.astype(str)[:, None] != sv.astype(str)[None, :]
    mism &= ~s_missing[None, :]          # unknown source value is not a mismatch
    w = 100.0 if c == "HeartDisease" else 1.0
    cost += w * mism

row_ind, col_ind = linear_sum_assignment(cost)
link = pd.DataFrame({
    "KaggleRow": row_ind,
    "Cohort": src.loc[col_ind, "Cohort"].to_numpy(),
    "SourceRow": src.loc[col_ind, "SourceRow"].to_numpy(),
    "Mismatches": cost[row_ind, col_ind],
})
# ambiguity: how many source records were equally good for this row?
best = cost.min(axis=1)
link["BestPossible"] = best[row_ind]
link["EquallyGoodCandidates"] = (cost[row_ind] == best[row_ind][:, None]).sum(axis=1)
cands_cohorts = []
for r in row_ind:
    cc = src.loc[np.where(cost[r] == best[r])[0], "Cohort"].unique()
    cands_cohorts.append("|".join(sorted(cc)))
link["CandidateCohorts"] = cands_cohorts
link = link.sort_values("KaggleRow").reset_index(drop=True)
link.to_csv(LOCO_ROOT / "01_record_linkage_full.csv", index=False)

audit = pd.DataFrame([
    ["kaggle_rows", len(K)],
    ["source_records_total", len(src_k)],
    ["rows_linked_exactly_(0_mismatch)", int((link.Mismatches == 0).sum())],
    ["rows_linked_with_1_mismatch", int((link.Mismatches == 1).sum())],
    ["rows_linked_with_2plus_mismatch", int((link.Mismatches >= 2).sum())],
    ["rows_with_outcome_mismatch", int((link.Mismatches >= 100).sum())],
    ["rows_with_candidates_in_more_than_one_cohort", int((link.CandidateCohorts.str.contains(r"\|")).sum())],
] + [[f"n_{c}", int((link.Cohort == c).sum())] for c in SOURCES],
    columns=["Item", "Value"])
audit.to_csv(LOCO_ROOT / "01_record_linkage_audit.csv", index=False)
print(audit.to_string(index=False))

cohort = link["Cohort"].to_numpy()
pd.crosstab(cohort, y.values, rownames=["Cohort"], colnames=["HeartDisease"]).to_csv(
    LOCO_ROOT / "02_cohort_outcome_table.csv")

def safe_bootstrap_ci(ytrue, prob, reps=1000, seed=SEED):
    """Like bootstrap_ci, but tolerant of metrics that are undefined in a
    resample (e.g. NPV when no negatives are predicted in Switzerland)."""
    ytrue = np.asarray(ytrue); prob = np.asarray(prob)
    rng_ = np.random.default_rng(seed)
    point = metric_dict(ytrue, prob)
    store = {k: [] for k in point}
    for _ in range(reps):
        idx = rng_.integers(0, len(ytrue), len(ytrue))
        if len(np.unique(ytrue[idx])) < 2:
            continue
        for k, v in metric_dict(ytrue[idx], prob[idx]).items():
            if np.isfinite(v):
                store[k].append(v)
    return pd.DataFrame([{"Metric": k, "Estimate": point[k],
                          "CI95_Low": np.percentile(store[k], 2.5) if store[k] else np.nan,
                          "CI95_High": np.percentile(store[k], 97.5) if store[k] else np.nan}
                         for k in point])


# ----------------------------------------------------------------------
# 4) Which Kaggle values were NOT recorded in the original source files?
# ----------------------------------------------------------------------
SRC_MAP = {"RestingBP": "trestbps", "Cholesterol": "chol", "FastingBS": "fbs",
           "RestingECG": "restecg", "MaxHR": "thalach", "ExerciseAngina": "exang",
           "Oldpeak": "oldpeak", "ST_Slope": "slope"}
src_idx = src.set_index(["Cohort", "SourceRow"])
linked_src = src_idx.loc[list(zip(link.Cohort, link.SourceRow))].reset_index(drop=True)
src_missing = pd.DataFrame({k: linked_src[v].isna().to_numpy() for k, v in SRC_MAP.items()})
# Cholesterol = 0 is also "not measured" in the source
src_missing["Cholesterol"] |= (linked_src["chol"].fillna(0).to_numpy() == 0)

miss_tab = src_missing.groupby(cohort).agg(["sum", "mean"])
miss_tab.to_csv(LOCO_ROOT / "02_source_missingness_by_cohort.csv")
print("\nNot recorded in original source (count, fraction) by cohort:")
print(miss_tab.round(3).to_string())


# ----------------------------------------------------------------------
# 5) Export per-record provenance file (supplementary data for the paper)
# ----------------------------------------------------------------------
prov = pd.DataFrame({
    "RecordID": np.arange(1, len(df) + 1),          # row order of heart.csv (1-based)
    "SourceCohort": link["Cohort"].to_numpy(),
    "SourceFile": link["Cohort"].map({k: v for k, v in SOURCES.items()}).to_numpy(),
    "SourceRow": link["SourceRow"].to_numpy() + 1,  # 1-based line in the UCI file
    "LinkageDiscrepancies": link["Mismatches"].astype(int).to_numpy(),
    "AmbiguousCohort": link["CandidateCohorts"].str.contains(r"\|").astype(int).to_numpy(),
})
for col in SRC_MAP:
    prov[f"{col}_completed"] = src_missing[col].astype(int).to_numpy()
prov["Any_completed"] = src_missing.any(axis=1).astype(int).to_numpy()
prov["HeartDisease"] = y.values
prov.to_csv(LOCO_ROOT / "HeartSense_provenance_flags.csv", index=False)

dd = pd.DataFrame([
    ["RecordID", "Row number of the record in heart.csv (1 = first data row)"],
    ["SourceCohort", "Original cohort: Cleveland, Hungarian, Switzerland, LongBeachVA"],
    ["SourceFile", "UCI processed file the record was linked to"],
    ["SourceRow", "Line number of the matched record in that UCI file (1-based)"],
    ["LinkageDiscrepancies", "Number of predictor/outcome disagreements with the matched source record (0 = exact)"],
    ["AmbiguousCohort", "1 if an equally good match existed in another cohort"],
] + [[f"{c}_completed", f"1 if {c} was not recorded in the source file (missing, or 0 for Cholesterol) and was completed in heart.csv"] for c in SRC_MAP]
  + [["Any_completed", "1 if any predictor of the record was completed"],
     ["HeartDisease", "Outcome as in heart.csv"]], columns=["Column", "Description"])
dd.to_csv(LOCO_ROOT / "HeartSense_provenance_data_dictionary.csv", index=False)

print("\nRecords with any completed value:", int(prov.Any_completed.sum()))
print(prov.groupby("SourceCohort")[[c for c in prov.columns if c.endswith("_completed")]].sum().to_string())
zp = shutil.make_archive(str(WORKDIR / "HeartSense_Provenance"), "zip", root_dir=LOCO_ROOT)
print("\nDOWNLOAD THIS ZIP:", zp)
