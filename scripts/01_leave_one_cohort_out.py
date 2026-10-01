# ======================================================================
# HeartSense — STANDALONE: Leave-one-cohort-out (internal-external)
# ======================================================================
# Runs on its own (no other cell needed). Runs as a script (python <file>) or pasted into a single notebook cell.
# Kaggle: attach dataset "fedesoriano/heart-failure-prediction" and set
# Settings -> Internet -> ON (to download the 4 original UCI files).
# Output: HeartSense_LOCO_Results.zip  
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


# ----------------------------------------------------------------------
# Random 5-fold CV reference (same folds/seed as V3) for Core-6 and
# Extended-11 only — used to compare random-fold vs cohort-held-out AUC.
# ----------------------------------------------------------------------
MODEL_NAMES = ["Logistic_Regression", "Gradient_Boosting", "Random_Forest",
               "XGBoost", "LightGBM", "CatBoost"]
oof = {t: {m: np.full(len(df), np.nan) for m in MODEL_NAMES} for t in ["Core_6", "Extended_11"]}
outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, te_idx) in enumerate(outer.split(df, y), start=1):
    for tier in oof:
        feats = FEATURE_TIERS[tier]
        Xtr, Xte = df.iloc[tr_idx][feats], df.iloc[te_idx][feats]
        for name, model in make_models(feats).items():
            model.fit(Xtr, y.iloc[tr_idx])
            oof[tier][name][te_idx] = model.predict_proba(Xte)[:, 1]
        a, b, cc = cb_prep(Xtr, Xte, feats)
        cbm = CatBoostClassifier(**CATBOOST_PARAMS); cbm.fit(a, y.iloc[tr_idx], cat_features=cc)
        oof[tier]["CatBoost"][te_idx] = cbm.predict_proba(b)[:, 1]
    print(f"Reference random-fold CV: fold {fold}/{OUTER_FOLDS} done")

import io, urllib.request
from scipy.optimize import linear_sum_assignment

LOCO_ROOT = WORKDIR / "HeartSense_LOCO_Results"
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
# 3) Leave-one-cohort-out
# ----------------------------------------------------------------------
LOCO_TIERS = ["Core_6", "Extended_11"]
loco_rows, loco_ci_rows = [], []
loco_pred = pd.DataFrame({"Actual": y.values, "Cohort": cohort})

for tier in LOCO_TIERS:
    feats = FEATURE_TIERS[tier]
    for held in SOURCES:
        te = np.where(cohort == held)[0]
        tr = np.where(cohort != held)[0]
        Xtr, Xte = df.iloc[tr][feats].copy(), df.iloc[te][feats].copy()
        ytr, yte = y.iloc[tr], y.iloc[te]
        probs = {}
        for name, model in make_models(feats).items():
            model.fit(Xtr, ytr)
            probs[name] = model.predict_proba(Xte)[:, 1]
        Xtr_cb, Xte_cb, catcols = cb_prep(Xtr, Xte, feats)
        cb = CatBoostClassifier(**CATBOOST_PARAMS)
        cb.fit(Xtr_cb, ytr, cat_features=catcols)
        probs["CatBoost"] = cb.predict_proba(Xte_cb)[:, 1]

        for name, p in probs.items():
            col = f"{tier}__{name}__Probability"
            if col not in loco_pred:
                loco_pred[col] = np.nan
            loco_pred.loc[te, col] = p
            m = metric_dict(yte.values, p)
            loco_rows.append({"FeatureTier": tier, "HeldOutCohort": held, "Model": name,
                              "n_test": len(te), "prevalence_test": float(yte.mean()),
                              "n_train": len(tr), **m})
            ci = safe_bootstrap_ci(yte.values, p, reps=1000)
            ci.insert(0, "Model", name); ci.insert(0, "HeldOutCohort", held); ci.insert(0, "FeatureTier", tier)
            loco_ci_rows.append(ci)
        print(f"LOCO done: {tier} | held-out {held} (n={len(te)})")

loco = pd.DataFrame(loco_rows)
loco.to_csv(LOCO_ROOT / "03_LOCO_metrics.csv", index=False)
pd.concat(loco_ci_rows, ignore_index=True).to_csv(LOCO_ROOT / "03_LOCO_metrics_95CI.csv", index=False)
loco_pred.to_csv(LOCO_ROOT / "04_LOCO_predictions.csv", index=False)

# Weighted / median summaries per tier x model
summ = (loco.assign(w=loco.n_test)
        .groupby(["FeatureTier", "Model"])
        .apply(lambda g: pd.Series({
            "Weighted_ROC_AUC": np.average(g.ROC_AUC, weights=g.w),
            "Median_ROC_AUC": g.ROC_AUC.median(),
            "Min_ROC_AUC": g.ROC_AUC.min(),
            "Max_ROC_AUC": g.ROC_AUC.max(),
            "Weighted_Accuracy": np.average(g.Accuracy, weights=g.w),
            "Weighted_Brier": np.average(g.Brier, weights=g.w)}))
        .reset_index())
summ.to_csv(LOCO_ROOT / "05_LOCO_summary.csv", index=False)

# Comparison with random-fold CV (pooled OOF from the main cell)
cmp_rows = []
for tier in LOCO_TIERS:
    for name in MODEL_NAMES:
        cmp_rows.append({"FeatureTier": tier, "Model": name,
                         "RandomFoldCV_ROC_AUC": roc_auc_score(y, oof[tier][name]),
                         "LOCO_Weighted_ROC_AUC": summ[(summ.FeatureTier == tier) & (summ.Model == name)].Weighted_ROC_AUC.iloc[0]})
cmp = pd.DataFrame(cmp_rows)
cmp["Drop"] = cmp.RandomFoldCV_ROC_AUC - cmp.LOCO_Weighted_ROC_AUC
cmp.to_csv(LOCO_ROOT / "06_RandomCV_vs_LOCO.csv", index=False)

# Quick figure (final figures will be redrawn in PowerPoint)
fig, ax = plt.subplots(figsize=(8, 5))
sub = loco[(loco.FeatureTier == "Extended_11")]
for name in MODEL_NAMES:
    s_ = sub[sub.Model == name].set_index("HeldOutCohort").loc[list(SOURCES)]
    ax.plot(list(SOURCES), s_.ROC_AUC, marker="o", label=name)
ax.set_ylabel("ROC-AUC (held-out cohort)")
ax.set_title("Leave-one-cohort-out — Extended-11")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(LOCO_ROOT / "07_LOCO_Extended11.png", dpi=200)
plt.close(fig)

zip_loco = shutil.make_archive(str(WORKDIR / "HeartSense_LOCO_Results"), "zip", root_dir=LOCO_ROOT)
print("\n" + "=" * 70)
print(summ.round(4).to_string(index=False))
print("\nDOWNLOAD THIS ZIP:", zip_loco)
print("=" * 70)
