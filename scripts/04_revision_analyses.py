# ======================================================================
# HeartSense — STANDALONE: Major-revision analyses (approx. 30-60 min on Kaggle CPU)
# ======================================================================
# Runs on its own (no other cell needed). Runs as a script (python <file>) or pasted into a single notebook cell.
# Kaggle: attach dataset "fedesoriano/heart-failure-prediction" and set
# Settings -> Internet -> ON (to download the 4 original UCI files).
# Output: HeartSense_Revision.zip  
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

OUT = WORKDIR / "HeartSense_Revision"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

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


# ======================================================================
# REVISION ANALYSES
# ======================================================================
import time, platform, scipy
from scipy import stats
T0 = time.time()
def tick(msg): print(f"[{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)

MODEL_NAMES = ["Logistic_Regression", "Gradient_Boosting", "Random_Forest",
               "XGBoost", "LightGBM", "CatBoost"]
PRED = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
        "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
NUMERIC_PRED = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS"]
SRC_COL = {"Age": "age", "Sex": "sex", "ChestPainType": "cp", "RestingBP": "trestbps",
           "Cholesterol": "chol", "FastingBS": "fbs", "RestingECG": "restecg",
           "MaxHR": "thalach", "ExerciseAngina": "exang", "Oldpeak": "oldpeak", "ST_Slope": "slope"}
COMPLETABLE = ["RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR",
               "ExerciseAngina", "Oldpeak", "ST_Slope"]
y_np = y.to_numpy()

# ----------------------------------------------------------------------
# R1. Record linkage: original (outcome used) vs outcome-blinded
# ----------------------------------------------------------------------
def build_cost(cols, outcome_weight=100.0):
    K = df[cols].reset_index(drop=True)
    cost = np.zeros((len(K), len(src_k)), dtype=float)
    for c in cols:
        kv = K[c].to_numpy(); sv = src_k[c].to_numpy(); s_missing = pd.isna(sv)
        if c in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "FastingBS", "HeartDisease"]:
            kf = pd.to_numeric(pd.Series(kv), errors="coerce").to_numpy(dtype=float)
            sf = pd.to_numeric(pd.Series(sv), errors="coerce").to_numpy(dtype=float)
            mism = np.abs(kf[:, None] - sf[None, :]) > 1e-6
            if c in ["Cholesterol", "RestingBP"]:
                mism &= ~((kf[:, None] == 0) & (np.isnan(sf)[None, :] | (sf[None, :] == 0)))
        else:
            mism = kv.astype(str)[:, None] != sv.astype(str)[None, :]
        mism &= ~s_missing[None, :]
        cost += (outcome_weight if c == "HeartDisease" else 1.0) * mism
    return cost

def run_linkage(cols):
    cost = build_cost(cols)
    r, c = linear_sum_assignment(cost)
    order = np.argsort(r); r, c = r[order], c[order]
    best = cost.min(axis=1)
    eq = cost == best[:, None]
    n_eq = eq.sum(axis=1)
    cand_coh = ["|".join(sorted(src.loc[np.where(eq[i])[0], "Cohort"].unique())) for i in range(len(r))]
    L = pd.DataFrame({"KaggleRow": r, "SrcIndex": c,
                      "Cohort": src.loc[c, "Cohort"].to_numpy(),
                      "SourceRow": src.loc[c, "SourceRow"].to_numpy(),
                      "Cost": cost[r, c], "BestPossible": best,
                      "EquallyGoodCandidates": n_eq, "CandidateCohorts": cand_coh})
    return L, cost, eq

LINK_FULL = PRED + ["HeartDisease"]
link_o, cost_o, eq_o = run_linkage(LINK_FULL)        # original
link_b, cost_b, eq_b = run_linkage(PRED)             # outcome-blinded
tick("linkage done")

def slope_missing_for(srcidx):
    return src_k.loc[srcidx, "ST_Slope"].isna().to_numpy()

def linkage_summary(L, eq, label):
    srcy = src_k.loc[L.SrcIndex, "HeartDisease"].to_numpy()
    comp = slope_missing_for(L.SrcIndex.to_numpy())
    # bounds over all equally-good candidates
    cmin = np.array([src_k.loc[np.where(eq[i])[0], "ST_Slope"].isna().all() for i in range(len(L))])
    cmax = np.array([src_k.loc[np.where(eq[i])[0], "ST_Slope"].isna().any() for i in range(len(L))])
    out = {"Linkage": label,
           "rows": len(L),
           "zero_cost_matches": int((L.Cost == 0).sum()),
           "nonzero_cost_matches": int((L.Cost > 0).sum()),
           "unique_best_candidate": int((L.EquallyGoodCandidates == 1).sum()),
           "ambiguous_best_candidate": int((L.EquallyGoodCandidates > 1).sum()),
           "ambiguous_across_cohorts": int(L.CandidateCohorts.str.contains(r"\|").sum()),
           "outcome_disagreement_with_source": int((srcy != y_np).sum()),
           "ST_Slope_completed": int(comp.sum()),
           "ST_Slope_completed_lower_bound": int(cmin.sum()),
           "ST_Slope_completed_upper_bound": int(cmax.sum())}
    for st in ["Up", "Flat", "Down"]:
        m = comp & (df.ST_Slope.to_numpy() == st)
        out[f"completed_{st}_n"] = int(m.sum()); out[f"completed_{st}_HD"] = int(y_np[m].sum())
        m2 = (~comp) & (df.ST_Slope.to_numpy() == st)
        out[f"recorded_{st}_n"] = int(m2.sum()); out[f"recorded_{st}_HD"] = int(y_np[m2].sum())
    for co in SOURCES:
        out[f"n_{co}"] = int((L.Cohort == co).sum())
    return out

summ = [linkage_summary(link_o, eq_o, "original_outcome_used"),
        linkage_summary(link_b, eq_b, "outcome_blinded")]
agree = pd.DataFrame([{
    "same_source_record": int((link_o.SrcIndex.to_numpy() == link_b.SrcIndex.to_numpy()).sum()),
    "same_cohort": int((link_o.Cohort.to_numpy() == link_b.Cohort.to_numpy()).sum()),
    "same_ST_Slope_completion_status": int((slope_missing_for(link_o.SrcIndex.to_numpy()) ==
                                            slope_missing_for(link_b.SrcIndex.to_numpy())).sum())}])
pd.DataFrame(summ).T.to_csv(OUT / "R1_linkage_original_vs_blinded.csv", header=["original", "blinded"])
agree.to_csv(OUT / "R1_linkage_agreement.csv", index=False)
link_o.to_csv(OUT / "R1_linkage_original_full.csv", index=False)
link_b.to_csv(OUT / "R1_linkage_blinded_full.csv", index=False)
print(pd.DataFrame(summ).T.to_string()); print(agree.to_string(index=False))

cohort = link_o.Cohort.to_numpy()
SRCIDX = link_o.SrcIndex.to_numpy()
LS = src.loc[SRCIDX].reset_index(drop=True)      # linked raw source rows

# ----------------------------------------------------------------------
# R2. Provenance flags, two zero rules
# ----------------------------------------------------------------------
def provenance(zero_as_missing=True):
    m = pd.DataFrame({p: LS[SRC_COL[p]].isna().to_numpy() for p in COMPLETABLE})
    if zero_as_missing:
        m["Cholesterol"] |= (LS["chol"].fillna(-1).to_numpy() == 0)
        m["RestingBP"] |= (LS["trestbps"].fillna(-1).to_numpy() == 0)
    return m
prov_A = provenance(True)    # zero-as-missing (primary)
prov_B = provenance(False)   # zero-as-recorded
zero_tab = pd.DataFrame({
    "completed_zero_as_missing": prov_A.groupby(cohort).sum().stack(),
    "completed_zero_as_recorded": prov_B.groupby(cohort).sum().stack()})
zero_tab.to_csv(OUT / "R2_provenance_zero_rules.csv")
pd.DataFrame([{"distributed_Cholesterol_eq_0": int((df.Cholesterol == 0).sum()),
               "distributed_RestingBP_eq_0": int((df.RestingBP == 0).sum()),
               "source_chol_eq_0": int((LS.chol == 0).sum()),
               "source_chol_missing": int(LS.chol.isna().sum()),
               "source_trestbps_eq_0": int((LS.trestbps == 0).sum()),
               "source_trestbps_missing": int(LS.trestbps.isna().sum())}]).to_csv(OUT / "R2_zero_counts.csv", index=False)

# ----------------------------------------------------------------------
# R3. Per-predictor audit: recorded vs completed x outcome
# ----------------------------------------------------------------------
def univ_auc(x, yy):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 5 or len(np.unique(yy[ok])) < 2: return np.nan
    a = roc_auc_score(yy[ok], x[ok]); return max(a, 1 - a)

rows, ext = [], []
for p in COMPLETABLE:
    comp = prov_A[p].to_numpy()
    for status, mask in [("recorded", ~comp), ("completed", comp)]:
        if mask.sum() == 0: continue
        v = df.loc[mask, p]; yy = y_np[mask]
        r = {"Predictor": p, "Status": status, "n": int(mask.sum()),
             "n_HD": int(yy.sum()), "pct_HD": yy.mean(),
             "n_distinct_values": int(v.nunique()),
             "top_values": "; ".join(f"{k}:{n}" for k, n in v.value_counts().head(5).items())}
        if p in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]:
            ct = pd.crosstab(v.astype(str), yy)
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, pval, _, _ = stats.chi2_contingency(ct, correction=False)
                r["CramersV"] = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
                r["chi2_p"] = pval
                if ct.shape == (2, 2):
                    r["Fisher_p"] = stats.fisher_exact(ct.values)[1]
            # purity: fraction of records whose value is the majority value within its outcome class
            r["pct_HD_by_value"] = "; ".join(f"{k}:{(yy[v.astype(str).to_numpy()==k]).mean():.2f}" for k in ct.index)
        if p in NUMERIC_PRED and p != "FastingBS":
            xv = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
            a, b = xv[yy == 1], xv[yy == 0]
            if len(a) > 1 and len(b) > 1:
                sd = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
                r["SMD_HD_vs_noHD"] = (np.nanmean(a) - np.nanmean(b)) / sd if sd > 0 else np.inf
                r["MannWhitney_p"] = stats.mannwhitneyu(a, b).pvalue
            r["univariable_AUC"] = univ_auc(xv, yy)
        rows.append(r)
    # outcome-stratified completion test (pooled recorded statistics)
    if comp.sum() > 0:
        rec = ~comp
        cat = p in ["RestingECG", "ExerciseAngina", "ST_Slope", "FastingBS"]
        def stat(sel):
            vv = df.loc[sel, p]
            return vv.astype(str).mode().iloc[0] if cat else pd.to_numeric(vv, errors="coerce").median()
        s_all = stat(rec); s1 = stat(rec & (y_np == 1)); s0 = stat(rec & (y_np == 0))
        vc = df.loc[comp, p]; yc = y_np[comp]
        if cat:
            eq_cls = np.where(yc == 1, vc.astype(str) == s1, vc.astype(str) == s0)
            eq_all = (vc.astype(str) == s_all).to_numpy()
        else:
            vn = pd.to_numeric(vc, errors="coerce").to_numpy(dtype=float)
            eq_cls = np.where(yc == 1, np.isclose(vn, s1), np.isclose(vn, s0))
            eq_all = np.isclose(vn, s_all)
        ext.append({"Predictor": p, "n_completed": int(comp.sum()),
                    "recorded_overall_stat": s_all, "recorded_HD_stat": s1, "recorded_noHD_stat": s0,
                    "pct_completed_equal_to_class_stat": float(np.mean(eq_cls)),
                    "pct_completed_equal_to_overall_stat": float(np.mean(eq_all))})
pd.DataFrame(rows).to_csv(OUT / "R3_predictor_audit_recorded_vs_completed.csv", index=False)
pd.DataFrame(ext).to_csv(OUT / "R3_outcome_stratified_completion_test.csv", index=False)

# extended cohort x predictor table
ext2 = []
for co in SOURCES:
    for p in COMPLETABLE:
        mco = cohort == co; comp = prov_A[p].to_numpy() & mco
        v = df.loc[comp, p]
        ext2.append({"Cohort": co, "Predictor": p, "n_cohort": int(mco.sum()),
                     "n_source_recorded": int((mco & ~prov_A[p].to_numpy()).sum()),
                     "n_not_recorded": int(comp.sum()),
                     "filled_values": "; ".join(f"{k}:{n}" for k, n in v.value_counts().head(6).items()),
                     "HD_within_filled": f"{int(y_np[comp].sum())}/{int(comp.sum())}" if comp.sum() else "",
                     "HD_by_filled_value": "; ".join(f"{k}:{int(y_np[comp & (df[p].astype(str).to_numpy()==str(k))].sum())}/{n}"
                                                     for k, n in v.value_counts().head(6).items())})
pd.DataFrame(ext2).to_csv(OUT / "R3_extended_cohort_by_predictor.csv", index=False)
tick("audit tables done")

# ----------------------------------------------------------------------
# Data versions
# ----------------------------------------------------------------------
def masked(cols, prov):
    d = df.copy()
    for p in cols:
        d.loc[prov[p].to_numpy(), p] = np.nan
    return d
D = {"distributed": df,
     "corrected": masked(COMPLETABLE, prov_A),
     "corrected_zero_as_recorded": masked(COMPLETABLE, prov_B),
     "STslope_masked_only": masked(["ST_Slope"], prov_A),
     "Oldpeak_masked_only": masked(["Oldpeak"], prov_A)}

def fit_predict(data, feats, tr, te, models=MODEL_NAMES):
    Xtr, Xte = data.iloc[tr][feats].copy(), data.iloc[te][feats].copy(); ytr = y.iloc[tr]
    out = {}
    mm = make_models(feats)
    for name in models:
        if name == "CatBoost":
            a, b, cc = cb_prep(Xtr, Xte, feats)
            m = CatBoostClassifier(**CATBOOST_PARAMS); m.fit(a, ytr, cat_features=cc)
            out[name] = m.predict_proba(b)[:, 1]
        else:
            m = mm[name]; m.fit(Xtr, ytr); out[name] = m.predict_proba(Xte)[:, 1]
    return out

def cv_oof(data, feats, seed=SEED, models=MODEL_NAMES):
    sk = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    o = {m: np.full(len(df), np.nan) for m in models}
    for tr, te in sk.split(df, y):
        for m, p in fit_predict(data, feats, tr, te, models).items(): o[m][te] = p
    return o

def paired_boot(p1, p0, reps=1000, seed=SEED + 11):
    rng = np.random.default_rng(seed); d = []
    for _ in range(reps):
        i = rng.integers(0, len(y_np), len(y_np))
        if len(np.unique(y_np[i])) < 2: continue
        d.append(roc_auc_score(y_np[i], p1[i]) - roc_auc_score(y_np[i], p0[i]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

def calib(p):
    pc = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc))
    try:
        f = sm.Logit(y_np, sm.add_constant(lg)).fit(disp=0); slope = float(f.params[1])
        f2 = sm.GLM(y_np, np.ones((len(y_np), 1)), family=sm.families.Binomial(), offset=lg).fit()
        citl = float(f2.params[0])
    except Exception:
        slope, citl = np.nan, np.nan
    return citl, slope

# fold IDs
fold_id = np.zeros(len(df), int)
for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y), 1): fold_id[te] = k
pd.DataFrame({"RecordID": np.arange(1, len(df) + 1), "Fold": fold_id}).to_csv(OUT / "fold_ids_seed42.csv", index=False)

# ----------------------------------------------------------------------
# R4. ST_Slope-only demonstration
# ----------------------------------------------------------------------
demo = []
for v in ["distributed", "corrected"]:
    o = cv_oof(D[v], ["ST_Slope"], models=["Logistic_Regression"])["Logistic_Regression"]
    demo.append({"Analysis": f"LR on ST_Slope only, {v}", "n": len(df), "ROC_AUC": roc_auc_score(y_np, o)})
rec = ~prov_A["ST_Slope"].to_numpy()
sub = df[rec].reset_index(drop=True); ysub = y_np[rec]
sk = StratifiedKFold(5, shuffle=True, random_state=SEED); o = np.zeros(len(sub))
for tr, te in sk.split(sub, ysub):
    m = make_models(["ST_Slope"])["Logistic_Regression"]; m.fit(sub.iloc[tr][["ST_Slope"]], ysub[tr])
    o[te] = m.predict_proba(sub.iloc[te][["ST_Slope"]])[:, 1]
demo.append({"Analysis": "LR on ST_Slope only, recorded values only", "n": int(rec.sum()), "ROC_AUC": roc_auc_score(ysub, o)})
comp = prov_A["ST_Slope"].to_numpy()
rule = (df.ST_Slope.to_numpy() == "Flat").astype(int)
demo.append({"Analysis": "Rule Flat->HD, Up->no HD, completed records", "n": int(comp.sum()),
             "Accuracy": float((rule[comp] == y_np[comp]).mean())})
demo.append({"Analysis": "Rule Flat/Down->HD, Up->no HD, recorded records", "n": int(rec.sum()),
             "Accuracy": float(((df.ST_Slope.to_numpy()[rec] != "Up").astype(int) == y_np[rec]).mean())})
pd.DataFrame(demo).to_csv(OUT / "R4_STslope_only_demo.csv", index=False)
print(pd.DataFrame(demo).to_string(index=False)); tick("demo done")

# ----------------------------------------------------------------------
# R5. Oldpeak / ST_Slope ablation (random 5-fold, seed 42)
# ----------------------------------------------------------------------
C9 = FEATURE_TIERS["Core_9"]
SETS = {"Core_9": C9, "Core_9+Oldpeak": C9 + ["Oldpeak"], "Core_9+ST_Slope": C9 + ["ST_Slope"],
        "Core_9+Oldpeak+ST_Slope": C9 + ["Oldpeak", "ST_Slope"], "Core_6": FEATURE_TIERS["Core_6"]}
ABL = {}
abl_rows = []
for v in ["distributed", "corrected", "STslope_masked_only", "Oldpeak_masked_only", "corrected_zero_as_recorded"]:
    sets = SETS if v in ["distributed", "corrected"] else {k: SETS[k] for k in ["Core_9", "Core_9+Oldpeak+ST_Slope"]}
    for sname, feats in sets.items():
        ABL[(v, sname)] = cv_oof(D[v], feats)
        for m in MODEL_NAMES:
            p = ABL[(v, sname)][m]; ci, sl = calib(p)
            abl_rows.append({"Version": v, "FeatureSet": sname, "Model": m, "ROC_AUC": roc_auc_score(y_np, p),
                             "Brier": brier_score_loss(y_np, p), "Accuracy_at_0.5": accuracy_score(y_np, p >= 0.5),
                             "Calibration_in_the_large": ci, "Calibration_slope": sl})
        tick(f"ablation {v} / {sname}")
abl = pd.DataFrame(abl_rows); abl.to_csv(OUT / "R5_ablation_metrics.csv", index=False)
d_rows = []
for (v, sname), o in ABL.items():
    if sname in ["Core_9", "Core_6"]: continue
    for m in MODEL_NAMES:
        for ref in ["Core_9", "Core_6"]:
            if (v, ref) not in ABL: continue
            lo_, hi_ = paired_boot(o[m], ABL[(v, ref)][m])
            d_rows.append({"Version": v, "FeatureSet": sname, "Reference": ref, "Model": m,
                           "dAUC": roc_auc_score(y_np, o[m]) - roc_auc_score(y_np, ABL[(v, ref)][m]),
                           "CI95_Low": lo_, "CI95_High": hi_})
pd.DataFrame(d_rows).to_csv(OUT / "R5_ablation_dAUC.csv", index=False)
# calibration points for RF and LR, Extended-11
cp = []
for v in ["distributed", "corrected"]:
    for m in ["Random_Forest", "Logistic_Regression"]:
        fr, mp = calibration_curve(y_np, ABL[(v, "Core_9+Oldpeak+ST_Slope")][m], n_bins=10, strategy="quantile")
        for a, b in zip(mp, fr): cp.append({"Version": v, "Model": m, "MeanPredicted": a, "Observed": b})
pd.DataFrame(cp).to_csv(OUT / "R5_calibration_points.csv", index=False)
tick("ablation done")

# ----------------------------------------------------------------------
# R6. Repeated stratified CV (10 x 5) for Core-6 vs Extended-11
# ----------------------------------------------------------------------
E11 = FEATURE_TIERS["Extended_11"]; C6 = FEATURE_TIERS["Core_6"]
rep_rows = []
for r in range(10):
    for v in ["distributed", "corrected"]:
        o11 = cv_oof(D[v], E11, seed=1000 + r); o6 = cv_oof(D[v], C6, seed=1000 + r)
        for m in MODEL_NAMES:
            rep_rows.append({"Repeat": r, "Version": v, "Model": m,
                             "AUC_E11": roc_auc_score(y_np, o11[m]), "AUC_C6": roc_auc_score(y_np, o6[m])})
    tick(f"repeated CV {r+1}/10")
rep = pd.DataFrame(rep_rows); rep["dAUC"] = rep.AUC_E11 - rep.AUC_C6
rep.to_csv(OUT / "R6_repeatedCV_all.csv", index=False)
rep.groupby(["Version", "Model"]).agg(AUC_E11_mean=("AUC_E11", "mean"), AUC_C6_mean=("AUC_C6", "mean"),
    dAUC_mean=("dAUC", "mean"), dAUC_min=("dAUC", "min"), dAUC_max=("dAUC", "max"),
    dAUC_p2_5=("dAUC", lambda s: np.percentile(s, 2.5)), dAUC_p97_5=("dAUC", lambda s: np.percentile(s, 97.5))
    ).reset_index().to_csv(OUT / "R6_repeatedCV_summary.csv", index=False)

# ----------------------------------------------------------------------
# R7. LOCO with bootstrap CIs, macro / weighted / pooled AUC
# ----------------------------------------------------------------------
def loco(data, feats, coh, models=MODEL_NAMES, boot=True):
    rows_, pooled = [], {m: np.full(len(df), np.nan) for m in models}
    for held in SOURCES:
        te = np.where(coh == held)[0]; tr = np.where(coh != held)[0]
        for m, p in fit_predict(data, feats, tr, te, models).items():
            pooled[m][te] = p
            r_ = {"HeldOutCohort": held, "Model": m, "n": len(te), "n_noHD": int((y_np[te] == 0).sum()),
                  "ROC_AUC": roc_auc_score(y_np[te], p), "Brier": brier_score_loss(y_np[te], p)}
            if boot:
                rng = np.random.default_rng(SEED); bs = []
                for _ in range(1000):
                    i = rng.integers(0, len(te), len(te))
                    if len(np.unique(y_np[te][i])) < 2: continue
                    bs.append(roc_auc_score(y_np[te][i], p[i]))
                r_["CI95_Low"], r_["CI95_High"] = np.percentile(bs, [2.5, 97.5])
            rows_.append(r_)
    R = pd.DataFrame(rows_)
    S = R.groupby("Model").apply(lambda g: pd.Series({
        "macro_mean_AUC": g.ROC_AUC.mean(), "size_weighted_AUC": np.average(g.ROC_AUC, weights=g.n),
        "min_AUC": g.ROC_AUC.min(), "max_AUC": g.ROC_AUC.max()})).reset_index()
    S["pooled_out_of_cohort_AUC"] = [roc_auc_score(y_np, pooled[m]) for m in S.Model]
    return R, S

lo_all, ls_all = [], []
for v in ["distributed", "corrected"]:
    for tname, feats in [("Core_6", C6), ("Extended_11", E11)]:
        R, S = loco(D[v], feats, cohort); R.insert(0, "FeatureSet", tname); R.insert(0, "Version", v)
        S.insert(0, "FeatureSet", tname); S.insert(0, "Version", v); lo_all.append(R); ls_all.append(S)
        tick(f"LOCO {v} {tname}")
pd.concat(lo_all).to_csv(OUT / "R7_LOCO_by_cohort_CI.csv", index=False)
pd.concat(ls_all).to_csv(OUT / "R7_LOCO_summary.csv", index=False)

# ----------------------------------------------------------------------
# R8. Alternative assignment of cross-cohort ambiguous records
# ----------------------------------------------------------------------
amb = np.where(link_o.CandidateCohorts.str.contains(r"\|"))[0]
alt_cohort = cohort.copy(); alt_src = SRCIDX.copy()
for i in amb:
    cands = np.where(eq_o[i])[0]
    other = [c for c in cands if src.loc[c, "Cohort"] != cohort[i]]
    if other: alt_cohort[i] = src.loc[other[0], "Cohort"]; alt_src[i] = other[0]
alt_rows = [{"ambiguous_records": len(amb),
             "original_cohorts": "|".join(cohort[amb]), "alternative_cohorts": "|".join(alt_cohort[amb]),
             "STslope_completed_original": int(src_k.loc[SRCIDX, "ST_Slope"].isna().sum()),
             "STslope_completed_alternative": int(src_k.loc[alt_src, "ST_Slope"].isna().sum())}]
for co in SOURCES: alt_rows[0][f"n_{co}_alternative"] = int((alt_cohort == co).sum())
pd.DataFrame(alt_rows).to_csv(OUT / "R8_ambiguous_alternative_summary.csv", index=False)
alt_lo = []
for v in ["distributed", "corrected"]:
    R, S = loco(D[v], E11, alt_cohort, models=["Logistic_Regression", "Random_Forest"], boot=False)
    R.insert(0, "Version", v); alt_lo.append(R)
pd.concat(alt_lo).to_csv(OUT / "R8_LOCO_alternative_assignment.csv", index=False)
tick("alternative assignment done")

# ----------------------------------------------------------------------
# R9. Supplementary File S1 v2 (record-level provenance)
# ----------------------------------------------------------------------
S1 = pd.DataFrame({"RecordID": np.arange(1, len(df) + 1),
                   "SourceCohort": cohort, "SourceFile": [SOURCES[c] for c in cohort],
                   "SourceRow": link_o.SourceRow.to_numpy() + 1,
                   "LinkageCost_outcome_used": link_o.Cost.to_numpy().astype(int),
                   "MatchUnique_outcome_used": (link_o.EquallyGoodCandidates == 1).astype(int).to_numpy(),
                   "MatchAmbiguousAcrossCohorts": link_o.CandidateCohorts.str.contains(r"\|").astype(int).to_numpy(),
                   "SameMatch_outcome_blinded": (link_o.SrcIndex.to_numpy() == link_b.SrcIndex.to_numpy()).astype(int),
                   "HeartDisease": y_np, "Source_num": LS["num"].to_numpy()})
for p in PRED:
    sv = LS[SRC_COL[p]]
    S1[f"{p}_source_value"] = sv.astype(object).where(sv.notna(), "?")
    S1[f"{p}_distributed_value"] = df[p].to_numpy()
    if p in COMPLETABLE:
        st = np.where(prov_A[p].to_numpy(), "completed", "source_recorded")
        if p in ["Cholesterol", "RestingBP"]:
            st = np.where((LS[SRC_COL[p]].fillna(-1).to_numpy() == 0), "source_zero_coded", st)
        S1[f"{p}_status"] = st
    else:
        S1[f"{p}_status"] = "source_recorded"
S1.to_csv(OUT / "S1_HeartSense_provenance_v2.csv", index=False)
dd = [["RecordID", "Row number in heart.csv (1 = first data row)"],
      ["SourceCohort / SourceFile / SourceRow", "Linked UCI processed file and 1-based line number"],
      ["LinkageCost_outcome_used", "Number of disagreements with the linked source record (primary linkage; 0 = exact)"],
      ["MatchUnique_outcome_used", "1 if the linked record was the only zero-cost candidate"],
      ["MatchAmbiguousAcrossCohorts", "1 if equally good candidates existed in more than one cohort"],
      ["SameMatch_outcome_blinded", "1 if outcome-blinded linkage selected the same source record"],
      ["HeartDisease", "Outcome in heart.csv (1 = heart disease)"],
      ["Source_num", "UCI 'num' value of the linked record; HeartDisease = 1 if num > 0, else 0"],
      ["<Predictor>_source_value", "Value in the UCI processed file ('?' = not recorded), UCI coding"],
      ["<Predictor>_distributed_value", "Value in heart.csv"],
      ["<Predictor>_status", "source_recorded: present in source; completed: '?' in source but present in heart.csv; "
                             "source_zero_coded: 0 in source (Cholesterol/RestingBP), treated as not recorded in the primary analysis"]]
pd.DataFrame(dd, columns=["Column", "Description"]).to_csv(OUT / "S1_data_dictionary.csv", index=False)

# environment
env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
       "scipy": scipy.__version__, "statsmodels": sm.__version__}
import sklearn, xgboost, lightgbm, catboost
env.update({"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "catboost": catboost.__version__, "seed": SEED})
json.dump(env, open(OUT / "environment_versions.json", "w"), indent=2)

zp = shutil.make_archive(str(WORKDIR / "HeartSense_Revision"), "zip", root_dir=OUT)
tick("ALL DONE")
print("\nDOWNLOAD THIS ZIP:", zp)
