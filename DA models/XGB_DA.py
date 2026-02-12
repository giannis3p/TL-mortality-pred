import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))

def youden_threshold(y_true, probs) -> float:
    fpr, tpr, thr = roc_curve(y_true, probs)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thr[idx])

def sensitivity_specificity(y_true, probs, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(probs) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec, tp, tn, fp, fn

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]

shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))
print("\nNumber of shared features between df and df_ext:", len(shared_feature_cols))

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

n_pos = (df_internal[target_column] == 1).sum()
n_neg = (df_internal[target_column] == 0).sum()
prev_internal = n_pos / (n_pos + n_neg)

print(f"\nInternal: n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_internal:.4f}")

X_full = df_internal[shared_feature_cols].copy()
y_full = df_internal[target_column].copy()

X_full = X_full.select_dtypes(include=["number"])
shared_feature_cols = X_full.columns.tolist()
print("\nShared numeric features used:", len(shared_feature_cols))

binary_cols = [c for c in shared_feature_cols if X_full[c].dropna().nunique() == 2]
continuous_cols = [c for c in shared_feature_cols if c not in binary_cols]

binary_mode = X_full[binary_cols].mode().iloc[0] if binary_cols else pd.Series(dtype=float)
continuous_median = X_full[continuous_cols].median() if continuous_cols else pd.Series(dtype=float)

if binary_cols:
    X_full[binary_cols] = X_full[binary_cols].fillna(binary_mode)
if continuous_cols:
    X_full[continuous_cols] = X_full[continuous_cols].fillna(continuous_median)

X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Internal train prevalence:", y_train.mean())
print("Internal test  prevalence:", y_test.mean())

df_ext_copy = df_ext[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

X_ext = df_ext_copy[shared_feature_cols].copy()
y_ext = df_ext_copy[target_column].copy()

ext_binary_cols = [c for c in binary_cols if c in X_ext.columns]
ext_continuous_cols = [c for c in continuous_cols if c in X_ext.columns]

if ext_binary_cols:
    X_ext[ext_binary_cols] = X_ext[ext_binary_cols].fillna(binary_mode[ext_binary_cols])
if ext_continuous_cols:
    X_ext[ext_continuous_cols] = X_ext[ext_continuous_cols].fillna(continuous_median[ext_continuous_cols])

print("\nX_ext shape:", X_ext.shape)
print("External outcome prevalence:", y_ext.mean())

base_params = dict(
  n_estimators= 1276,
  learning_rate= 0.012883864113018737,
  max_depth=10,
  subsample= 0.497031380963004,
  colsample_bytree= 0.5109366122090663,
  gamma= 4.541447745087957,
  min_child_weight= 11.357305655375951,
  reg_alpha= 0.04180492475930316,
  reg_lambda= 6.447536920879301e-08,
  scale_pos_weight= 2.7071640044982996,
  eval_metric='logloss',
  random_state=42,
)

base_params_gpu = dict(
    **base_params,
    tree_method="hist",
   # predictor="gpu_predictor",
    device="cuda",
)


clf_base = xgb.XGBClassifier(**base_params_gpu)
clf_base.fit(X_train, y_train)

def da_objective(trial):

    C_dom  = trial.suggest_float("C_dom", 1e-6, 1e4, log=True)
    w_min  = trial.suggest_float("w_min", 0.1, 1.0)
    w_max  = trial.suggest_float("w_max", 1.0, 50.0)
    alpha  = trial.suggest_float("alpha", 0.5, 5.0)
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    X_domain = pd.concat([X_train, X_ext], axis=0)
    d_domain = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_ext), dtype=int),
    ])

    dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
    dom_clf.fit(X_domain, d_domain)

    p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
    w = np.power(p_ext / (1 - p_ext), alpha)
    w = np.clip(w, w_min, w_max)

    clf_da = xgb.XGBClassifier(**base_params_gpu)
    clf_da.fit(X_train, y_train, sample_weight=w)

    auc_improvements = []
    auprc_improvements = []

    for seed in seeds:

        X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
            X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
        )

        base_probs = clf_base.predict_proba(X_ext_test)[:, 1]
        da_probs   = clf_da.predict_proba(X_ext_test)[:, 1]

        auc_improvements.append(
            roc_auc_score(y_ext_test, da_probs) - roc_auc_score(y_ext_test, base_probs)
        )
        auprc_improvements.append(
            average_precision_score(y_ext_test, da_probs)
            - average_precision_score(y_ext_test, base_probs)
        )

    return 0.8 * np.mean(auc_improvements) + 0.2 * np.mean(auprc_improvements)


print("\n Running Optuna (DA hyperparameters) ")
study_da = optuna.create_study(direction="maximize")
study_da.optimize(da_objective, n_trials=1000)
best_da_params = study_da.best_params

print("\nBest DA params:", best_da_params)

C_dom = best_da_params["C_dom"]
w_min = best_da_params["w_min"]
w_max = best_da_params["w_max"]
alpha = best_da_params["alpha"]
if w_min > w_max:
    w_min, w_max = w_max, w_min

X_domain = pd.concat([X_train, X_ext], axis=0)
d_domain = np.concatenate([
    np.zeros(len(X_train), dtype=int),
    np.ones(len(X_ext), dtype=int),
])

dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
dom_clf.fit(X_domain, d_domain)

p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
w = np.power(p_ext / (1 - p_ext), alpha)
w = np.clip(w, w_min, w_max)

clf_da = xgb.XGBClassifier(**base_params_gpu)
clf_da.fit(X_train, y_train, sample_weight=w)

records_da = []

roc_curves_da = {"base": [], "da": [], "platt": [], "iso": []}
pr_curves_da  = {"base": [], "da": [], "platt": [], "iso": []}

for seed in seeds:
    print(f"\nSEED {seed}")

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
    )

    base_probs_tl = clf_base.predict_proba(X_ext_tl)[:, 1]
    base_probs = clf_base.predict_proba(X_ext_test)[:, 1]

    da_probs = clf_da.predict_proba(X_ext_test)[:, 1]
    da_probs_tl = clf_da.predict_proba(X_ext_tl)[:, 1]

    lr = LogisticRegression(max_iter=10000)
    lr.fit(da_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs_tl = lr.predict_proba(da_probs_tl.reshape(-1, 1))[:, 1]
    platt_probs = lr.predict_proba(da_probs.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(da_probs_tl, y_ext_tl)
    iso_probs_tl = iso.predict(da_probs_tl)
    iso_probs = iso.predict(da_probs)

    thr_base = youden_threshold(y_ext_tl, base_probs_tl)
    thr_da = youden_threshold(y_ext_tl, da_probs_tl)
    thr_platt = youden_threshold(y_ext_tl, platt_probs_tl)
    thr_iso = youden_threshold(y_ext_tl, iso_probs_tl)

    sens_base, spec_base, tp_b, tn_b, fp_b, fn_b = sensitivity_specificity(y_ext_test, base_probs, thr_base)
    sens_da,   spec_da,   tp_d, tn_d, fp_d, fn_d = sensitivity_specificity(y_ext_test, da_probs, thr_da)
    sens_platt,spec_platt,tp_p, tn_p, fp_p, fn_p = sensitivity_specificity(y_ext_test, platt_probs, thr_platt)
    sens_iso,  spec_iso,  tp_i, tn_i, fp_i, fn_i = sensitivity_specificity(y_ext_test, iso_probs, thr_iso)

    auc_base = roc_auc_score(y_ext_test, base_probs)
    auc_da   = roc_auc_score(y_ext_test, da_probs)
    auc_platt= roc_auc_score(y_ext_test, platt_probs)
    auc_iso  = roc_auc_score(y_ext_test, iso_probs)

    auprc_base = average_precision_score(y_ext_test, base_probs)
    auprc_da   = average_precision_score(y_ext_test, da_probs)
    auprc_platt= average_precision_score(y_ext_test, platt_probs)
    auprc_iso  = average_precision_score(y_ext_test, iso_probs)

    brier_base = brier_score_loss(y_ext_test, base_probs)
    brier_da   = brier_score_loss(y_ext_test, da_probs)
    brier_platt= brier_score_loss(y_ext_test, platt_probs)
    brier_iso  = brier_score_loss(y_ext_test, iso_probs)

    records_da.append({
        "seed": seed,

        "auc_base": auc_base, "auc_da": auc_da,
        "auc_platt": auc_platt, "auc_iso": auc_iso,

        "auprc_base": auprc_base, "auprc_da": auprc_da,
        "auprc_platt": auprc_platt, "auprc_iso": auprc_iso,

        "brier_base": brier_base, "brier_da": brier_da,
        "brier_platt": brier_platt, "brier_iso": brier_iso,

        "thr_base": thr_base, "thr_da": thr_da,
        "thr_platt": thr_platt, "thr_iso": thr_iso,

        "sens_base": sens_base, "spec_base": spec_base,
        "sens_da": sens_da, "spec_da": spec_da,
        "sens_platt": sens_platt, "spec_platt": spec_platt,
        "sens_iso": sens_iso, "spec_iso": spec_iso,

        "tp_base": tp_b, "tn_base": tn_b, "fp_base": fp_b, "fn_base": fn_b,
        "tp_da": tp_d, "tn_da": tn_d, "fp_da": fp_d, "fn_da": fn_d,
        "tp_platt": tp_p, "tn_platt": tn_p, "fp_platt": fp_p, "fn_platt": fn_p,
        "tp_iso": tp_i, "tn_iso": tn_i, "fp_iso": fp_i, "fn_iso": fn_i,
    })

    fpr_b, tpr_b, _ = roc_curve(y_ext_test, base_probs)
    fpr_d, tpr_d, _ = roc_curve(y_ext_test, da_probs)
    fpr_p, tpr_p, _ = roc_curve(y_ext_test, platt_probs)
    fpr_i, tpr_i, _ = roc_curve(y_ext_test, iso_probs)

    roc_curves_da["base"].append((fpr_b, tpr_b, seed))
    roc_curves_da["da"].append((fpr_d, tpr_d, seed))
    roc_curves_da["platt"].append((fpr_p, tpr_p, seed))
    roc_curves_da["iso"].append((fpr_i, tpr_i, seed))

    prec_b, rec_b, _ = precision_recall_curve(y_ext_test, base_probs)
    prec_d, rec_d, _ = precision_recall_curve(y_ext_test, da_probs)
    prec_p, rec_p, _ = precision_recall_curve(y_ext_test, platt_probs)
    prec_i, rec_i, _ = precision_recall_curve(y_ext_test, iso_probs)

    pr_curves_da["base"].append((rec_b, prec_b, seed))
    pr_curves_da["da"].append((rec_d, prec_d, seed))
    pr_curves_da["platt"].append((rec_p, prec_p, seed))
    pr_curves_da["iso"].append((rec_i, prec_i, seed))

df_da = pd.DataFrame(records_da)
df_da.to_csv("metrics_da_per_seed.csv", index=False)

with open("roc_pr_curves_da.pkl", "wb") as f:
    pickle.dump(
        {"roc_curves_da": roc_curves_da, "pr_curves_da": pr_curves_da},
        f
    )

print("\nSaved metrics_da_per_seed.csv and roc_pr_curves_da.pkl!")

