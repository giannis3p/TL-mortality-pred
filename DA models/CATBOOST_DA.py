import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))
THRESHOLD = 0.5

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return f1, sens, spec

def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, n_grid: int = 1001):
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_f1, thr_f1 = -1.0, 0.5
    best_j, thr_yj = -2.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if f1 > best_f1:
            best_f1, thr_f1 = f1, t
        if j > best_j:
            best_j, thr_yj = j, t
    return float(thr_f1), float(thr_yj)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))
print("\nNumber of shared features:", len(shared_feature_cols))

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()
X_full = df_internal[shared_feature_cols].select_dtypes(include=["number"]).copy()
shared_feature_cols = X_full.columns.tolist()
y_full = df_internal[target_column].astype(int).copy()

binary_cols = [c for c in shared_feature_cols if X_full[c].nunique(dropna=True) == 2]
continuous_cols = [c for c in shared_feature_cols if c not in binary_cols]
binary_mode = X_full[binary_cols].mode().iloc[0] if binary_cols else None
continuous_median = X_full[continuous_cols].median() if continuous_cols else None
if binary_cols:     X_full[binary_cols] = X_full[binary_cols].fillna(binary_mode)
if continuous_cols: X_full[continuous_cols] = X_full[continuous_cols].fillna(continuous_median)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

X_ext = df_ext[shared_feature_cols].copy()
y_ext = df_ext[target_column].astype(int).copy()
if binary_cols:     X_ext[binary_cols] = X_ext[binary_cols].fillna(binary_mode)
if continuous_cols: X_ext[continuous_cols] = X_ext[continuous_cols].fillna(continuous_median)

print("\nShapes:", X_train.shape, X_test.shape, "| X_ext:", X_ext.shape)
print("External prevalence:", float(y_ext.mean()))

base_cb_params = dict(
    iterations=727,
    learning_rate=0.013717386230253304,
    depth=13,
    l2_leaf_reg=13.999183589638127,
    random_strength=0.16296340751402472,
    bootstrap_type="Bernoulli",
    subsample=0.5369133048391227,
    rsm=0.9915296051933175,
    grow_policy="Lossguide",
    min_data_in_leaf=88,
    max_leaves=475,
    auto_class_weights="Balanced",
    score_function="L2",
    od_type="Iter",
    od_wait=190,
    loss_function="Logloss",
    eval_metric="AUC",
    task_type="CPU",
    thread_count=-1,
    random_seed=42,
    verbose=False,
    allow_writing_files=False
)

clf_base = CatBoostClassifier(**base_cb_params)
clf_base.fit(X_train, y_train)

def da_objective(trial):
    C_dom  = trial.suggest_float("C_dom", 1e-6, 1e4, log=True)
    w_min  = trial.suggest_float("w_min", 0.1, 1.0)
    w_max  = trial.suggest_float("w_max", 1.0, 50.0)
    alpha  = trial.suggest_float("alpha", 0.5, 5.0)
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    X_domain = pd.concat([X_train, X_ext], axis=0)
    d_domain = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)])
    dom_clf = LogisticRegression(max_iter=10000, C=C_dom, solver="lbfgs")
    dom_clf.fit(X_domain, d_domain)

    p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
    w = (p_ext / (1.0 - p_ext)) ** alpha
    w = np.clip(w, w_min, w_max)

    clf_da = CatBoostClassifier(**base_cb_params)
    clf_da.fit(X_train, y_train, sample_weight=w)

    auc_improvements = []
    for seed in (25, 26, 27, 28, 29):
        X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
            X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
        )
        base_probs = clf_base.predict_proba(X_ext_test)[:, 1]
        da_probs   = clf_da.predict_proba(X_ext_test)[:, 1]
        auc_improvements.append(
            roc_auc_score(y_ext_test, da_probs) - roc_auc_score(y_ext_test, base_probs)
        )
    return float(np.mean(auc_improvements))

print("\n Running Optuna (DA hyperparameters) ")
study_da = optuna.create_study(direction="maximize")
study_da.optimize(da_objective, n_trials=300)
best_da_params = study_da.best_params
print("Best DA params:", best_da_params)

C_dom = best_da_params["C_dom"]
w_min = best_da_params["w_min"]
w_max = best_da_params["w_max"]
alpha = best_da_params["alpha"]
if w_min > w_max:
    w_min, w_max = w_max, w_min

X_domain = pd.concat([X_train, X_ext], axis=0)
d_domain = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)])
dom_clf = LogisticRegression(max_iter=10000, C=C_dom, solver="lbfgs")
dom_clf.fit(X_domain, d_domain)

p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
w = (p_ext / (1.0 - p_ext)) ** alpha
w = np.clip(w, w_min, w_max)

clf_da = CatBoostClassifier(**base_cb_params)
clf_da.fit(X_train, y_train, sample_weight=w)

records_da = []
roc_curves_da = {"base": [], "da": [], "platt": [], "iso": []}
pr_curves_da  = {"base": [], "da": [], "platt": [], "iso": []}

for seed in seeds:
    print(f"\n SEED {seed} ")

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
    )

    base_probs_tl   = clf_base.predict_proba(X_ext_tl)[:, 1]
    base_probs_test = clf_base.predict_proba(X_ext_test)[:, 1]

    da_probs_tl   = clf_da.predict_proba(X_ext_tl)[:, 1]
    da_probs_test = clf_da.predict_proba(X_ext_test)[:, 1]

    lr = LogisticRegression(max_iter=10000, solver="lbfgs")
    lr.fit(da_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs_tl   = lr.predict_proba(da_probs_tl.reshape(-1, 1))[:, 1]
    platt_probs_test = lr.predict_proba(da_probs_test.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(da_probs_tl, y_ext_tl)
    iso_probs_tl   = iso.predict(da_probs_tl)
    iso_probs_test = iso.predict(da_probs_test)

    auc_base   = roc_auc_score(y_ext_test, base_probs_test)
    auc_da     = roc_auc_score(y_ext_test, da_probs_test)
    auc_platt  = roc_auc_score(y_ext_test, platt_probs_test)
    auc_iso    = roc_auc_score(y_ext_test, iso_probs_test)

    auprc_base  = average_precision_score(y_ext_test, base_probs_test)
    auprc_da    = average_precision_score(y_ext_test, da_probs_test)
    auprc_platt = average_precision_score(y_ext_test, platt_probs_test)
    auprc_iso   = average_precision_score(y_ext_test, iso_probs_test)

    brier_base  = brier_score_loss(y_ext_test, base_probs_test)
    brier_da    = brier_score_loss(y_ext_test, da_probs_test)
    brier_platt = brier_score_loss(y_ext_test, platt_probs_test)
    brier_iso   = brier_score_loss(y_ext_test, iso_probs_test)

    f1_b_05, sens_b_05, spec_b_05 = binary_metrics(y_ext_test, base_probs_test, THRESHOLD)
    f1_d_05, sens_d_05, spec_d_05 = binary_metrics(y_ext_test, da_probs_test, THRESHOLD)
    f1_p_05, sens_p_05, spec_p_05 = binary_metrics(y_ext_test, platt_probs_test, THRESHOLD)
    f1_i_05, sens_i_05, spec_i_05 = binary_metrics(y_ext_test, iso_probs_test, THRESHOLD)

    thr_b_f1, thr_b_yj = find_best_thresholds(y_ext_tl, base_probs_tl)
    f1_b_f1, sens_b_f1, spec_b_f1 = binary_metrics(y_ext_test, base_probs_test, thr_b_f1)
    f1_b_yj,  sens_b_yj,  spec_b_yj  = binary_metrics(y_ext_test, base_probs_test, thr_b_yj)

    thr_d_f1, thr_d_yj = find_best_thresholds(y_ext_tl, da_probs_tl)
    f1_d_f1, sens_d_f1, spec_d_f1 = binary_metrics(y_ext_test, da_probs_test, thr_d_f1)
    f1_d_yj,  sens_d_yj,  spec_d_yj  = binary_metrics(y_ext_test, da_probs_test, thr_d_yj)

    thr_p_f1, thr_p_yj = find_best_thresholds(y_ext_tl, platt_probs_tl)
    f1_p_f1, sens_p_f1, spec_p_f1 = binary_metrics(y_ext_test, platt_probs_test, thr_p_f1)
    f1_p_yj,  sens_p_yj,  spec_p_yj  = binary_metrics(y_ext_test, platt_probs_test, thr_p_yj)

    thr_i_f1, thr_i_yj = find_best_thresholds(y_ext_tl, iso_probs_tl)
    f1_i_f1, sens_i_f1, spec_i_f1 = binary_metrics(y_ext_test, iso_probs_test, thr_i_f1)
    f1_i_yj,  sens_i_yj,  spec_i_yj  = binary_metrics(y_ext_test, iso_probs_test, thr_i_yj)

    fpr_b, tpr_b, _ = roc_curve(y_ext_test, base_probs_test)
    fpr_d, tpr_d, _ = roc_curve(y_ext_test, da_probs_test)
    fpr_p, tpr_p, _ = roc_curve(y_ext_test, platt_probs_test)
    fpr_i, tpr_i, _ = roc_curve(y_ext_test, iso_probs_test)
    roc_curves_da["base"].append((fpr_b, tpr_b, seed))
    roc_curves_da["da"].append((fpr_d, tpr_d, seed))
    roc_curves_da["platt"].append((fpr_p, tpr_p, seed))
    roc_curves_da["iso"].append((fpr_i, tpr_i, seed))

    prec_b, rec_b, _ = precision_recall_curve(y_ext_test, base_probs_test)
    prec_d, rec_d, _ = precision_recall_curve(y_ext_test, da_probs_test)
    prec_p, rec_p, _ = precision_recall_curve(y_ext_test, platt_probs_test)
    prec_i, rec_i, _ = precision_recall_curve(y_ext_test, iso_probs_test)
    pr_curves_da["base"].append((rec_b, prec_b, seed))
    pr_curves_da["da"].append((rec_d, prec_d, seed))
    pr_curves_da["platt"].append((rec_p, prec_p, seed))
    pr_curves_da["iso"].append((rec_i, prec_i, seed))

    records_da.append({
        "seed": seed,

        "auc_base": auc_base, "auc_da": auc_da, "auc_platt": auc_platt, "auc_iso": auc_iso,
        "auprc_base": auprc_base, "auprc_da": auprc_da, "auprc_platt": auprc_platt, "auprc_iso": auprc_iso,
        "brier_base": brier_base, "brier_da": brier_da, "brier_platt": brier_platt, "brier_iso": brier_iso,

        "f1_base_0_5": f1_b_05, "sens_base_0_5": sens_b_05, "spec_base_0_5": spec_b_05,
        "f1_da_0_5": f1_d_05,   "sens_da_0_5": sens_d_05,   "spec_da_0_5": spec_d_05,
        "f1_platt_0_5": f1_p_05, "sens_platt_0_5": sens_p_05, "spec_platt_0_5": spec_p_05,
        "f1_iso_0_5": f1_i_05,   "sens_iso_0_5": sens_i_05,   "spec_iso_0_5": spec_i_05,

        "thr_base_f1": thr_b_f1, "thr_base_yj": thr_b_yj,
        "thr_da_f1": thr_d_f1,   "thr_da_yj": thr_d_yj,
        "thr_platt_f1": thr_p_f1, "thr_platt_yj": thr_p_yj,
        "thr_iso_f1": thr_i_f1,   "thr_iso_yj": thr_i_yj,

        "f1_base_f1": f1_b_f1, "sens_base_f1": sens_b_f1, "spec_base_f1": spec_b_f1,
        "f1_da_f1": f1_d_f1,   "sens_da_f1": sens_d_f1,   "spec_da_f1": spec_d_f1,
        "f1_platt_f1": f1_p_f1, "sens_platt_f1": sens_p_f1, "spec_platt_f1": spec_p_f1,
        "f1_iso_f1": f1_i_f1,   "sens_iso_f1": sens_i_f1,   "spec_iso_f1": spec_i_f1,

        "f1_base_yj": f1_b_yj, "sens_base_yj": sens_b_yj, "spec_base_yj": spec_b_yj,
        "f1_da_yj": f1_d_yj,   "sens_da_yj": sens_d_yj,   "spec_da_yj": spec_d_yj,
        "f1_platt_yj": f1_p_yj, "sens_platt_yj": sens_p_yj, "spec_platt_yj": spec_p_yj,
        "f1_iso_yj": f1_i_yj,   "sens_iso_yj": sens_i_yj,   "spec_iso_yj": spec_i_yj,
    })

df_da = pd.DataFrame(records_da)
df_da.to_csv("catboost_metrics_da_per_seed.csv", index=False)

with open("catboost_roc_pr_curves_da.pkl", "wb") as f:
    pickle.dump({"roc_curves_da": roc_curves_da, "pr_curves_da": pr_curves_da}, f)

print("\nSaved catboost_metrics_da_per_seed.csv (incl. tuned thresholds & metrics) and catboost_roc_pr_curves_da.pkl!")