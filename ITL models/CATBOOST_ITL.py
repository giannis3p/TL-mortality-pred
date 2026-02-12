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

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))
THRESHOLD_REPORT = 0.5

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
    best_j, thr_youden = -2.0, 0.5

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
            best_j, thr_youden = j, t

    return float(thr_f1), float(thr_youden)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()
X_full = df_internal[shared_feature_cols].copy()
y_full = df_internal[target_column].astype(int).copy()

X_full = X_full.select_dtypes(include=["number"]).copy()
shared_feature_cols = X_full.columns.tolist()

binary_cols = [c for c in shared_feature_cols if X_full[c].dropna().nunique() == 2]
continuous_cols = [c for c in shared_feature_cols if c not in binary_cols]

binary_mode = X_full[binary_cols].mode().iloc[0] if binary_cols else pd.Series(dtype=float)
continuous_median = X_full[continuous_cols].median() if continuous_cols else pd.Series(dtype=float)
if binary_cols: X_full[binary_cols] = X_full[binary_cols].fillna(binary_mode)
if continuous_cols: X_full[continuous_cols] = X_full[continuous_cols].fillna(continuous_median)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

df_ext_copy = df_ext.copy()
X_ext = df_ext_copy[shared_feature_cols].copy()
y_ext = df_ext_copy[target_column].astype(int).copy()
ext_binary_cols = [c for c in binary_cols if c in X_ext.columns]
ext_continuous_cols = [c for c in continuous_cols if c in X_ext.columns]
if ext_binary_cols: X_ext[ext_binary_cols] = X_ext[ext_binary_cols].fillna(binary_mode[ext_binary_cols])
if ext_continuous_cols: X_ext[ext_continuous_cols] = X_ext[ext_continuous_cols].fillna(continuous_median[ext_continuous_cols])

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

cb_base = CatBoostClassifier(**base_cb_params)
cb_base.fit(X_train, y_train)

y_test_probs_base = cb_base.predict_proba(X_test)[:, 1]
print("Internal TEST AUC/AUPRC/Brier:",
      f"{roc_auc_score(y_test, y_test_probs_base):.4f}",
      f"{average_precision_score(y_test, y_test_probs_base):.4f}",
      f"{brier_score_loss(y_test, y_test_probs_base):.4f}"
)

roc_curves = {"base": [], "tl": [], "platt": []}
pr_curves  = {"base": [], "tl": [], "platt": []}
records = []

for seed in seeds:
    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
    )

    y_ext_tl_probs_base = cb_base.predict_proba(X_ext_tl)[:, 1]
    y_ext_test_probs_base = cb_base.predict_proba(X_ext_test)[:, 1]

    auc_base = roc_auc_score(y_ext_test, y_ext_test_probs_base)
    auprc_base = average_precision_score(y_ext_test, y_ext_test_probs_base)
    brier_base = brier_score_loss(y_ext_test, y_ext_test_probs_base)

    f1_b_05, sens_b_05, spec_b_05 = binary_metrics(y_ext_test, y_ext_test_probs_base, THRESHOLD_REPORT)

    thr_b_f1, thr_b_youden = find_best_thresholds(y_ext_tl, y_ext_tl_probs_base)
    f1_b_f1, sens_b_f1, spec_b_f1 = binary_metrics(y_ext_test, y_ext_test_probs_base, thr_b_f1)
    f1_b_yj,  sens_b_yj,  spec_b_yj  = binary_metrics(y_ext_test, y_ext_test_probs_base, thr_b_youden)

    def tl_objective(trial):
        lr_factor = trial.suggest_float("lr_factor", 0.1, 0.5)
        num_new_trees = trial.suggest_int("num_new_trees", 50, 400)
        pos_weight = trial.suggest_float("pos_weight", 0.5, 5.0, log=True)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_ext_tl, y_ext_tl, test_size=0.2, random_state=seed, stratify=y_ext_tl
        )

        tl_params = base_cb_params.copy()
        tl_params["iterations"] = num_new_trees
        tl_params["learning_rate"] = base_cb_params["learning_rate"] * lr_factor
        tl_params["auto_class_weights"] = None
        tl_params["class_weights"] = [1.0, pos_weight]

        model = CatBoostClassifier(**tl_params)
        model.fit(X_tr, y_tr, init_model=cb_base, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
        y_val_pred = model.predict_proba(X_val)[:, 1]
        return -roc_auc_score(y_val, y_val_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(tl_objective, n_trials=700)
    best = study.best_params

    tl_params_final = base_cb_params.copy()
    tl_params_final["iterations"] = best["num_new_trees"]
    tl_params_final["learning_rate"] = base_cb_params["learning_rate"] * best["lr_factor"]
    tl_params_final["auto_class_weights"] = None
    tl_params_final["class_weights"] = [1.0, best["pos_weight"]]

    tl_model = CatBoostClassifier(**tl_params_final)
    tl_model.fit(X_ext_tl, y_ext_tl, init_model=cb_base, verbose=False)

    y_ext_test_probs_tl = tl_model.predict_proba(X_ext_test)[:, 1]
    y_ext_tl_probs_tl   = tl_model.predict_proba(X_ext_tl)[:, 1]

    auc_tl = roc_auc_score(y_ext_test, y_ext_test_probs_tl)
    auprc_tl = average_precision_score(y_ext_test, y_ext_test_probs_tl)
    brier_tl = brier_score_loss(y_ext_test, y_ext_test_probs_tl)

    f1_t_05, sens_t_05, spec_t_05 = binary_metrics(y_ext_test, y_ext_test_probs_tl, THRESHOLD_REPORT)

    thr_t_f1, thr_t_youden = find_best_thresholds(y_ext_tl, y_ext_tl_probs_tl)
    f1_t_f1, sens_t_f1, spec_t_f1 = binary_metrics(y_ext_test, y_ext_test_probs_tl, thr_t_f1)
    f1_t_yj,  sens_t_yj,  spec_t_yj  = binary_metrics(y_ext_test, y_ext_test_probs_tl, thr_t_youden)

    lr = LogisticRegression(solver="lbfgs", max_iter=10000)
    lr.fit(y_ext_tl_probs_tl.reshape(-1, 1), y_ext_tl)
    y_ext_test_probs_platt = lr.predict_proba(y_ext_test_probs_tl.reshape(-1, 1))[:, 1]
    y_ext_tl_probs_platt   = lr.predict_proba(y_ext_tl_probs_tl.reshape(-1, 1))[:, 1]

    auc_platt = roc_auc_score(y_ext_test, y_ext_test_probs_platt)
    auprc_platt = average_precision_score(y_ext_test, y_ext_test_probs_platt)
    brier_platt = brier_score_loss(y_ext_test, y_ext_test_probs_platt)

    f1_p_05, sens_p_05, spec_p_05 = binary_metrics(y_ext_test, y_ext_test_probs_platt, THRESHOLD_REPORT)

    thr_p_f1, thr_p_youden = find_best_thresholds(y_ext_tl, y_ext_tl_probs_platt)
    f1_p_f1, sens_p_f1, spec_p_f1 = binary_metrics(y_ext_test, y_ext_test_probs_platt, thr_p_f1)
    f1_p_yj,  sens_p_yj,  spec_p_yj  = binary_metrics(y_ext_test, y_ext_test_probs_platt, thr_p_youden)

    fpr_b, tpr_b, _ = roc_curve(y_ext_test, y_ext_test_probs_base)
    fpr_t, tpr_t, _ = roc_curve(y_ext_test, y_ext_test_probs_tl)
    fpr_p, tpr_p, _ = roc_curve(y_ext_test, y_ext_test_probs_platt)
    roc_curves["base"].append((fpr_b, tpr_b, seed))
    roc_curves["tl"].append((fpr_t, tpr_t, seed))
    roc_curves["platt"].append((fpr_p, tpr_p, seed))

    prec_b, rec_b, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_base)
    prec_t, rec_t, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_tl)
    prec_p, rec_p, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_platt)
    pr_curves["base"].append((rec_b, prec_b, seed))
    pr_curves["tl"].append((rec_t, prec_t, seed))
    pr_curves["platt"].append((rec_p, prec_p, seed))

    records.append({
        "seed": seed,
        "auc_base": auc_base, "auprc_base": auprc_base, "brier_base": brier_base,
        "auc_tl": auc_tl, "auprc_tl": auprc_tl, "brier_tl": brier_tl,
        "auc_platt": auc_platt, "auprc_platt": auprc_platt, "brier_platt": brier_platt,

        "f1_base_0_5": f1_b_05, "sens_base_0_5": sens_b_05, "spec_base_0_5": spec_b_05,
        "f1_tl_0_5": f1_t_05,   "sens_tl_0_5": sens_t_05,   "spec_tl_0_5": spec_t_05,
        "f1_platt_0_5": f1_p_05, "sens_platt_0_5": sens_p_05, "spec_platt_0_5": spec_p_05,

        "thr_base_f1": thr_b_f1, "f1_base_f1": f1_b_f1, "sens_base_f1": sens_b_f1, "spec_base_f1": spec_b_f1,
        "thr_tl_f1": thr_t_f1,   "f1_tl_f1": f1_t_f1,   "sens_tl_f1": sens_t_f1,   "spec_tl_f1": spec_t_f1,
        "thr_platt_f1": thr_p_f1, "f1_platt_f1": f1_p_f1, "sens_platt_f1": sens_p_f1, "spec_platt_f1": spec_p_f1,

        "thr_base_yj": thr_b_youden, "f1_base_yj": f1_b_yj, "sens_base_yj": sens_b_yj, "spec_base_yj": spec_b_yj,
        "thr_tl_yj": thr_t_youden,   "f1_tl_yj": f1_t_yj,   "sens_tl_yj": sens_t_yj,   "spec_tl_yj": spec_t_yj,
        "thr_platt_yj": thr_p_youden, "f1_platt_yj": f1_p_yj, "sens_platt_yj": sens_p_yj, "spec_platt_yj": spec_p_yj,

        "lr_factor_best": best["lr_factor"],
        "num_new_trees_best": best["num_new_trees"],
        "pos_weight_best": best["pos_weight"],
    })

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("catboost_metrics_per_seed.csv", index=False)

with open("catboost_roc_pr_curves.pkl", "wb") as f:
    pickle.dump({"roc_curves": roc_curves, "pr_curves": pr_curves}, f)

print("Saved catboost_metrics_per_seed.csv with tuned thresholds and metrics.")
print("Saved catboost_roc_pr_curves.pkl")