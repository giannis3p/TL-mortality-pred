import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
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

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

roc_curves = {"base": [], "tl": [], "platt": []}
pr_curves = {"base": [], "tl": [], "platt": []}
records = []

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))

print("\nNumber of shared features between df and df_ext:", len(shared_feature_cols))
print("Shared features:", shared_feature_cols)

def _sanitize_lgbm_feature_names(cols):
    out = []
    seen = {}
    for c in cols:
        s = str(c)
        s = re.sub(r'[\[\]\{\}":,\\\n\r\t]', "_", s)
        s = re.sub(r"\s+", "_", s).strip("_")
        if s == "":
            s = "feature"
        if s in seen:
            seen[s] += 1
            s = f"{s}__{seen[s]}"
        else:
            seen[s] = 0
        out.append(s)
    return out


def youden_optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    threshold = float(thr[idx])
    sensitivity = float(tpr[idx])
    specificity = float(1.0 - fpr[idx])
    return threshold, sensitivity, specificity, fpr, tpr, thr

sanitized_shared = _sanitize_lgbm_feature_names(shared_feature_cols)
rename_map = dict(zip(shared_feature_cols, sanitized_shared))
shared_feature_cols = sanitized_shared

df_internal = df[list(rename_map.keys()) + [target_column]].copy()
df_internal = df_internal.dropna(subset=[target_column]).rename(columns=rename_map)
df_internal[target_column] = df_internal[target_column].astype(int)

df_ext_copy = df_ext.copy().rename(columns=rename_map)
df_ext_copy = df_ext_copy.dropna(subset=[target_column]).copy()
df_ext_copy[target_column] = df_ext_copy[target_column].astype(int)

n_pos = int((df_internal[target_column] == 1).sum())
n_neg = int((df_internal[target_column] == 0).sum())
prev_internal = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0
print(f"\nInternal : n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_internal:.4f}")

X_full = df_internal[shared_feature_cols].copy()
y_full = df_internal[target_column].copy()

X_full = X_full.select_dtypes(include=["number"]).copy()
shared_feature_cols = X_full.columns.tolist()

binary_cols = [c for c in shared_feature_cols if X_full[c].dropna().nunique() == 2]
continuous_cols = [c for c in shared_feature_cols if c not in binary_cols]

print("\nBinary shared features:", binary_cols)
print("Continuous shared features:", continuous_cols)

binary_mode = X_full[binary_cols].mode().iloc[0] if binary_cols else pd.Series(dtype=float)
continuous_median = X_full[continuous_cols].median() if continuous_cols else pd.Series(dtype=float)

if binary_cols:
    X_full[binary_cols] = X_full[binary_cols].fillna(binary_mode)
if continuous_cols:
    X_full[continuous_cols] = X_full[continuous_cols].fillna(continuous_median)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Internal train prevalence:", float(y_train.mean()))
print("Internal test  prevalence:", float(y_test.mean()))

X_ext = df_ext_copy[shared_feature_cols].copy()
y_ext = df_ext_copy[target_column].copy()

ext_binary_cols = [c for c in binary_cols if c in X_ext.columns]
ext_continuous_cols = [c for c in continuous_cols if c in X_ext.columns]

if ext_binary_cols:
    X_ext[ext_binary_cols] = X_ext[ext_binary_cols].fillna(binary_mode[ext_binary_cols])
if ext_continuous_cols:
    X_ext[ext_continuous_cols] = X_ext[ext_continuous_cols].fillna(continuous_median[ext_continuous_cols])

print("\nX_ext shape:", X_ext.shape)
print("External outcome prevalence:", float(y_ext.mean()))

base_params = dict(
    objective="binary",
    max_depth=9,
    n_estimators=913,
    learning_rate=0.014186085276160578,
    num_leaves=354,
    subsample=0.7936434734677719,
    subsample_freq=2,
    colsample_bytree=0.6256418317036112,
    scale_pos_weight=3.1819882811944264,
    min_child_samples=94,
    min_child_weight=0.02116807200498526,
    min_split_gain=0.14290833051635013,
    reg_alpha=0.644205382837298,
    reg_lambda=0.01396884928662634,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

clf_base = lgb.LGBMClassifier(**base_params)
clf_base.fit(X_train, y_train)

y_test_probs_base = clf_base.predict_proba(X_test)[:, 1]
auc_int = roc_auc_score(y_test, y_test_probs_base)
auprc_int = average_precision_score(y_test, y_test_probs_base)
brier_int = brier_score_loss(y_test, y_test_probs_base)

thr_int, sens_int, spec_int, *_ = youden_optimal_threshold(y_test, y_test_probs_base)

print("\n Internal TEST performance ")
print(f"Internal TEST AUC:   {auc_int:.4f}")
print(f"Internal TEST AUPRC: {auprc_int:.4f}")
print(f"Internal TEST Brier: {brier_int:.4f}")
print(f"Internal TEST Youden threshold: {thr_int:.6f} | Sensitivity: {sens_int:.4f} | Specificity: {spec_int:.4f}")

seeds = list(range(27, 32))

for seed in seeds:
    print(f"\n SEED {seed} ")

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
    )

    y_ext_test_probs_base = clf_base.predict_proba(X_ext_test)[:, 1]

    auc_base = roc_auc_score(y_ext_test, y_ext_test_probs_base)
    auprc_base = average_precision_score(y_ext_test, y_ext_test_probs_base)
    brier_base = brier_score_loss(y_ext_test, y_ext_test_probs_base)

    thr_base, sens_base, spec_base, fpr_base, tpr_base, _ = youden_optimal_threshold(
        y_ext_test, y_ext_test_probs_base
    )

    print(
        f"External TEST (BASE) - AUC: {auc_base:.4f}, AUPRC: {auprc_base:.4f}, Brier: {brier_base:.4f} | "
        f"Youden thr: {thr_base:.6f}, Sens: {sens_base:.4f}, Spec: {spec_base:.4f}"
    )

    def tl_objective(trial):
        eta_factor = trial.suggest_float("eta_factor", 0.1, 0.5)
        num_new_trees = trial.suggest_int("num_new_trees", 50, 400)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 5.0, log=True)

        X_tl_tr, X_tl_val, y_tl_tr, y_tl_val = train_test_split(
            X_ext_tl, y_ext_tl, test_size=0.2, random_state=seed, stratify=y_ext_tl
        )

        clf_for_tl = lgb.LGBMClassifier(**base_params)
        clf_for_tl.fit(X_train, y_train)
        booster_base = clf_for_tl.booster_

        params_tl = dict(base_params)
        params_tl.pop("n_estimators", None)
        params_tl["learning_rate"] = base_params["learning_rate"] * eta_factor
        params_tl["scale_pos_weight"] = scale_pos_weight
        params_tl["objective"] = "binary"
        params_tl["metric"] = "auc"

        d_tl_tr = lgb.Dataset(X_tl_tr, label=y_tl_tr, free_raw_data=False)
        d_tl_val = lgb.Dataset(X_tl_val, label=y_tl_val, free_raw_data=False)

        updated_model = lgb.train(
            params=params_tl,
            train_set=d_tl_tr,
            num_boost_round=num_new_trees,
            init_model=booster_base,
            valid_sets=[d_tl_val],
        )

        y_val_pred = updated_model.predict(X_tl_val)
        return -roc_auc_score(y_tl_val, y_val_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(tl_objective, n_trials=200)

    best_params_seed = study.best_params
    print(f"Best TL params for seed {seed}: {best_params_seed}")

    eta_factor_best = best_params_seed["eta_factor"]
    num_new_trees_best = best_params_seed["num_new_trees"]
    scale_pos_weight_best = best_params_seed["scale_pos_weight"]

    clf_for_tl = lgb.LGBMClassifier(**base_params)
    clf_for_tl.fit(X_train, y_train)
    booster_base = clf_for_tl.booster_

    params_tl = dict(base_params)
    params_tl.pop("n_estimators", None)
    params_tl["learning_rate"] = base_params["learning_rate"] * eta_factor_best
    params_tl["scale_pos_weight"] = scale_pos_weight_best
    params_tl["objective"] = "binary"
    params_tl["metric"] = "auc"

    d_ext_tl = lgb.Dataset(X_ext_tl, label=y_ext_tl, free_raw_data=False)

    updated_model = lgb.train(
        params=params_tl,
        train_set=d_ext_tl,
        num_boost_round=num_new_trees_best,
        init_model=booster_base,
        valid_sets=[],
    )

    y_ext_test_probs_tl = updated_model.predict(X_ext_test)
    y_ext_tl_probs_tl = updated_model.predict(X_ext_tl)

    auc_tl = roc_auc_score(y_ext_test, y_ext_test_probs_tl)
    auprc_tl = average_precision_score(y_ext_test, y_ext_test_probs_tl)
    brier_tl = brier_score_loss(y_ext_test, y_ext_test_probs_tl)

    thr_tl, sens_tl, spec_tl, fpr_tl, tpr_tl, _ = youden_optimal_threshold(
        y_ext_test, y_ext_test_probs_tl
    )

    print(
        f"External TEST (TL tuned) - AUC: {auc_tl:.4f}, AUPRC: {auprc_tl:.4f}, Brier: {brier_tl:.4f} | "
        f"Youden thr: {thr_tl:.6f}, Sens: {sens_tl:.4f}, Spec: {spec_tl:.4f}"
    )

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(y_ext_tl_probs_tl.reshape(-1, 1), y_ext_tl)

    y_ext_test_probs_platt = lr.predict_proba(y_ext_test_probs_tl.reshape(-1, 1))[:, 1]

    auc_platt = roc_auc_score(y_ext_test, y_ext_test_probs_platt)
    auprc_platt = average_precision_score(y_ext_test, y_ext_test_probs_platt)
    brier_platt = brier_score_loss(y_ext_test, y_ext_test_probs_platt)

    thr_platt, sens_platt, spec_platt, fpr_platt, tpr_platt, _ = youden_optimal_threshold(
        y_ext_test, y_ext_test_probs_platt
    )

    print(
        f"External TEST (TL+Platt) - AUC: {auc_platt:.4f}, AUPRC: {auprc_platt:.4f}, Brier: {brier_platt:.4f} | "
        f"Youden thr: {thr_platt:.6f}, Sens: {sens_platt:.4f}, Spec: {spec_platt:.4f}"
    )

    roc_curves["base"].append((fpr_base, tpr_base, seed))
    roc_curves["tl"].append((fpr_tl, tpr_tl, seed))
    roc_curves["platt"].append((fpr_platt, tpr_platt, seed))

    prec_base, recall_base, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_base)
    prec_tl, recall_tl, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_tl)
    prec_platt, recall_platt, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_platt)

    pr_curves["base"].append((recall_base, prec_base, seed))
    pr_curves["tl"].append((recall_tl, prec_tl, seed))
    pr_curves["platt"].append((recall_platt, prec_platt, seed))

    records.append({
        "seed": seed,

        "auc_base": auc_base,
        "auc_tl": auc_tl,
        "auc_platt": auc_platt,

        "auprc_base": auprc_base,
        "auprc_tl": auprc_tl,
        "auprc_platt": auprc_platt,

        "brier_base": brier_base,
        "brier_tl": brier_tl,
        "brier_platt": brier_platt,

        "youden_thr_base": thr_base,
        "sens_base": sens_base,
        "spec_base": spec_base,

        "youden_thr_tl": thr_tl,
        "sens_tl": sens_tl,
        "spec_tl": spec_tl,

        "youden_thr_platt": thr_platt,
        "sens_platt": sens_platt,
        "spec_platt": spec_platt,

        "eta_factor_best": eta_factor_best,
        "num_new_trees_best": num_new_trees_best,
        "scale_pos_weight_best": scale_pos_weight_best,
    })

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("metrics_per_seed.csv", index=False)
print("\nSaved metrics_per_seed.csv")

with open("roc_pr_curves.pkl", "wb") as f:
    pickle.dump({"roc_curves": roc_curves, "pr_curves": pr_curves}, f)
print("Saved ROC/PR curves to roc_pr_curves.pkl")

