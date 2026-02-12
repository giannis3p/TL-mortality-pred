import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

def youden_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite_mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite_mask], tpr[finite_mask], thresholds[finite_mask]

    j = tpr - fpr
    best_idx = int(np.argmax(j))
    thr = float(thresholds[best_idx])
    sens = float(tpr[best_idx])
    spec = float(1.0 - fpr[best_idx])
    return thr, sens, spec

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

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

pos_df = df_internal[df_internal[target_column] == 1]
neg_df = df_internal[df_internal[target_column] == 0]

n_pos = len(pos_df)
n_neg = len(neg_df)
prev_internal = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else np.nan
print(f"\nInternal original : n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_internal:.4f}")

df_model = df_internal.sample(frac=1.0, random_state=42).reset_index(drop=True)

X_full = df_model[shared_feature_cols].copy()
y_full = df_model[target_column].copy()

X_full = X_full.select_dtypes(include=["number"])
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

df_ext_copy = df_ext.copy()
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

base_rf_params = dict(
    n_estimators=1108,
    max_depth=27,
    min_samples_split=12,
    min_samples_leaf=2,
    max_features="log2",
    bootstrap=False,
    class_weight=None,
    criterion="entropy",
    random_state=42,
    n_jobs=-1,
)

clf_base = RandomForestClassifier(**base_rf_params)
clf_base.fit(X_train, y_train)

y_test_probs_base = clf_base.predict_proba(X_test)[:, 1]
print("\n Internal TEST performance ")
print(f"Internal TEST AUC:   {roc_auc_score(y_test, y_test_probs_base):.4f}")
print(f"Internal TEST AUPRC: {average_precision_score(y_test, y_test_probs_base):.4f}")
print(f"Internal TEST Brier: {brier_score_loss(y_test, y_test_probs_base):.4f}")

thr_int, sens_int, spec_int = youden_optimal_threshold(y_test.values, y_test_probs_base)
print(f"Internal TEST Youden-optimal: thr={thr_int:.6f}, sens={sens_int:.4f}, spec={spec_int:.4f}")

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

    thr_base, sens_base, spec_base = youden_optimal_threshold(
        y_ext_test.values, y_ext_test_probs_base
    )

    print(
        f"External TEST (BASE) - AUC: {auc_base:.4f}, AUPRC: {auprc_base:.4f}, Brier: {brier_base:.4f} | "
        f"Youden thr={thr_base:.6f}, sens={sens_base:.4f}, spec={spec_base:.4f}"
    )

    def tl_objective(trial):
        add_trees = trial.suggest_int("add_trees", 50, 400)
        max_depth_tl = trial.suggest_int("max_depth_tl", 5, 40)
        min_leaf_tl = trial.suggest_int("min_samples_leaf_tl", 1, 10)

        X_tl_tr, X_tl_val, y_tl_tr, y_tl_val = train_test_split(
            X_ext_tl, y_ext_tl, test_size=0.2, random_state=seed, stratify=y_ext_tl
        )

        rf = RandomForestClassifier(**base_rf_params, warm_start=True)
        rf.fit(X_train, y_train)

        rf.set_params(
            n_estimators=base_rf_params["n_estimators"] + add_trees,
            max_depth=max_depth_tl,
            min_samples_leaf=min_leaf_tl,
        )
        rf.fit(X_tl_tr, y_tl_tr)

        y_val_probs = rf.predict_proba(X_tl_val)[:, 1]
        return -roc_auc_score(y_tl_val, y_val_probs)

    study = optuna.create_study(direction="minimize")
    study.optimize(tl_objective, n_trials=400)

    best_params_seed = study.best_params
    print(f"Best TL params for seed {seed}: {best_params_seed}")

    add_trees_best = best_params_seed["add_trees"]
    max_depth_tl_best = best_params_seed["max_depth_tl"]
    min_leaf_tl_best = best_params_seed["min_samples_leaf_tl"]

    rf_tl = RandomForestClassifier(**base_rf_params, warm_start=True)
    rf_tl.fit(X_train, y_train)

    rf_tl.set_params(
        n_estimators=base_rf_params["n_estimators"] + add_trees_best,
        max_depth=max_depth_tl_best,
        min_samples_leaf=min_leaf_tl_best,
    )
    rf_tl.fit(X_ext_tl, y_ext_tl)

    y_ext_test_probs_tl = rf_tl.predict_proba(X_ext_test)[:, 1]
    y_ext_tl_probs_tl = rf_tl.predict_proba(X_ext_tl)[:, 1]

    auc_tl = roc_auc_score(y_ext_test, y_ext_test_probs_tl)
    auprc_tl = average_precision_score(y_ext_test, y_ext_test_probs_tl)
    brier_tl = brier_score_loss(y_ext_test, y_ext_test_probs_tl)

    thr_tl, sens_tl, spec_tl = youden_optimal_threshold(
        y_ext_test.values, y_ext_test_probs_tl
    )

    print(
        f"External TEST (TL tuned) - AUC: {auc_tl:.4f}, AUPRC: {auprc_tl:.4f}, Brier: {brier_tl:.4f} | "
        f"Youden thr={thr_tl:.6f}, sens={sens_tl:.4f}, spec={spec_tl:.4f}"
    )

    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(y_ext_tl_probs_tl.reshape(-1, 1), y_ext_tl)

    y_ext_test_probs_platt = lr.predict_proba(
        y_ext_test_probs_tl.reshape(-1, 1)
    )[:, 1]

    auc_platt = roc_auc_score(y_ext_test, y_ext_test_probs_platt)
    auprc_platt = average_precision_score(y_ext_test, y_ext_test_probs_platt)
    brier_platt = brier_score_loss(y_ext_test, y_ext_test_probs_platt)

    thr_platt, sens_platt, spec_platt = youden_optimal_threshold(
        y_ext_test.values, y_ext_test_probs_platt
    )

    print(
        f"External TEST (TL+Platt) - AUC: {auc_platt:.4f}, AUPRC: {auprc_platt:.4f}, Brier: {brier_platt:.4f} | "
        f"Youden thr={thr_platt:.6f}, sens={sens_platt:.4f}, spec={spec_platt:.4f}"
    )

    records.append({
        "seed": seed,

        "auc_base": auc_base, "auc_tl": auc_tl, "auc_platt": auc_platt,
        "auprc_base": auprc_base, "auprc_tl": auprc_tl, "auprc_platt": auprc_platt,
        "brier_base": brier_base, "brier_tl": brier_tl, "brier_platt": brier_platt,

        "youden_thr_base": thr_base, "sens_base": sens_base, "spec_base": spec_base,
        "youden_thr_tl": thr_tl, "sens_tl": sens_tl, "spec_tl": spec_tl,
        "youden_thr_platt": thr_platt, "sens_platt": sens_platt, "spec_platt": spec_platt,

        "add_trees_best": add_trees_best,
        "max_depth_tl_best": max_depth_tl_best,
        "min_leaf_tl_best": min_leaf_tl_best
    })

    fpr_base, tpr_base, _ = roc_curve(y_ext_test, y_ext_test_probs_base)
    fpr_tl, tpr_tl, _ = roc_curve(y_ext_test, y_ext_test_probs_tl)
    fpr_pl, tpr_pl, _ = roc_curve(y_ext_test, y_ext_test_probs_platt)
    roc_curves["base"].append((fpr_base, tpr_base, seed))
    roc_curves["tl"].append((fpr_tl, tpr_tl, seed))
    roc_curves["platt"].append((fpr_pl, tpr_pl, seed))

    prec_base, recall_base, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_base)
    prec_tl, recall_tl, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_tl)
    prec_pl, recall_pl, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_platt)
    pr_curves["base"].append((recall_base, prec_base, seed))
    pr_curves["tl"].append((recall_tl, prec_tl, seed))
    pr_curves["platt"].append((recall_pl, prec_pl, seed))

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("rf_metrics_per_seed.csv", index=False)
print("\nSaved rf_metrics_per_seed.csv")

with open("rf_roc_pr_curves.pkl", "wb") as f:
    pickle.dump({"roc_curves": roc_curves, "pr_curves": pr_curves}, f)
print("Saved RF ROC/PR curves to rf_roc_pr_curves.pkl")
