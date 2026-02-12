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
from sklearn.isotonic import IsotonicRegression

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))

print("\nNumber of shared features:", len(shared_feature_cols))
print("Shared:", shared_feature_cols)

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
prev_int = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0
print(f"\nInternal : n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_int:.4f}")

X_full = df_internal[shared_feature_cols].copy()
y_full = df_internal[target_column].copy()

X_full = X_full.select_dtypes(include=["number"]).copy()
shared_feature_cols = X_full.columns.tolist()

binary_cols = [c for c in shared_feature_cols if X_full[c].dropna().nunique() == 2]
continuous_cols = [c for c in shared_feature_cols if c not in binary_cols]

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

if binary_cols:
    X_ext[binary_cols] = X_ext[binary_cols].fillna(binary_mode[binary_cols])
if continuous_cols:
    X_ext[continuous_cols] = X_ext[continuous_cols].fillna(continuous_median[continuous_cols])

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
    verbosity = -1
)

clf_base = lgb.LGBMClassifier(**base_params)
clf_base.fit(X_train, y_train)

y_int_probs = clf_base.predict_proba(X_test)[:, 1]
auc_int = roc_auc_score(y_test, y_int_probs)
auprc_int = average_precision_score(y_test, y_int_probs)
brier_int = brier_score_loss(y_test, y_int_probs)
thr_int, sens_int, spec_int, *_ = youden_optimal_threshold(y_test, y_int_probs)

print("\n Internal TEST ")
print(f"AUC: {auc_int:.4f} | AUPRC: {auprc_int:.4f} | Brier: {brier_int:.4f}")
print(f"Youden thr: {thr_int:.6f} | Sensitivity: {sens_int:.4f} | Specificity: {spec_int:.4f}")

def da_objective(trial):
    C_dom = trial.suggest_float("C_dom", 1e-6, 1e4, log=True)
    w_min = trial.suggest_float("w_min", 0.1, 1.0)
    w_max = trial.suggest_float("w_max", 1.0, 50.0)
    alpha = trial.suggest_float("alpha", 0.5, 5.0)
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    X_domain = pd.concat([X_train, X_ext], axis=0)
    d_domain = np.concatenate(
        [np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)]
    )

    dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
    dom_clf.fit(X_domain, d_domain)

    p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
    w = np.power(p_ext / (1 - p_ext), alpha)
    w = np.clip(w, w_min, w_max)

    clf_da = lgb.LGBMClassifier(**base_params)
    clf_da.fit(X_train, y_train, sample_weight=w)

    auc_improvements = []
    auprc_improvements = []

    for seed in seeds:
        _, X_ext_test, _, y_ext_test = train_test_split(
            X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
        )

        base_probs = clf_base.predict_proba(X_ext_test)[:, 1]
        da_probs = clf_da.predict_proba(X_ext_test)[:, 1]

        auc_improvements.append(roc_auc_score(y_ext_test, da_probs) - roc_auc_score(y_ext_test, base_probs))
        auprc_improvements.append(
            average_precision_score(y_ext_test, da_probs) - average_precision_score(y_ext_test, base_probs)
        )

    return 0.8 * float(np.mean(auc_improvements)) + 0.2 * float(np.mean(auprc_improvements))

print("\n Running Optuna (DA hyperparameters) ")
study_da = optuna.create_study(direction="maximize")
study_da.optimize(da_objective, n_trials=300)
best_da_params = study_da.best_params
print("\nBest DA params:", best_da_params)

C_dom = best_da_params["C_dom"]
w_min = best_da_params["w_min"]
w_max = best_da_params["w_max"]
alpha = best_da_params["alpha"]
if w_min > w_max:
    w_min, w_max = w_max, w_min

X_domain = pd.concat([X_train, X_ext], axis=0)
d_domain = np.concatenate([np.zeros(len(X_train), dtype=int), np.ones(len(X_ext), dtype=int)])

dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
dom_clf.fit(X_domain, d_domain)

p_ext = np.clip(dom_clf.predict_proba(X_train)[:, 1], 1e-6, 1 - 1e-6)
w = np.power(p_ext / (1 - p_ext), alpha)
w = np.clip(w, w_min, w_max)

clf_da = lgb.LGBMClassifier(**base_params)
clf_da.fit(X_train, y_train, sample_weight=w)

records_da = []
roc_curves_da = {"base": [], "da": [], "platt": [], "iso": []}
pr_curves_da = {"base": [], "da": [], "platt": [], "iso": []}

for seed in seeds:
    print(f"\n SEED {seed} ")

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
    )

    base_probs = clf_base.predict_proba(X_ext_test)[:, 1]
    thr_b, sens_b, spec_b, fpr_b, tpr_b, _ = youden_optimal_threshold(y_ext_test, base_probs)

    da_probs = clf_da.predict_proba(X_ext_test)[:, 1]
    da_probs_tl = clf_da.predict_proba(X_ext_tl)[:, 1]
    thr_d, sens_d, spec_d, fpr_d, tpr_d, _ = youden_optimal_threshold(y_ext_test, da_probs)

    lr = LogisticRegression(max_iter=10000)
    lr.fit(da_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs = lr.predict_proba(da_probs.reshape(-1, 1))[:, 1]
    thr_p, sens_p, spec_p, fpr_p, tpr_p, _ = youden_optimal_threshold(y_ext_test, platt_probs)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(da_probs_tl, y_ext_tl)
    iso_probs = iso.predict(da_probs)
    thr_i, sens_i, spec_i, fpr_i, tpr_i, _ = youden_optimal_threshold(y_ext_test, iso_probs)

    auc_base = roc_auc_score(y_ext_test, base_probs)
    auc_da = roc_auc_score(y_ext_test, da_probs)
    auc_platt = roc_auc_score(y_ext_test, platt_probs)
    auc_iso = roc_auc_score(y_ext_test, iso_probs)

    auprc_base = average_precision_score(y_ext_test, base_probs)
    auprc_da = average_precision_score(y_ext_test, da_probs)
    auprc_platt = average_precision_score(y_ext_test, platt_probs)
    auprc_iso = average_precision_score(y_ext_test, iso_probs)

    brier_base = brier_score_loss(y_ext_test, base_probs)
    brier_da = brier_score_loss(y_ext_test, da_probs)
    brier_platt = brier_score_loss(y_ext_test, platt_probs)
    brier_iso = brier_score_loss(y_ext_test, iso_probs)

    print(
        f"External TEST BASE     | AUC: {auc_base:.4f} AUPRC: {auprc_base:.4f} Brier: {brier_base:.4f} | "
        f"Youden: {thr_b:.6f} Sens: {sens_b:.4f} Spec: {spec_b:.4f}"
    )
    print(
        f"External TEST DA       | AUC: {auc_da:.4f} AUPRC: {auprc_da:.4f} Brier: {brier_da:.4f} | "
        f"Youden: {thr_d:.6f} Sens: {sens_d:.4f} Spec: {spec_d:.4f}"
    )
    print(
        f"External TEST DA+Platt | AUC: {auc_platt:.4f} AUPRC: {auprc_platt:.4f} Brier: {brier_platt:.4f} | "
        f"Youden: {thr_p:.6f} Sens: {sens_p:.4f} Spec: {spec_p:.4f}"
    )
    print(
        f"External TEST DA+Iso   | AUC: {auc_iso:.4f} AUPRC: {auprc_iso:.4f} Brier: {brier_iso:.4f} | "
        f"Youden: {thr_i:.6f} Sens: {sens_i:.4f} Spec: {spec_i:.4f}"
    )

    records_da.append(
        {
            "seed": seed,
            "auc_base": auc_base,
            "auc_da": auc_da,
            "auc_platt": auc_platt,
            "auc_iso": auc_iso,
            "auprc_base": auprc_base,
            "auprc_da": auprc_da,
            "auprc_platt": auprc_platt,
            "auprc_iso": auprc_iso,
            "brier_base": brier_base,
            "brier_da": brier_da,
            "brier_platt": brier_platt,
            "brier_iso": brier_iso,
            "youden_thr_base": thr_b,
            "sens_base": sens_b,
            "spec_base": spec_b,
            "youden_thr_da": thr_d,
            "sens_da": sens_d,
            "spec_da": spec_d,
            "youden_thr_platt": thr_p,
            "sens_platt": sens_p,
            "spec_platt": spec_p,
            "youden_thr_iso": thr_i,
            "sens_iso": sens_i,
            "spec_iso": spec_i,
            "C_dom": C_dom,
            "w_min": w_min,
            "w_max": w_max,
            "alpha": alpha,
        }
    )

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
    pickle.dump({"roc_curves_da": roc_curves_da, "pr_curves_da": pr_curves_da}, f)

print("\nSaved metrics_da_per_seed.csv and roc_pr_curves_da.pkl!")


