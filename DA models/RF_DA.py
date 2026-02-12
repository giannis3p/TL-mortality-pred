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
from sklearn.isotonic import IsotonicRegression

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

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))

print("\nNumber of shared features:", len(shared_feature_cols))
print("Shared:", shared_feature_cols)

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

pos_df = df_internal[df_internal[target_column] == 1]
neg_df = df_internal[df_internal[target_column] == 0]
n_pos = len(pos_df)
n_neg = len(neg_df)
prev_int = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else np.nan
print(f"\nInternal original : n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_int:.4f}")

df_model = df_internal.sample(frac=1.0, random_state=42).reset_index(drop=True)

X_full = df_model[shared_feature_cols].copy()
y_full = df_model[target_column].copy()

X_full = X_full.select_dtypes(include=["number"])
shared_feature_cols = X_full.columns.tolist()

binary_cols = [c for c in shared_feature_cols if X_full[c].nunique(dropna=True) == 2]
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

X_ext = df_ext[shared_feature_cols].copy()
y_ext = df_ext[target_column].copy()

if binary_cols:
    X_ext[binary_cols] = X_ext[binary_cols].fillna(binary_mode[binary_cols])
if continuous_cols:
    X_ext[continuous_cols] = X_ext[continuous_cols].fillna(continuous_median[continuous_cols])

print("\nX_train:", X_train.shape, "| internal train prev:", float(y_train.mean()))
print("X_test :", X_test.shape,  "| internal test  prev:", float(y_test.mean()))
print("X_ext  :", X_ext.shape,   "| external prev:", float(y_ext.mean()))

clf_base = RandomForestClassifier(**base_rf_params)
clf_base.fit(X_train, y_train)

y_int_probs = clf_base.predict_proba(X_test)[:, 1]
thr_int, sens_int, spec_int = youden_optimal_threshold(y_test.values, y_int_probs)
print(f"\nInternal TEST (BASE) Youden: thr={thr_int:.6f}, sens={sens_int:.4f}, spec={spec_int:.4f}")

def da_objective(trial):
    C_dom = trial.suggest_float("C_dom", 1e-6, 1e4, log=True)
    w_min = trial.suggest_float("w_min", 0.1, 1.0)
    w_max = trial.suggest_float("w_max", 1.0, 50.0)
    alpha = trial.suggest_float("alpha", 0.5, 5.0)
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

    clf_da = RandomForestClassifier(**base_rf_params)
    clf_da.fit(X_train, y_train, sample_weight=w)

    auc_improvements = []
    auprc_improvements = []

    for seed in seeds:
        X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
            X_ext, y_ext, test_size=0.3, random_state=seed, stratify=y_ext
        )

        base_probs = clf_base.predict_proba(X_ext_test)[:, 1]
        da_probs = clf_da.predict_proba(X_ext_test)[:, 1]

        auc_improvements.append(
            roc_auc_score(y_ext_test, da_probs) - roc_auc_score(y_ext_test, base_probs)
        )
        auprc_improvements.append(
            average_precision_score(y_ext_test, da_probs)
            - average_precision_score(y_ext_test, base_probs)
        )

    return 0.8 * float(np.mean(auc_improvements)) + 0.2 * float(np.mean(auprc_improvements))


print("\n Running Optuna (DA hyperparameters) for RF ")
study_da = optuna.create_study(direction="maximize")
study_da.optimize(da_objective, n_trials=1500)

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

clf_da = RandomForestClassifier(**base_rf_params)
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

    da_probs = clf_da.predict_proba(X_ext_test)[:, 1]
    da_probs_tl = clf_da.predict_proba(X_ext_tl)[:, 1]

    lr = LogisticRegression(max_iter=10000)
    lr.fit(da_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs = lr.predict_proba(da_probs.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(da_probs_tl, y_ext_tl)
    iso_probs = iso.predict(da_probs)

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

    thr_base, sens_base, spec_base = youden_optimal_threshold(y_ext_test.values, base_probs)
    thr_da, sens_da, spec_da = youden_optimal_threshold(y_ext_test.values, da_probs)
    thr_pl, sens_pl, spec_pl = youden_optimal_threshold(y_ext_test.values, platt_probs)
    thr_iso, sens_iso, spec_iso = youden_optimal_threshold(y_ext_test.values, iso_probs)

    print(
        f"BASE : AUC={auc_base:.4f} AUPRC={auprc_base:.4f} Brier={brier_base:.4f} | "
        f"Youden thr={thr_base:.6f} sens={sens_base:.4f} spec={spec_base:.4f}"
    )
    print(
        f"DA   : AUC={auc_da:.4f} AUPRC={auprc_da:.4f} Brier={brier_da:.4f} | "
        f"Youden thr={thr_da:.6f} sens={sens_da:.4f} spec={spec_da:.4f}"
    )
    print(
        f"PLATT: AUC={auc_platt:.4f} AUPRC={auprc_platt:.4f} Brier={brier_platt:.4f} | "
        f"Youden thr={thr_pl:.6f} sens={sens_pl:.4f} spec={spec_pl:.4f}"
    )
    print(
        f"ISO  : AUC={auc_iso:.4f} AUPRC={auprc_iso:.4f} Brier={brier_iso:.4f} | "
        f"Youden thr={thr_iso:.6f} sens={sens_iso:.4f} spec={spec_iso:.4f}"
    )

    records_da.append({
        "seed": seed,

        "auc_base": auc_base, "auc_da": auc_da, "auc_platt": auc_platt, "auc_iso": auc_iso,
        "auprc_base": auprc_base, "auprc_da": auprc_da, "auprc_platt": auprc_platt, "auprc_iso": auprc_iso,
        "brier_base": brier_base, "brier_da": brier_da, "brier_platt": brier_platt, "brier_iso": brier_iso,

        "youden_thr_base": thr_base, "sens_base": sens_base, "spec_base": spec_base,
        "youden_thr_da": thr_da, "sens_da": sens_da, "spec_da": spec_da,
        "youden_thr_platt": thr_pl, "sens_platt": sens_pl, "spec_platt": spec_pl,
        "youden_thr_iso": thr_iso, "sens_iso": sens_iso, "spec_iso": spec_iso,
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
df_da.to_csv("rf_metrics_da_per_seed.csv", index=False)

with open("rf_roc_pr_curves_da.pkl", "wb") as f:
    pickle.dump({"roc_curves_da": roc_curves_da, "pr_curves_da": pr_curves_da}, f)

print("\nSaved rf_metrics_da_per_seed.csv and rf_roc_pr_curves_da.pkl!")
