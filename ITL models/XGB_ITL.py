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
    confusion_matrix,
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


def sensitivity_specificity(y_true, y_prob, threshold: float):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return sensitivity, specificity, (tn, fp, fn, tp)


def youdens_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)

    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]

    j = tpr - fpr
    i = np.argmax(j)

    best_thr = float(thr[i])
    best_j = float(j[i])
    best_sens = float(tpr[i])
    best_spec = float(1.0 - fpr[i])
    return best_thr, best_j, best_sens, best_spec

try:
    df
    df_ext
except NameError as exc:
    raise RuntimeError(
        "This script expects pandas DataFrames `df` (internal) and `df_ext` (external) to be defined."
    ) from exc


roc_curves = {
    "base": [],
    "tl": [],
    "platt": []
}

pr_curves = {
    "base": [],
    "tl": [],
    "platt": []
}

records = []
target_column = "OUTCOME (1=DEAD, 0=ALIVE)"

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
  n_estimators=1276,
  learning_rate=0.012883864113018737,
  max_depth=10,
  subsample=0.497031380963004,
  colsample_bytree=0.5109366122090663,
  gamma=4.541447745087957,
  min_child_weight=11.357305655375951,
  reg_alpha=0.04180492475930316,
  reg_lambda=6.447536920879301e-08,
  scale_pos_weight=2.7071640044982996,
  eval_metric="logloss",
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

y_test_probs_base = clf_base.predict_proba(X_test)[:, 1]
print("\n Internal TEST performance (BASE) ")
print(f"Internal TEST AUC:   {roc_auc_score(y_test, y_test_probs_base):.4f}")
print(f"Internal TEST AUPRC: {average_precision_score(y_test, y_test_probs_base):.4f}")
print(f"Internal TEST Brier: {brier_score_loss(y_test, y_test_probs_base):.4f}")

thr_int, j_int, sens_int, spec_int = youdens_threshold(y_test, y_test_probs_base)
print(f"Internal TEST Youden thr: {thr_int:.6f} (J={j_int:.4f})")
print(f"Internal TEST Sens@Youden: {sens_int:.4f}")
print(f"Internal TEST Spec@Youden: {spec_int:.4f}")

seeds = list(range(27, 32))

for seed in seeds:
    print(f"\n SEED {seed} ")

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext,
        y_ext,
        test_size=0.3,
        random_state=seed,
        stratify=y_ext
    )

    y_ext_test_probs_base = clf_base.predict_proba(X_ext_test)[:, 1]

    auc_base = roc_auc_score(y_ext_test, y_ext_test_probs_base)
    auprc_base = average_precision_score(y_ext_test, y_ext_test_probs_base)
    brier_base = brier_score_loss(y_ext_test, y_ext_test_probs_base)

    thr_base, j_base, sens_base, spec_base = youdens_threshold(y_ext_test, y_ext_test_probs_base)

    print(f"External TEST (BASE) - AUC: {auc_base:.4f}, AUPRC: {auprc_base:.4f}, Brier: {brier_base:.4f}")
    print(f"External TEST (BASE) - Youden thr: {thr_base:.6f} (J={j_base:.4f})")
    print(f"External TEST (BASE) - Sens@Youden: {sens_base:.4f}, Spec@Youden: {spec_base:.4f}")

    def tl_objective(trial):
        eta_factor = trial.suggest_float("eta_factor", 0.1, 0.5)
        num_new_trees = trial.suggest_int("num_new_trees", 50, 400)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.5, 5.0, log=True)

        X_tl_tr, X_tl_val, y_tl_tr, y_tl_val = train_test_split(
            X_ext_tl,
            y_ext_tl,
            test_size=0.2,
            random_state=seed,
            stratify=y_ext_tl
        )

        clf_for_tl = xgb.XGBClassifier(**base_params_gpu)
        clf_for_tl.fit(X_train, y_train)
        booster_base = clf_for_tl.get_booster()

        d_tl_tr = xgb.DMatrix(X_tl_tr, label=y_tl_tr)
        d_tl_val = xgb.DMatrix(X_tl_val, label=y_tl_val)

        params_tl = clf_for_tl.get_xgb_params()
        base_eta = params_tl.get("learning_rate", params_tl.get("eta", 0.05))

        new_eta = base_eta * eta_factor
        params_tl["learning_rate"] = new_eta
        params_tl["eta"] = new_eta
        params_tl["scale_pos_weight"] = scale_pos_weight

        updated_model = xgb.train(
            params_tl,
            d_tl_tr,
            num_boost_round=num_new_trees,
            xgb_model=booster_base
        )

        y_val_pred = updated_model.predict(d_tl_val)
        return -roc_auc_score(y_tl_val, y_val_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(tl_objective, n_trials=300)

    best_params_seed = study.best_params
    print(f"Best TL params for seed {seed}: {best_params_seed}")

    eta_factor_best = best_params_seed["eta_factor"]
    num_new_trees_best = best_params_seed["num_new_trees"]
    scale_pos_weight_best = best_params_seed["scale_pos_weight"]

    clf_for_tl = xgb.XGBClassifier(**base_params_gpu)
    clf_for_tl.fit(X_train, y_train)
    booster_base = clf_for_tl.get_booster()

    d_ext_tl = xgb.DMatrix(X_ext_tl, label=y_ext_tl)
    d_ext_test = xgb.DMatrix(X_ext_test)
    d_ext_tl_eval = xgb.DMatrix(X_ext_tl)

    params_tl = clf_for_tl.get_xgb_params()
    base_eta = params_tl.get("learning_rate", params_tl.get("eta", 0.05))

    new_eta = base_eta * eta_factor_best
    params_tl["learning_rate"] = new_eta
    params_tl["eta"] = new_eta
    params_tl["scale_pos_weight"] = scale_pos_weight_best

    updated_model = xgb.train(
        params_tl,
        d_ext_tl,
        num_boost_round=num_new_trees_best,
        xgb_model=booster_base
    )

    y_ext_test_probs_tl = updated_model.predict(d_ext_test)
    y_ext_tl_probs_tl = updated_model.predict(d_ext_tl_eval)

    auc_tl = roc_auc_score(y_ext_test, y_ext_test_probs_tl)
    auprc_tl = average_precision_score(y_ext_test, y_ext_test_probs_tl)
    brier_tl = brier_score_loss(y_ext_test, y_ext_test_probs_tl)

    thr_tl, j_tl, sens_tl, spec_tl = youdens_threshold(y_ext_test, y_ext_test_probs_tl)

    print(f"External TEST (TL tuned) - AUC: {auc_tl:.4f}, AUPRC: {auprc_tl:.4f}, Brier: {brier_tl:.4f}")
    print(f"External TEST (TL tuned) - Youden thr: {thr_tl:.6f} (J={j_tl:.4f})")
    print(f"External TEST (TL tuned) - Sens@Youden: {sens_tl:.4f}, Spec@Youden: {spec_tl:.4f}")

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(y_ext_tl_probs_tl.reshape(-1, 1), y_ext_tl)

    y_ext_test_probs_platt = lr.predict_proba(
        y_ext_test_probs_tl.reshape(-1, 1)
    )[:, 1]

    auc_platt = roc_auc_score(y_ext_test, y_ext_test_probs_platt)
    auprc_platt = average_precision_score(y_ext_test, y_ext_test_probs_platt)
    brier_platt = brier_score_loss(y_ext_test, y_ext_test_probs_platt)

    thr_platt, j_platt, sens_platt, spec_platt = youdens_threshold(y_ext_test, y_ext_test_probs_platt)

    print(f"External TEST (TL+Platt) - AUC: {auc_platt:.4f}, AUPRC: {auprc_platt:.4f}, Brier: {brier_platt:.4f}")
    print(f"External TEST (TL+Platt) - Youden thr: {thr_platt:.6f} (J={j_platt:.4f})")
    print(f"External TEST (TL+Platt) - Sens@Youden: {sens_platt:.4f}, Spec@Youden: {spec_platt:.4f}")

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
        "eta_factor_best": eta_factor_best,
        "num_new_trees_best": num_new_trees_best,
        "scale_pos_weight_best": scale_pos_weight_best,

        "youden_thr_base": thr_base,
        "youden_j_base": j_base,
        "sens_base_youden": sens_base,
        "spec_base_youden": spec_base,

        "youden_thr_tl": thr_tl,
        "youden_j_tl": j_tl,
        "sens_tl_youden": sens_tl,
        "spec_tl_youden": spec_tl,

        "youden_thr_platt": thr_platt,
        "youden_j_platt": j_platt,
        "sens_platt_youden": sens_platt,
        "spec_platt_youden": spec_platt,
    })

    fpr_base, tpr_base, _ = roc_curve(y_ext_test, y_ext_test_probs_base)
    fpr_tl, tpr_tl, _ = roc_curve(y_ext_test, y_ext_test_probs_tl)
    fpr_platt, tpr_platt, _ = roc_curve(y_ext_test, y_ext_test_probs_platt)

    roc_curves["base"].append((fpr_base, tpr_base, seed))
    roc_curves["tl"].append((fpr_tl, tpr_tl, seed))
    roc_curves["platt"].append((fpr_platt, tpr_platt, seed))

    prec_base, recall_base, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_base)
    prec_tl, recall_tl, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_tl)
    prec_platt, recall_platt, _ = precision_recall_curve(y_ext_test, y_ext_test_probs_platt)

    pr_curves["base"].append((recall_base, prec_base, seed))
    pr_curves["tl"].append((recall_tl, prec_tl, seed))
    pr_curves["platt"].append((recall_platt, prec_platt, seed))

metrics_df = pd.DataFrame(records)
metrics_df.to_csv("metrics_per_seed.csv", index=False)
print("Saved metrics_per_seed.csv")

with open("roc_pr_curves.pkl", "wb") as f:
    pickle.dump(
        {
            "roc_curves": roc_curves,
            "pr_curves": pr_curves
        },
        f
    )
print("Saved ROC/PR curves to roc_pr_curves.pkl")

