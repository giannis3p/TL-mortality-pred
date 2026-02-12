import warnings
import random
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
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
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.isotonic import IsotonicRegression

file_path = "../MIMIC-IV.csv"
file_path2 = "../prospective.csv"
df = pd.read_csv(file_path)
df_ext = pd.read_csv(file_path2)

SEED_GLOBAL = 42
random.seed(SEED_GLOBAL)
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_GLOBAL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

target_column = "OUTCOME (1=DEAD, 0=ALIVE)"
seeds = list(range(27, 32))

NODE_PARAMS = {
    "scale_continuous": True,
    "num_layers": 4,
    "num_trees": 406,
    "depth": 2,
    "colsample": 0.8823354671820333,
    "tau": 0.2018004745467103,
    "tree_dropout": 0.3040306357224089,
    "lr":  0.022824553455909717,
    "weight_decay": 0.006954138730498161,
    "batch_size": 512,
    "clip_norm": 0.5840870642006418,
}

BATCH_SIZE = int(NODE_PARAMS["batch_size"])
PRED_BATCH = 256

BASE_LR = float(NODE_PARAMS["lr"])
BASE_WD = float(NODE_PARAMS["weight_decay"])
BASE_EPOCHS = 200
BASE_PATIENCE = 20
BASE_CLIP_NORM = float(NODE_PARAMS["clip_norm"])

DA_LR = BASE_LR
DA_WD = BASE_WD
DA_EPOCHS = BASE_EPOCHS
DA_PATIENCE = BASE_PATIENCE
DA_CLIP_NORM = BASE_CLIP_NORM

N_TRIALS_DA = 500
DA_WARMUP_STEPS = 3

print("Seeds:", seeds)

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))
print("\nNumber of shared features:", len(shared_feature_cols))

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

n_pos = int((df_internal[target_column] == 1).sum())
n_neg = int((df_internal[target_column] == 0).sum())
prev_internal = n_pos / max(n_pos + n_neg, 1)
print(f"Internal ORIGINAL: n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_internal:.4f}")

X_full = df_internal[shared_feature_cols].copy()
y_full = df_internal[target_column].astype(np.float32).copy()

X_full = X_full.select_dtypes(include=["number"]).copy()
shared_feature_cols = X_full.columns.tolist()
print("Numeric shared features used:", len(shared_feature_cols))

X_full = X_full.replace([np.inf, -np.inf], np.nan)
all_nan_cols = X_full.columns[X_full.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN columns (internal): {len(all_nan_cols)}")
    X_full = X_full.drop(columns=all_nan_cols)
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
    X_full, y_full, test_size=0.2, random_state=SEED_GLOBAL, stratify=y_full
)

X_ext = df_ext[shared_feature_cols].copy()
y_ext = df_ext[target_column].astype(np.float32).copy()

X_ext = X_ext.replace([np.inf, -np.inf], np.nan)
if binary_cols:
    X_ext[binary_cols] = X_ext[binary_cols].fillna(binary_mode[binary_cols])
if continuous_cols:
    X_ext[continuous_cols] = X_ext[continuous_cols].fillna(continuous_median[continuous_cols])

print("Internal train:", X_train.shape, "External:", X_ext.shape, "External prev:", float(y_ext.mean()))

def fit_standard_scaler(train_df: pd.DataFrame, cols: list[str]) -> tuple[dict, dict]:
    if not cols:
        return {}, {}
    mu = train_df[cols].mean().to_dict()
    sd = train_df[cols].std(ddof=0).replace(0.0, 1.0).to_dict()
    return mu, sd

def apply_standard_scaler(df_: pd.DataFrame, cols: list[str], mu: dict, sd: dict) -> pd.DataFrame:
    if not cols:
        return df_
    out = df_.copy()
    for c in cols:
        out[c] = (out[c].astype(np.float32) - np.float32(mu[c])) / np.float32(sd[c])
    return out

scale_cont = bool(NODE_PARAMS["scale_continuous"])
cont_mean, cont_std = fit_standard_scaler(X_train, continuous_cols)
if scale_cont and continuous_cols:
    X_train_s = apply_standard_scaler(X_train, continuous_cols, cont_mean, cont_std)
    X_test_s  = apply_standard_scaler(X_test,  continuous_cols, cont_mean, cont_std)
    X_ext_s   = apply_standard_scaler(X_ext,   continuous_cols, cont_mean, cont_std)
else:
    X_train_s, X_test_s, X_ext_s = X_train, X_test, X_ext

class ODSTLayer(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_trees: int,
        depth: int,
        colsample: float = 1.0,
        tau: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.n_features = n_features
        self.num_trees = num_trees
        self.depth = depth
        self.colsample = float(colsample)
        self.tau = float(tau)

        self.feature_logits = nn.Parameter(torch.empty(num_trees, depth, n_features))
        nn.init.xavier_uniform_(self.feature_logits)

        self.thresholds = nn.Parameter(torch.zeros(num_trees, depth))

        self.leaf_responses = nn.Parameter(torch.zeros(num_trees, 2 ** depth))
        nn.init.normal_(self.leaf_responses, mean=0.0, std=0.01)

        self.out_dropout = nn.Dropout(dropout)

        if self.colsample < 1.0:
            k = max(1, int(round(self.colsample * n_features)))
            idx = torch.randperm(n_features)[:k]
            mask = torch.zeros(n_features, dtype=torch.bool)
            mask[idx] = True
            self.register_buffer("feature_mask", mask)
        else:
            self.register_buffer("feature_mask", torch.ones(n_features, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        if F != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F}.")

        logits = self.feature_logits
        if self.feature_mask is not None and self.feature_mask.sum().item() < self.n_features:
            masked_logits = logits.clone()
            masked_logits[..., ~self.feature_mask] = -1e9
            logits = masked_logits

        w = torch.softmax(logits, dim=-1)
        x_sel = torch.einsum("bf,tdf->btd", x, w)

        tau = max(self.tau, 1e-4)
        p_right = torch.sigmoid((x_sel - self.thresholds.unsqueeze(0)) / tau)

        probs = x.new_ones((B, self.num_trees, 1))
        for d in range(self.depth):
            pr = p_right[:, :, d:d + 1]
            probs = torch.cat([probs * (1.0 - pr), probs * pr], dim=-1)

        out = torch.sum(probs * self.leaf_responses.unsqueeze(0), dim=-1)
        return self.out_dropout(out)

class NODEBinaryClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        num_layers: int,
        num_trees: int,
        depth: int,
        colsample: float,
        tau: float,
        tree_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ODSTLayer(
                n_features=n_features,
                num_trees=num_trees,
                depth=depth,
                colsample=colsample,
                tau=tau,
                dropout=tree_dropout,
            )
            for _ in range(num_layers)
        ])
        self.readouts = nn.ModuleList([nn.Linear(num_trees, 1) for _ in range(num_layers)])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.bias.expand(x.size(0))
        for layer, ro in zip(self.layers, self.readouts):
            h = layer(x)
            logit = logit + ro(h).squeeze(-1)
        return logit

def predict_proba(model, X_np, batch_size=PRED_BATCH):
    ds = TensorDataset(torch.from_numpy(X_np.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0)
    model.eval()
    logits_all = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits_all.append(model(xb).detach().cpu().numpy())
    logits = np.concatenate(logits_all)
    return 1.0 / (1.0 + np.exp(-logits))

def train_node(
    X_tr_np, y_tr_np,
    X_val_np, y_val_np,
    sample_w_np=None,
    init_state_dict=None,
    lr=1e-3, wd=1e-4,
    max_epochs=200, patience=20,
    clip_norm=0.0,
):
    n_features = X_tr_np.shape[1]
    model = NODEBinaryClassifier(
        n_features=n_features,
        num_layers=int(NODE_PARAMS["num_layers"]),
        num_trees=int(NODE_PARAMS["num_trees"]),
        depth=int(NODE_PARAMS["depth"]),
        colsample=float(NODE_PARAMS["colsample"]),
        tau=float(NODE_PARAMS["tau"]),
        tree_dropout=float(NODE_PARAMS["tree_dropout"]),
    ).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    if sample_w_np is None:
        train_ds = TensorDataset(
            torch.from_numpy(X_tr_np.astype(np.float32)),
            torch.from_numpy(y_tr_np.astype(np.float32)),
        )
    else:
        train_ds = TensorDataset(
            torch.from_numpy(X_tr_np.astype(np.float32)),
            torch.from_numpy(y_tr_np.astype(np.float32)),
            torch.from_numpy(sample_w_np.astype(np.float32)),
        )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=0
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_np.astype(np.float32)),
        torch.from_numpy(y_val_np.astype(np.float32)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0
    )

    best_val = np.inf
    best_state = None
    no_imp = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            if sample_w_np is None:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss_vec = bce_none(logits, yb)

            if wb is not None:
                wb = wb.to(device, non_blocking=True)
                loss = (loss_vec * wb).mean()
            else:
                loss = loss_vec.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                break

            loss.backward()
            if clip_norm and clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            opt.step()

        model.eval()
        va_loss_sum, n_va = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                va_loss_sum += bce_none(logits, yb).mean().item() * xb.size(0)
                n_va += xb.size(0)
        va_loss = va_loss_sum / max(n_va, 1)

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model

def youden_optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.nanargmax(j))
    thr_star = float(thr[best_idx])
    sens_star = float(tpr[best_idx])
    spec_star = float(1.0 - fpr[best_idx])
    return thr_star, sens_star, spec_star

X_train_np = X_train_s.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
X_test_np  = X_test_s.values.astype(np.float32)
y_test_np  = y_test.values.astype(np.float32)

idx_all = np.arange(len(X_train_np))
idx_tr, idx_val, _, _ = train_test_split(
    idx_all, y_train_np, test_size=0.1, random_state=SEED_GLOBAL, stratify=y_train_np
)
X_tr_np, y_tr_np = X_train_np[idx_tr], y_train_np[idx_tr]
X_val_np, y_val_np = X_train_np[idx_val], y_train_np[idx_val]

print("\n Training BASE NODE ")
base_model = train_node(
    X_tr_np, y_tr_np,
    X_val_np, y_val_np,
    sample_w_np=None,
    init_state_dict=None,
    lr=BASE_LR, wd=BASE_WD,
    max_epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    clip_norm=BASE_CLIP_NORM,
)

base_state_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

base_internal_probs = predict_proba(base_model, X_test_np)
print("Internal TEST AUC (BASE):", roc_auc_score(y_test_np, base_internal_probs))

X_ext_np_all = X_ext_s.values.astype(np.float32)
y_ext_np_all = y_ext.values.astype(np.float32)

def da_objective(trial):
    C_dom = trial.suggest_float("C_dom", 1e-6, 1e4, log=True)
    w_min = trial.suggest_float("w_min", 0.3, 1.0)
    w_max = trial.suggest_float("w_max", 1.0, 15.0)
    alpha = trial.suggest_float("alpha", 0.1, 1.5)
    if w_min > w_max:
        w_min, w_max = w_max, w_min

    X_domain = pd.concat([X_train_s, X_ext_s], axis=0)
    d_domain = np.concatenate([np.zeros(len(X_train_s), dtype=int), np.ones(len(X_ext_s), dtype=int)])

    dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
    dom_clf.fit(X_domain, d_domain)

    p_ext = np.clip(dom_clf.predict_proba(X_train_s)[:, 1], 1e-6, 1 - 1e-6)
    w_all = np.power(p_ext / (1 - p_ext), alpha)
    w_all = np.clip(w_all, w_min, w_max).astype(np.float32)

    w_tr = w_all[idx_tr]

    da_model = train_node(
        X_tr_np, y_tr_np,
        X_val_np, y_val_np,
        sample_w_np=w_tr,
        init_state_dict=base_state_cpu,
        lr=DA_LR, wd=DA_WD,
        max_epochs=DA_EPOCHS, patience=DA_PATIENCE,
        clip_norm=DA_CLIP_NORM,
    )

    auc_improvements = []
    auprc_improvements = []

    for seed in seeds:
        _, X_ext_test_s, _, y_ext_test_s = train_test_split(
            X_ext_np_all, y_ext_np_all, test_size=0.3, random_state=seed, stratify=y_ext_np_all
        )

        base_probs = predict_proba(base_model, X_ext_test_s)
        da_probs   = predict_proba(da_model,   X_ext_test_s)

        auc_improvements.append(roc_auc_score(y_ext_test_s, da_probs) - roc_auc_score(y_ext_test_s, base_probs))
        auprc_improvements.append(
            average_precision_score(y_ext_test_s, da_probs) - average_precision_score(y_ext_test_s, base_probs)
        )

    return 0.8 * float(np.mean(auc_improvements)) + 0.2 * float(np.mean(auprc_improvements))

print("\n Running Optuna ")
study_da = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=SEED_GLOBAL),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=DA_WARMUP_STEPS),
)
study_da.optimize(da_objective, n_trials=N_TRIALS_DA, n_jobs=1)
best_da_params = study_da.best_params
print("\nBest DA params:", best_da_params)

C_dom = best_da_params["C_dom"]
w_min = best_da_params["w_min"]
w_max = best_da_params["w_max"]
alpha = best_da_params["alpha"]
if w_min > w_max:
    w_min, w_max = w_max, w_min

X_domain = pd.concat([X_train_s, X_ext_s], axis=0)
d_domain = np.concatenate([np.zeros(len(X_train_s), dtype=int), np.ones(len(X_ext_s), dtype=int)])

dom_clf = LogisticRegression(max_iter=10000, C=C_dom)
dom_clf.fit(X_domain, d_domain)

p_ext = np.clip(dom_clf.predict_proba(X_train_s)[:, 1], 1e-6, 1 - 1e-6)
w_all = np.power(p_ext / (1 - p_ext), alpha)
w_all = np.clip(w_all, w_min, w_max).astype(np.float32)
w_tr = w_all[idx_tr]

print("\n Training final DA NODE ")
da_model_final = train_node(
    X_tr_np, y_tr_np,
    X_val_np, y_val_np,
    sample_w_np=w_tr,
    init_state_dict=base_state_cpu,
    lr=DA_LR, wd=DA_WD,
    max_epochs=DA_EPOCHS, patience=DA_PATIENCE,
    clip_norm=DA_CLIP_NORM,
)

records_da = []
roc_curves_da = {"base": [], "da": [], "platt": [], "iso": []}
pr_curves_da  = {"base": [], "da": [], "platt": [], "iso": []}

for seed in seeds:
    print(f"\n SEED {seed} ")

    X_ext_tl, X_ext_test_s, y_ext_tl, y_ext_test_s = train_test_split(
        X_ext_np_all, y_ext_np_all, test_size=0.3, random_state=seed, stratify=y_ext_np_all
    )

    base_probs = predict_proba(base_model, X_ext_test_s)

    da_probs = predict_proba(da_model_final, X_ext_test_s)
    da_probs_tl = predict_proba(da_model_final, X_ext_tl)

    lr_cal = LogisticRegression(max_iter=10000)
    lr_cal.fit(da_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs = lr_cal.predict_proba(da_probs.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(da_probs_tl, y_ext_tl)
    iso_probs = iso.predict(da_probs)

    auc_base = roc_auc_score(y_ext_test_s, base_probs)
    auc_da   = roc_auc_score(y_ext_test_s, da_probs)
    auc_platt= roc_auc_score(y_ext_test_s, platt_probs)
    auc_iso  = roc_auc_score(y_ext_test_s, iso_probs)

    auprc_base = average_precision_score(y_ext_test_s, base_probs)
    auprc_da   = average_precision_score(y_ext_test_s, da_probs)
    auprc_platt= average_precision_score(y_ext_test_s, platt_probs)
    auprc_iso  = average_precision_score(y_ext_test_s, iso_probs)

    brier_base = brier_score_loss(y_ext_test_s, base_probs)
    brier_da   = brier_score_loss(y_ext_test_s, da_probs)
    brier_platt= brier_score_loss(y_ext_test_s, platt_probs)
    brier_iso  = brier_score_loss(y_ext_test_s, iso_probs)

    thr_b, sens_b, spec_b = youden_optimal_threshold(y_ext_test_s, base_probs)
    thr_d, sens_d, spec_d = youden_optimal_threshold(y_ext_test_s, da_probs)
    thr_p, sens_p, spec_p = youden_optimal_threshold(y_ext_test_s, platt_probs)
    thr_i, sens_i, spec_i = youden_optimal_threshold(y_ext_test_s, iso_probs)

    records_da.append({
        "seed": seed,
        "auc_base": auc_base, "auc_da": auc_da, "auc_platt": auc_platt, "auc_iso": auc_iso,
        "auprc_base": auprc_base, "auprc_da": auprc_da, "auprc_platt": auprc_platt, "auprc_iso": auprc_iso,
        "brier_base": brier_base, "brier_da": brier_da, "brier_platt": brier_platt, "brier_iso": brier_iso,

        "thr_youden_base": thr_b,   "sens_youden_base": sens_b,   "spec_youden_base": spec_b,
        "thr_youden_da": thr_d,     "sens_youden_da": sens_d,     "spec_youden_da": spec_d,
        "thr_youden_platt": thr_p,  "sens_youden_platt": sens_p,  "spec_youden_platt": spec_p,
        "thr_youden_iso": thr_i,    "sens_youden_iso": sens_i,    "spec_youden_iso": spec_i,

        "da_C_dom": float(C_dom), "da_w_min": float(w_min), "da_w_max": float(w_max), "da_alpha": float(alpha),
        "scale_continuous": int(scale_cont),
        "node_num_layers": int(NODE_PARAMS["num_layers"]),
        "node_num_trees": int(NODE_PARAMS["num_trees"]),
        "node_depth": int(NODE_PARAMS["depth"]),
        "node_tau": float(NODE_PARAMS["tau"]),
        "node_colsample": float(NODE_PARAMS["colsample"]),
        "node_tree_dropout": float(NODE_PARAMS["tree_dropout"]),
    })

    fpr_b, tpr_b, _ = roc_curve(y_ext_test_s, base_probs)
    fpr_d, tpr_d, _ = roc_curve(y_ext_test_s, da_probs)
    fpr_p, tpr_p, _ = roc_curve(y_ext_test_s, platt_probs)
    fpr_i, tpr_i, _ = roc_curve(y_ext_test_s, iso_probs)

    roc_curves_da["base"].append((fpr_b, tpr_b, seed))
    roc_curves_da["da"].append((fpr_d, tpr_d, seed))
    roc_curves_da["platt"].append((fpr_p, tpr_p, seed))
    roc_curves_da["iso"].append((fpr_i, tpr_i, seed))

    prec_b, rec_b, _ = precision_recall_curve(y_ext_test_s, base_probs)
    prec_d, rec_d, _ = precision_recall_curve(y_ext_test_s, da_probs)
    prec_p, rec_p, _ = precision_recall_curve(y_ext_test_s, platt_probs)
    prec_i, rec_i, _ = precision_recall_curve(y_ext_test_s, iso_probs)

    pr_curves_da["base"].append((rec_b, prec_b, seed))
    pr_curves_da["da"].append((rec_d, prec_d, seed))
    pr_curves_da["platt"].append((rec_p, prec_p, seed))
    pr_curves_da["iso"].append((rec_i, prec_i, seed))

df_da = pd.DataFrame(records_da)
df_da.to_csv("metrics_da_per_seed_node.csv", index=False)
print("\nSaved metrics_da_per_seed_node.csv")

with open("roc_pr_curves_da_node.pkl", "wb") as f:
    pickle.dump({"roc_curves_da": roc_curves_da, "pr_curves_da": pr_curves_da}, f)
print("Saved roc_pr_curves_da_node.pkl")