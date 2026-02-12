import warnings
import random
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    brier_score_loss
)
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from typing import Optional

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

SAINT_PARAMS = {
    "scale_continuous": True,
    "d_model": 192,
    "n_heads": 2,
    "n_layers": 6,
    "attn_dropout": 0.1758859261861787,
    "ff_dropout": 0.10899817739727993,
    "token_dropout": 0.18259172380175237,
    "mlp_dropout": 0.2223053711267587,
    "lr": 0.0005169998340355127,
    "weight_decay": 0.0013135513765397684,
    "batch_size": 1024,
    "clip_norm": 0.8291177845522079,
}

BATCH_SIZE = int(SAINT_PARAMS["batch_size"])
PRED_BATCH = 256

BASE_LR = float(SAINT_PARAMS["lr"])
BASE_WD = float(SAINT_PARAMS["weight_decay"])
BASE_EPOCHS = 100
BASE_PATIENCE = 10
BASE_CLIP_NORM = float(SAINT_PARAMS["clip_norm"])

ALIGN_BATCH_SIZE = 256
ALIGN_LR = BASE_LR
ALIGN_WD = BASE_WD
ALIGN_EPOCHS = 100
ALIGN_PATIENCE = 10
ALIGN_CLIP_NORM = BASE_CLIP_NORM

print("Seeds:", seeds)
BEST_KIND = "mmd"
BEST_LAMBDA = 0.0009745399020374089
BEST_FREEZE = 3
BEST_SIGMA =  13.826232179369875

print(
    "\nUsing FIXED params:",
    {"align_kind": BEST_KIND, "align_lambda": BEST_LAMBDA, "freeze_epochs": BEST_FREEZE, "mmd_sigma": BEST_SIGMA},
)

if "df" not in globals() or "df_ext" not in globals():
    raise NameError("You must have DataFrames `df` (internal) and `df_ext` (external) loaded before running this cell.")

if target_column not in df.columns or target_column not in df_ext.columns:
    raise KeyError(f"Target column '{target_column}' not found in df and/or df_ext.")

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))
print("\nShared features (pre-numeric-filter):", len(shared_feature_cols))

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
print("Binary cols:", len(binary_cols), "| Continuous cols:", len(continuous_cols))

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

print("Internal train:", X_train.shape, "| External:", X_ext.shape, "| External prev:", float(np.mean(y_ext.values)))

def fit_standard_scaler(train_df: pd.DataFrame, cols):
    if not cols:
        return {}, {}
    mu = train_df[cols].mean().to_dict()
    sd = train_df[cols].std(ddof=0).replace(0.0, 1.0).to_dict()
    return mu, sd

def apply_standard_scaler(df_: pd.DataFrame, cols, mu: dict, sd: dict) -> pd.DataFrame:
    if not cols:
        return df_
    out = df_.copy()
    for c in cols:
        out[c] = (out[c].astype(np.float32) - np.float32(mu[c])) / np.float32(sd[c])
    return out

scale_cont = bool(SAINT_PARAMS["scale_continuous"])
cont_mean, cont_std = fit_standard_scaler(X_train, continuous_cols)

if scale_cont and continuous_cols:
    X_train_s = apply_standard_scaler(X_train, continuous_cols, cont_mean, cont_std)
    X_test_s  = apply_standard_scaler(X_test,  continuous_cols, cont_mean, cont_std)
    X_ext_s   = apply_standard_scaler(X_ext,   continuous_cols, cont_mean, cont_std)
else:
    X_train_s, X_test_s, X_ext_s = X_train, X_test, X_ext

class SAINTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, ff_dropout: float, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(attn_dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ff(self.ln2(x))
        return x

class SAINTNumericClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        attn_dropout: float,
        ff_dropout: float,
        token_dropout: float,
        mlp_dropout: float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_features = n_features
        self.d_model = d_model

        self.feat_scale = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.token_dropout = nn.Dropout(token_dropout)

        self.blocks = nn.ModuleList(
            [
                SAINTBlock(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
                for _ in range(n_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(d_model, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError(f"Expected (B,{self.n_features}) got {tuple(x.shape)}")

        xt = x.unsqueeze(-1) * self.feat_scale.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        xt = self.token_dropout(xt)
        for blk in self.blocks:
            xt = blk(xt)
        return xt.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(x)
        return self.head(pooled).squeeze(-1)

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

def train_saint(
    X_tr_np, y_tr_np,
    X_val_np, y_val_np,
    init_state_dict=None,
    lr=1e-3, wd=1e-4,
    max_epochs=200, patience=20,
    clip_norm=0.0,
):
    n_features = X_tr_np.shape[1]
    model = SAINTNumericClassifier(
        n_features=n_features,
        d_model=int(SAINT_PARAMS["d_model"]),
        n_heads=int(SAINT_PARAMS["n_heads"]),
        n_layers=int(SAINT_PARAMS["n_layers"]),
        attn_dropout=float(SAINT_PARAMS["attn_dropout"]),
        ff_dropout=float(SAINT_PARAMS["ff_dropout"]),
        token_dropout=float(SAINT_PARAMS["token_dropout"]),
        mlp_dropout=float(SAINT_PARAMS["mlp_dropout"]),
    ).to(device)

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce_none = nn.BCEWithLogitsLoss(reduction="none")

    train_ds = TensorDataset(
        torch.from_numpy(X_tr_np.astype(np.float32)),
        torch.from_numpy(y_tr_np.astype(np.float32)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0, drop_last=False
    )

    val_ds = TensorDataset(
        torch.from_numpy(X_val_np.astype(np.float32)),
        torch.from_numpy(y_val_np.astype(np.float32)),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=0, drop_last=False
    )

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for _ep in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce_none(logits, yb).mean()

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
    return model

def youden_optimal_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.nanargmax(j))
    thr_star = float(thr[best_idx])
    sens_star = float(tpr[best_idx])
    spec_star = float(1.0 - fpr[best_idx])
    return thr_star, sens_star, spec_star

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).t()
    return (x2 + y2 - 2.0 * (x @ y.t())).clamp_min(0.0)

def mmd_rbf_loss(src: torch.Tensor, tgt: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    if sigma is None:
        with torch.no_grad():
            all_ = torch.cat([src, tgt], dim=0)
            dists = _pairwise_sq_dists(all_, all_)
            med = torch.median(dists[dists > 0])
            sigma2 = med.clamp_min(1e-6)
    else:
        sigma2 = torch.tensor(float(sigma) ** 2, device=src.device).clamp_min(1e-6)

    k_ss = torch.exp(-_pairwise_sq_dists(src, src) / (2.0 * sigma2))
    k_tt = torch.exp(-_pairwise_sq_dists(tgt, tgt) / (2.0 * sigma2))
    k_st = torch.exp(-_pairwise_sq_dists(src, tgt) / (2.0 * sigma2))
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

def _set_freeze_feature_extractor(model: SAINTNumericClassifier, freeze: bool) -> None:
    for p in model.blocks.parameters():
        p.requires_grad = not freeze
    model.feat_scale.requires_grad = not freeze
    model.feat_bias.requires_grad = not freeze
    for p in model.head.parameters():
        p.requires_grad = True

def train_saint_align(
    Xs_tr_np, ys_tr_np,
    Xs_val_np, ys_val_np,
    Xt_unl_np,
    init_state_dict,
    *,
    lr: float,
    wd: float,
    max_epochs: int,
    patience: int,
    clip_norm: float,
    batch_size: int,
    align_lambda: float,
    mmd_sigma: Optional[float],
    freeze_epochs: int = 0,
):
    if Xt_unl_np is None or len(Xt_unl_np) == 0:
        raise ValueError("Xt_unl_np (external TL) is empty.")

    n_features = Xs_tr_np.shape[1]
    model = SAINTNumericClassifier(
        n_features=n_features,
        d_model=int(SAINT_PARAMS["d_model"]),
        n_heads=int(SAINT_PARAMS["n_heads"]),
        n_layers=int(SAINT_PARAMS["n_layers"]),
        attn_dropout=float(SAINT_PARAMS["attn_dropout"]),
        ff_dropout=float(SAINT_PARAMS["ff_dropout"]),
        token_dropout=float(SAINT_PARAMS["token_dropout"]),
        mlp_dropout=float(SAINT_PARAMS["mlp_dropout"]),
    ).to(device)
    model.load_state_dict(init_state_dict)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    src_ds = TensorDataset(
        torch.from_numpy(Xs_tr_np.astype(np.float32)),
        torch.from_numpy(ys_tr_np.astype(np.float32)),
    )
    tgt_ds = TensorDataset(torch.from_numpy(Xt_unl_np.astype(np.float32)))

    bs = int(min(batch_size, len(src_ds), len(tgt_ds)))
    if bs <= 1:
        raise ValueError("Batch size too small after min(...).")

    src_loader = DataLoader(src_ds, batch_size=bs, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=0, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=bs, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=0, drop_last=True)

    val_ds = TensorDataset(
        torch.from_numpy(Xs_val_np.astype(np.float32)),
        torch.from_numpy(ys_val_np.astype(np.float32)),
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda"), num_workers=0, drop_last=False)

    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_state = None
    no_imp = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        _set_freeze_feature_extractor(model, freeze=(freeze_epochs > 0 and ep <= freeze_epochs))

        tgt_iter = iter(tgt_loader)

        for xb_s, yb_s in src_loader:
            try:
                (xb_t,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (xb_t,) = next(tgt_iter)

            xb_s = xb_s.to(device, non_blocking=True)
            yb_s = yb_s.to(device, non_blocking=True)
            xb_t = xb_t.to(device, non_blocking=True)

            xb = torch.cat([xb_s, xb_t], dim=0)

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                pooled = model.forward_features(xb)
                logits_all = model.head(pooled).squeeze(-1)
                logits_s = logits_all[:bs]

                loss_task = bce(logits_s, yb_s)

                feat_s = pooled[:bs]
                feat_t = pooled[bs:]

                loss_align = mmd_rbf_loss(feat_s, feat_t, sigma=mmd_sigma)
                loss = loss_task + float(align_lambda) * loss_align

            if torch.isnan(loss) or torch.isinf(loss):
                break

            scaler.scale(loss).backward()

            if clip_norm and clip_norm > 0.0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            scaler.step(opt)
            scaler.update()

        model.eval()
        va_sum, n_va = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                va_sum += bce(model(xb), yb).item() * xb.size(0)
                n_va += xb.size(0)
        va_loss = va_sum / max(n_va, 1)

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
    return model

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

print("\n Training BASE SAINT ")
base_model = train_saint(
    X_tr_np, y_tr_np,
    X_val_np, y_val_np,
    init_state_dict=None,
    lr=BASE_LR, wd=BASE_WD,
    max_epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    clip_norm=BASE_CLIP_NORM,
)
base_state_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
base_internal_probs = predict_proba(base_model, X_test_np)
print("Internal TEST AUC (BASE):", float(roc_auc_score(y_test_np, base_internal_probs)))

X_ext_np_all = X_ext_s.values.astype(np.float32)
y_ext_np_all = y_ext.values.astype(np.float32)

records = []
roc_curves = {"base": [], "align": [], "platt": [], "iso": []}
pr_curves  = {"base": [], "align": [], "platt": [], "iso": []}

for seed in seeds:
    print(f"\n SEED {seed} (FIXED {BEST_KIND}) ")
    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext_np_all, y_ext_np_all, test_size=0.3, random_state=seed, stratify=y_ext_np_all
    )

    base_probs = predict_proba(base_model, X_ext_test)

    align_model = train_saint_align(
        Xs_tr_np=X_tr_np, ys_tr_np=y_tr_np,
        Xs_val_np=X_val_np, ys_val_np=y_val_np,
        Xt_unl_np=X_ext_tl,
        init_state_dict=base_state_cpu,
        lr=ALIGN_LR, wd=ALIGN_WD,
        max_epochs=ALIGN_EPOCHS, patience=ALIGN_PATIENCE,
        clip_norm=ALIGN_CLIP_NORM,
        batch_size=ALIGN_BATCH_SIZE,
        align_lambda=BEST_LAMBDA,
        mmd_sigma=BEST_SIGMA,
        freeze_epochs=BEST_FREEZE,
    )

    align_probs = predict_proba(align_model, X_ext_test)
    align_probs_tl = predict_proba(align_model, X_ext_tl)

    lr_cal = LogisticRegression(max_iter=10000)
    lr_cal.fit(align_probs_tl.reshape(-1, 1), y_ext_tl)
    platt_probs = lr_cal.predict_proba(align_probs.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(align_probs_tl, y_ext_tl)
    iso_probs = iso.predict(align_probs)

    auc_base  = roc_auc_score(y_ext_test, base_probs)
    auc_align = roc_auc_score(y_ext_test, align_probs)
    auc_platt = roc_auc_score(y_ext_test, platt_probs)
    auc_iso   = roc_auc_score(y_ext_test, iso_probs)

    auprc_base  = average_precision_score(y_ext_test, base_probs)
    auprc_align = average_precision_score(y_ext_test, align_probs)
    auprc_platt = average_precision_score(y_ext_test, platt_probs)
    auprc_iso   = average_precision_score(y_ext_test, iso_probs)

    brier_base  = brier_score_loss(y_ext_test, base_probs)
    brier_align = brier_score_loss(y_ext_test, align_probs)
    brier_platt = brier_score_loss(y_ext_test, platt_probs)
    brier_iso   = brier_score_loss(y_ext_test, iso_probs)

    thr_b, sens_b, spec_b = youden_optimal_threshold(y_ext_test, base_probs)
    thr_a, sens_a, spec_a = youden_optimal_threshold(y_ext_test, align_probs)
    thr_p, sens_p, spec_p = youden_optimal_threshold(y_ext_test, platt_probs)
    thr_i, sens_i, spec_i = youden_optimal_threshold(y_ext_test, iso_probs)

    records.append({
        "seed": seed,
        "auc_base": float(auc_base), "auc_align": float(auc_align), "auc_platt": float(auc_platt), "auc_iso": float(auc_iso),
        "auprc_base": float(auprc_base), "auprc_align": float(auprc_align), "auprc_platt": float(auprc_platt), "auprc_iso": float(auprc_iso),
        "brier_base": float(brier_base), "brier_align": float(brier_align), "brier_platt": float(brier_platt), "brier_iso": float(brier_iso),

        "thr_youden_base": thr_b, "sens_youden_base": sens_b, "spec_youden_base": spec_b,
        "thr_youden_align": thr_a, "sens_youden_align": sens_a, "spec_youden_align": spec_a,
        "thr_youden_platt": thr_p, "sens_youden_platt": sens_p, "spec_youden_platt": spec_p,
        "thr_youden_iso": thr_i, "sens_youden_iso": sens_i, "spec_youden_iso": spec_i,

        "best_kind": BEST_KIND,
        "best_lambda": float(BEST_LAMBDA),
        "best_sigma": float(BEST_SIGMA),
        "best_freeze_epochs": int(BEST_FREEZE),
    })

    fpr_b, tpr_b, _ = roc_curve(y_ext_test, base_probs)
    fpr_a, tpr_a, _ = roc_curve(y_ext_test, align_probs)
    fpr_p, tpr_p, _ = roc_curve(y_ext_test, platt_probs)
    fpr_i, tpr_i, _ = roc_curve(y_ext_test, iso_probs)
    roc_curves["base"].append((fpr_b, tpr_b, seed))
    roc_curves["align"].append((fpr_a, tpr_a, seed))
    roc_curves["platt"].append((fpr_p, tpr_p, seed))
    roc_curves["iso"].append((fpr_i, tpr_i, seed))

    rec_b, prec_b, _ = precision_recall_curve(y_ext_test, base_probs)
    rec_a, prec_a, _ = precision_recall_curve(y_ext_test, align_probs)
    rec_p, prec_p, _ = precision_recall_curve(y_ext_test, platt_probs)
    rec_i, prec_i, _ = precision_recall_curve(y_ext_test, iso_probs)
    pr_curves["base"].append((rec_b, prec_b, seed))
    pr_curves["align"].append((rec_a, prec_a, seed))
    pr_curves["platt"].append((rec_p, prec_p, seed))
    pr_curves["iso"].append((rec_i, prec_i, seed))

    if device.type == "cuda":
        torch.cuda.empty_cache()

df_metrics = pd.DataFrame(records)
df_metrics.to_csv("metrics_mmd_optuna_best_per_seed_saint.csv", index=False)
print("\nSaved metrics_mmd_optuna_best_per_seed_saint.csv")

with open("roc_pr_curves_mmd_optuna_best_saint.pkl", "wb") as f:
    pickle.dump(
        {
            "roc_curves": roc_curves,
            "pr_curves": pr_curves,
            "fixed_best_params": {
                "align_kind": BEST_KIND,
                "align_lambda": BEST_LAMBDA,
                "freeze_epochs": BEST_FREEZE,
                "mmd_sigma": BEST_SIGMA,
            },
        },
        f,
    )
print("Saved roc_pr_curves_mmd_optuna_best_saint.pkl")