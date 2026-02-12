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
    confusion_matrix,
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

BATCH_SIZE_BASE = int(SAINT_PARAMS["batch_size"])
BATCH_SIZE_PRED = 256

BASE_EPOCHS = 400
BASE_PATIENCE = 30

N_TRIALS_TL = 600
TL_MAX_EPOCHS = 200
TL_PATIENCE = 20
TL_MIN_EPOCHS = 100

TL_ES_FRAC = 0.10

HEAD_ONLY = False

seeds = list(range(27, 32))
print("Seeds:", seeds)

roc_curves = {"base": [], "tl": [], "platt": []}
pr_curves = {"base": [], "tl": [], "platt": []}
records = []

def youden_sens_spec(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    j = int(np.argmax(youden))
    best_thr = float(thr[j])

    y_pred = (y_prob >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return best_thr, float(sens), float(spec), float(youden[j])

print("Target in df:", target_column in df.columns)
print("Target in df_ext:", target_column in df_ext.columns)

features_df = [c for c in df.columns if c != target_column]
features_df_ext = [c for c in df_ext.columns if c != target_column]
shared_feature_cols = sorted(list(set(features_df).intersection(features_df_ext)))
print("\n# shared features:", len(shared_feature_cols))

df_internal = df[shared_feature_cols + [target_column]].dropna(subset=[target_column]).copy()

pos_df = df_internal[df_internal[target_column] == 1]
neg_df = df_internal[df_internal[target_column] == 0]
n_pos = len(pos_df)
n_neg = len(neg_df)
prev_internal = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0
print(
    f"Internal original : n={len(df_internal)}, pos={n_pos}, neg={n_neg}, prev={prev_internal:.4f}"
)

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
print("Internal train shape:", X_train.shape, "Internal test shape:", X_test.shape)

df_ext_copy = df_ext.copy()
X_ext = df_ext_copy[shared_feature_cols].copy()
y_ext = df_ext_copy[target_column].astype(np.float32).copy()

X_ext = X_ext.replace([np.inf, -np.inf], np.nan)
ext_binary_cols = [c for c in binary_cols if c in X_ext.columns]
ext_continuous_cols = [c for c in continuous_cols if c in X_ext.columns]

if ext_binary_cols:
    X_ext[ext_binary_cols] = X_ext[ext_binary_cols].fillna(binary_mode[ext_binary_cols])
if ext_continuous_cols:
    X_ext[ext_continuous_cols] = X_ext[ext_continuous_cols].fillna(continuous_median[ext_continuous_cols])

print("External shape:", X_ext.shape, "External prev:", float(y_ext.mean()))

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

cont_mean, cont_std = fit_standard_scaler(X_train, continuous_cols)

def maybe_scale(df_: pd.DataFrame, do_scale: bool) -> pd.DataFrame:
    if do_scale and continuous_cols:
        return apply_standard_scaler(df_, continuous_cols, cont_mean, cont_std)
    return df_

X_train_s = maybe_scale(X_train, SAINT_PARAMS["scale_continuous"])
X_test_s  = maybe_scale(X_test,  SAINT_PARAMS["scale_continuous"])
X_ext_s   = maybe_scale(X_ext,   SAINT_PARAMS["scale_continuous"])

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
        self.feat_bias  = nn.Parameter(torch.zeros(n_features, d_model))

        self.token_dropout = nn.Dropout(token_dropout)

        self.blocks = nn.ModuleList([
            SAINTBlock(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.n_features:
            raise ValueError(f"Expected (B,{self.n_features}) got {tuple(x.shape)}")

        xt = x.unsqueeze(-1) * self.feat_scale.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        xt = self.token_dropout(xt)

        for blk in self.blocks:
            xt = blk(xt)

        pooled = xt.mean(dim=1)
        logit = self.head(pooled).squeeze(-1)
        return logit

def predict_proba(model, X_np, batch_size=BATCH_SIZE_PRED):
    ds = TensorDataset(torch.from_numpy(X_np.astype(np.float32)))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    model.eval()
    logits_all = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            logits_all.append(model(xb).detach().cpu().numpy())
    logits = np.concatenate(logits_all)
    return 1.0 / (1.0 + np.exp(-logits))

X_tr_np = X_train_s.values.astype(np.float32)
y_tr_np = y_train.values.astype(np.float32)
X_te_np = X_test_s.values.astype(np.float32)
y_te_np = y_test.values.astype(np.float32)

X_tr2_np, X_val2_np, y_tr2_np, y_val2_np = train_test_split(
    X_tr_np, y_tr_np, test_size=0.1, random_state=SEED_GLOBAL, stratify=y_tr_np
)

n_features = X_tr_np.shape[1]
base_model = SAINTNumericClassifier(
    n_features=n_features,
    d_model=int(SAINT_PARAMS["d_model"]),
    n_heads=int(SAINT_PARAMS["n_heads"]),
    n_layers=int(SAINT_PARAMS["n_layers"]),
    attn_dropout=float(SAINT_PARAMS["attn_dropout"]),
    ff_dropout=float(SAINT_PARAMS["ff_dropout"]),
    token_dropout=float(SAINT_PARAMS["token_dropout"]),
    mlp_dropout=float(SAINT_PARAMS["mlp_dropout"]),
).to(device)

opt = torch.optim.AdamW(
    base_model.parameters(),
    lr=float(SAINT_PARAMS["lr"]),
    weight_decay=float(SAINT_PARAMS["weight_decay"]),
)
criterion = nn.BCEWithLogitsLoss()
clip_norm_base = float(SAINT_PARAMS["clip_norm"])

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_tr2_np), torch.from_numpy(y_tr2_np)),
    batch_size=BATCH_SIZE_BASE,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
    num_workers=0,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val2_np), torch.from_numpy(y_val2_np)),
    batch_size=BATCH_SIZE_BASE,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    num_workers=0,
)

best_val = np.inf
best_state = None
no_imp = 0

print("\n Training BASE SAINT ")
for epoch in range(1, BASE_EPOCHS + 1):
    base_model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = criterion(base_model(xb), yb)
        loss.backward()
        if clip_norm_base > 0.0:
            nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=clip_norm_base)
        opt.step()
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    base_model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = criterion(base_model(xb), yb)
            va_loss += loss.item() * xb.size(0)
    va_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

    if va_loss < best_val - 1e-4:
        best_val = va_loss
        best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= BASE_PATIENCE:
            print("Early stop BASE.")
            break

if best_state is not None:
    base_model.load_state_dict(best_state)
base_model.to(device)

y_test_probs_base_internal = predict_proba(base_model, X_te_np)
print("\n Internal TEST performance ")
print(f"Internal TEST AUC:   {roc_auc_score(y_te_np, y_test_probs_base_internal):.4f}")
print(f"Internal TEST AUPRC: {average_precision_score(y_te_np, y_test_probs_base_internal):.4f}")
print(f"Internal TEST Brier: {brier_score_loss(y_te_np, y_test_probs_base_internal):.4f}")
thr_int, sens_int, spec_int, j_int = youden_sens_spec(y_te_np, y_test_probs_base_internal)
print(f"Internal TEST Youden thr={thr_int:.6f} | sens={sens_int:.4f} | spec={spec_int:.4f} | J={j_int:.4f}")

base_state_cpu = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

seen = set()

X_ext_np_full = X_ext_s.values.astype(np.float32)
y_ext_np_full = y_ext.values.astype(np.float32)

for seed in seeds:
    print(f"\n SEED {seed} ")
    assert seed not in seen, f"Duplicate seed {seed}"
    seen.add(seed)

    X_ext_tl, X_ext_test, y_ext_tl, y_ext_test = train_test_split(
        X_ext_np_full, y_ext_np_full, test_size=0.3, random_state=seed, stratify=y_ext_np_full
    )

    X_ext_test_np = X_ext_test.astype(np.float32)
    y_ext_test_np = y_ext_test.astype(np.float32)

    y_ext_test_probs_base = predict_proba(base_model, X_ext_test_np)
    auc_base = roc_auc_score(y_ext_test_np, y_ext_test_probs_base)
    auprc_base = average_precision_score(y_ext_test_np, y_ext_test_probs_base)
    brier_base = brier_score_loss(y_ext_test_np, y_ext_test_probs_base)
    thr_b, sens_b, spec_b, j_b = youden_sens_spec(y_ext_test_np, y_ext_test_probs_base)
    print(
        f"External TEST (BASE) - AUC: {auc_base:.4f}, AUPRC: {auprc_base:.4f}, Brier: {brier_base:.4f} | "
        f"Youden thr={thr_b:.6f}, sens={sens_b:.4f}, spec={spec_b:.4f}"
    )

    X_tl_np = X_ext_tl.astype(np.float32)
    y_tl_np = y_ext_tl.astype(np.float32)

    X_tl_tr_np, X_tl_val_np, y_tl_tr_np, y_tl_val_np = train_test_split(
        X_tl_np, y_tl_np, test_size=0.2, random_state=seed, stratify=y_tl_np
    )

    npos = float((y_tl_tr_np == 1).sum())
    nneg = float((y_tl_tr_np == 0).sum())
    pos_weight_value = (nneg / npos) if npos > 0 else 1.0

    def tl_objective(trial):
        torch.manual_seed(seed + trial.number)
        np.random.seed(seed + trial.number)
        random.seed(seed + trial.number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + trial.number)

        lr_head = trial.suggest_float("lr_head", 5e-5, 5e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-8, 5e-2, log=True)
        clip_norm = trial.suggest_float("clip_norm", 0.0, 2.0)

        model_tl = SAINTNumericClassifier(
            n_features=n_features,
            d_model=int(SAINT_PARAMS["d_model"]),
            n_heads=int(SAINT_PARAMS["n_heads"]),
            n_layers=int(SAINT_PARAMS["n_layers"]),
            attn_dropout=float(SAINT_PARAMS["attn_dropout"]),
            ff_dropout=float(SAINT_PARAMS["ff_dropout"]),
            token_dropout=float(SAINT_PARAMS["token_dropout"]),
            mlp_dropout=float(SAINT_PARAMS["mlp_dropout"]),
        )
        model_tl.load_state_dict(base_state_cpu)
        model_tl.to(device)

        for p in model_tl.parameters():
            p.requires_grad = False
        for p in model_tl.head.parameters():
            p.requires_grad = True

        params = [{"params": list(model_tl.head.parameters()), "lr": lr_head}]

        if not HEAD_ONLY:
            lr_feature = trial.suggest_float("lr_feature", 1e-5, 5e-3, log=True)
            for p in model_tl.blocks[-1].parameters():
                p.requires_grad = True
            params.append({"params": model_tl.blocks[-1].parameters(), "lr": lr_feature})

        opt_tl = torch.optim.AdamW(params, weight_decay=weight_decay)

        pos_w = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
        crit_tl = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        bs = BATCH_SIZE_BASE
        train_loader_tl = DataLoader(
            TensorDataset(torch.from_numpy(X_tl_tr_np), torch.from_numpy(y_tl_tr_np)),
            batch_size=bs,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        val_loader_tl = DataLoader(
            TensorDataset(torch.from_numpy(X_tl_val_np), torch.from_numpy(y_tl_val_np)),
            batch_size=bs,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )

        best_auc_local = -np.inf
        best_epoch_local = 0
        no_imp_local = 0

        for ep in range(TL_MAX_EPOCHS):
            model_tl.train()
            for xb, yb in train_loader_tl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt_tl.zero_grad(set_to_none=True)
                loss = crit_tl(model_tl(xb), yb)
                loss.backward()
                if clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(model_tl.parameters(), max_norm=clip_norm)
                opt_tl.step()

            model_tl.eval()
            logits_chunks = []
            with torch.no_grad():
                for xb, _ in val_loader_tl:
                    xb = xb.to(device, non_blocking=True)
                    logits_chunks.append(model_tl(xb).detach().cpu())
            val_logits = torch.cat(logits_chunks).numpy()
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            auc_val = roc_auc_score(y_tl_val_np, val_probs)

            trial.report(auc_val, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if auc_val > best_auc_local + 1e-4:
                best_auc_local = auc_val
                best_epoch_local = ep + 1
                no_imp_local = 0
            else:
                if ep + 1 >= TL_MIN_EPOCHS:
                    no_imp_local += 1
                    if no_imp_local >= TL_PATIENCE:
                        break

        trial.set_user_attr("best_epoch", int(best_epoch_local))
        return float(best_auc_local)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    sampler = TPESampler(seed=seed, n_startup_trials=20)

    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(tl_objective, n_trials=N_TRIALS_TL, n_jobs=1)

    best_tl = study.best_params
    best_epoch = int(study.best_trial.user_attrs.get("best_epoch", 10))
    print("Best TL params:", best_tl, "| best_epoch:", best_epoch, "| best_trial#:", study.best_trial.number)

    X_tl_fit_np, X_tl_es_np, y_tl_fit_np, y_tl_es_np = train_test_split(
        X_tl_np,
        y_tl_np,
        test_size=TL_ES_FRAC,
        random_state=seed,
        stratify=y_tl_np,
    )

    npos2 = float((y_tl_fit_np == 1).sum())
    nneg2 = float((y_tl_fit_np == 0).sum())
    pos_weight_value2 = (nneg2 / npos2) if npos2 > 0 else 1.0

    model_tl_final = SAINTNumericClassifier(
        n_features=n_features,
        d_model=int(SAINT_PARAMS["d_model"]),
        n_heads=int(SAINT_PARAMS["n_heads"]),
        n_layers=int(SAINT_PARAMS["n_layers"]),
        attn_dropout=float(SAINT_PARAMS["attn_dropout"]),
        ff_dropout=float(SAINT_PARAMS["ff_dropout"]),
        token_dropout=float(SAINT_PARAMS["token_dropout"]),
        mlp_dropout=float(SAINT_PARAMS["mlp_dropout"]),
    )
    model_tl_final.load_state_dict(base_state_cpu)
    model_tl_final.to(device)

    for p in model_tl_final.parameters():
        p.requires_grad = False
    for p in model_tl_final.head.parameters():
        p.requires_grad = True

    params_final = [{"params": list(model_tl_final.head.parameters()), "lr": best_tl["lr_head"]}]
    if not HEAD_ONLY:
        for p in model_tl_final.blocks[-1].parameters():
            p.requires_grad = True
        params_final.append({"params": model_tl_final.blocks[-1].parameters(), "lr": best_tl["lr_feature"]})

    opt_tl_final = torch.optim.AdamW(params_final, weight_decay=best_tl["weight_decay"])

    pos_w2 = torch.tensor([pos_weight_value2], device=device, dtype=torch.float32)
    crit_tl2 = nn.BCEWithLogitsLoss(pos_weight=pos_w2)

    train_loader_fit = DataLoader(
        TensorDataset(torch.from_numpy(X_tl_fit_np), torch.from_numpy(y_tl_fit_np)),
        batch_size=BATCH_SIZE_BASE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    val_loader_es = DataLoader(
        TensorDataset(torch.from_numpy(X_tl_es_np), torch.from_numpy(y_tl_es_np)),
        batch_size=BATCH_SIZE_BASE,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    clip_norm_tl = float(best_tl.get("clip_norm", 0.0))

    best_auc_es = -np.inf
    best_state_es = None
    no_imp_es = 0
    best_epoch_es = 0

    for ep in range(TL_MAX_EPOCHS):
        model_tl_final.train()
        for xb, yb in train_loader_fit:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt_tl_final.zero_grad(set_to_none=True)
            loss = crit_tl2(model_tl_final(xb), yb)
            loss.backward()
            if clip_norm_tl > 0.0:
                nn.utils.clip_grad_norm_(model_tl_final.parameters(), max_norm=clip_norm_tl)
            opt_tl_final.step()

        model_tl_final.eval()
        logits_chunks = []
        with torch.no_grad():
            for xb, _ in val_loader_es:
                xb = xb.to(device, non_blocking=True)
                logits_chunks.append(model_tl_final(xb).detach().cpu())
        es_logits = torch.cat(logits_chunks).numpy()
        es_probs = 1.0 / (1.0 + np.exp(-es_logits))
        auc_es = roc_auc_score(y_tl_es_np, es_probs)

        if auc_es > best_auc_es + 1e-4:
            best_auc_es = auc_es
            best_epoch_es = ep + 1
            best_state_es = {k: v.detach().cpu().clone() for k, v in model_tl_final.state_dict().items()}
            no_imp_es = 0
        else:
            if ep + 1 >= TL_MIN_EPOCHS:
                no_imp_es += 1
                if no_imp_es >= TL_PATIENCE:
                    break

    if best_state_es is not None:
        model_tl_final.load_state_dict(best_state_es)
    model_tl_final.to(device)

    print(f"Final TL early-stop epoch={best_epoch_es} | ES AUC={best_auc_es:.4f} | TL_fit n={len(y_tl_fit_np)} | TL_es n={len(y_tl_es_np)}")

    y_ext_test_probs_tl = predict_proba(model_tl_final, X_ext_test_np)
    auc_tl = roc_auc_score(y_ext_test_np, y_ext_test_probs_tl)
    auprc_tl = average_precision_score(y_ext_test_np, y_ext_test_probs_tl)
    brier_tl = brier_score_loss(y_ext_test_np, y_ext_test_probs_tl)
    thr_t, sens_t, spec_t, j_t = youden_sens_spec(y_ext_test_np, y_ext_test_probs_tl)
    print(
        f"External TEST (TL) - AUC: {auc_tl:.4f}, AUPRC: {auprc_tl:.4f}, Brier: {brier_tl:.4f} | "
        f"Youden thr={thr_t:.6f}, sens={sens_t:.4f}, spec={spec_t:.4f}"
    )

    y_ext_tl_probs_tl = predict_proba(model_tl_final, X_tl_np)
    lr_platt = LogisticRegression(solver="lbfgs", max_iter=10000)
    lr_platt.fit(y_ext_tl_probs_tl.reshape(-1, 1), y_tl_np)

    y_ext_test_probs_platt = lr_platt.predict_proba(y_ext_test_probs_tl.reshape(-1, 1))[:, 1]
    auc_platt = roc_auc_score(y_ext_test_np, y_ext_test_probs_platt)
    auprc_platt = average_precision_score(y_ext_test_np, y_ext_test_probs_platt)
    brier_platt = brier_score_loss(y_ext_test_np, y_ext_test_probs_platt)
    thr_p, sens_p, spec_p, j_p = youden_sens_spec(y_ext_test_np, y_ext_test_probs_platt)
    print(
        f"External TEST (TL+Platt) - AUC: {auc_platt:.4f}, AUPRC: {auprc_platt:.4f}, Brier: {brier_platt:.4f} | "
        f"Youden thr={thr_p:.6f}, sens={sens_p:.4f}, spec={spec_p:.4f}"
    )

    rec = {
        "seed": seed,
        "auc_base": auc_base, "auc_tl": auc_tl, "auc_platt": auc_platt,
        "auprc_base": auprc_base, "auprc_tl": auprc_tl, "auprc_platt": auprc_platt,
        "brier_base": brier_base, "brier_tl": brier_tl, "brier_platt": brier_platt,
        "youden_thr_base": thr_b, "sens_base": sens_b, "spec_base": spec_b, "youdenJ_base": j_b,
        "youden_thr_tl": thr_t,   "sens_tl": sens_t,   "spec_tl": spec_t,   "youdenJ_tl": j_t,
        "youden_thr_platt": thr_p, "sens_platt": sens_p, "spec_platt": spec_p, "youdenJ_platt": j_p,
        "tl_lr_head": best_tl["lr_head"],
        "tl_weight_decay": best_tl["weight_decay"],
        "tl_clip_norm": float(best_tl.get("clip_norm", 0.0)),
        "tl_best_epoch_optuna": best_epoch,
        "tl_best_epoch_es": int(best_epoch_es),
        "tl_es_auc": float(best_auc_es),
        "tl_pos_weight": float(pos_weight_value2),
        "head_only": int(HEAD_ONLY),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_value": float(study.best_value),
        "tl_es_frac": float(TL_ES_FRAC),
    }
    if not HEAD_ONLY:
        rec["tl_lr_feature"] = best_tl["lr_feature"]
    records.append(rec)

    fpr_base, tpr_base, _ = roc_curve(y_ext_test_np, y_ext_test_probs_base)
    fpr_tl, tpr_tl, _ = roc_curve(y_ext_test_np, y_ext_test_probs_tl)
    fpr_pl, tpr_pl, _ = roc_curve(y_ext_test_np, y_ext_test_probs_platt)
    roc_curves["base"].append((fpr_base, tpr_base, seed))
    roc_curves["tl"].append((fpr_tl, tpr_tl, seed))
    roc_curves["platt"].append((fpr_pl, tpr_pl, seed))

    prec_b, rec_b, _ = precision_recall_curve(y_ext_test_np, y_ext_test_probs_base)
    prec_t, rec_t, _ = precision_recall_curve(y_ext_test_np, y_ext_test_probs_tl)
    prec_p, rec_p, _ = precision_recall_curve(y_ext_test_np, y_ext_test_probs_platt)
    pr_curves["base"].append((rec_b, prec_b, seed))
    pr_curves["tl"].append((rec_t, prec_t, seed))
    pr_curves["platt"].append((rec_p, prec_p, seed))

metrics_df = pd.DataFrame(records)

print("\nRows:", len(metrics_df), " Unique seeds:", metrics_df["seed"].nunique())
print("Seeds recorded:", metrics_df["seed"].tolist())

assert metrics_df["seed"].nunique() == len(seeds), "Not all seeds were recorded!"
assert len(metrics_df) == len(seeds), "Expected exactly 1 row per seed!"

metrics_df.to_csv("metrics_per_seed_saint_itl.csv", index=False)
print("Saved metrics_per_seed_saint_itl.csv")

with open("roc_pr_curves_saint_itl.pkl", "wb") as f:
    pickle.dump({"roc_curves": roc_curves, "pr_curves": pr_curves}, f)
print("Saved roc_pr_curves_saint_itl.pkl")
