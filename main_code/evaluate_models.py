"""Standardised evaluation of every model alternative on the shared lesion-level 5-fold split (common.make_folds).

Writes the MCDM decision matrix, per-class metrics, confusion matrices, out-of-fold (OOF) predictions and
latency / size tables to outputs/mcdm/.

Cached artefacts are reused and nothing is fine-tuned here:
  outputs/cache/finetune/reference.npz   OOF of LightGBM and frozen DINOv2-large + LogReg   (02_finetune)
  outputs/cache/finetune/fold{k}.npz/.pt OOF (8 TTA views) and weights of TwoViewDino       (02_finetune)
  outputs/cache/dinov2_large_train.npz   frozen DINOv2-large features                       (01_baseline)
What did not exist yet is built once and cached: ResNet features (same file format as 01_baseline) and the
fitted per-fold LightGBM / LogReg objects (outputs/cache/mcdm/), needed for latency and size.

Usage (from the repository root, GPU required):
    .venv\\Scripts\\python.exe main_code\\evaluate_models.py [--allow-download]
--allow-download lets torchvision fetch missing ResNet ImageNet weights; without it those models are skipped.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import transformers
from huggingface_hub import try_to_load_from_cache
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "notebooks"))
from common import (CLASSES, N_FOLDS, SEED, image_model, lesion_ids, load_labels, load_split, make_folds,  # noqa: E402
                    tabular_features, tabular_model)
from ft_data import LesionDataset, preprocess  # noqa: E402

DATA = ROOT / "data"
CACHE = ROOT / "outputs" / "cache"
FT_DIR = CACHE / "finetune"
MCDM_CACHE = CACHE / "mcdm"
OUT = ROOT / "outputs" / "mcdm"

LABELS = list(range(len(CLASSES)))
SUBGROUPS = ["age_approx", "sex", "skin_tone_class", "site"]
MB = 1e6  # all sizes are decimal megabytes

# ---- copies of the definitions in 01_baseline.ipynb / 02_finetune.ipynb ------------------------------------------
# (they live inside notebook cells, so they cannot be imported; the cache checks below catch any drift)
DINO_TRANSFORM = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
RESNETS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}
LAYERS = ["layer2", "layer3", "layer4"]
# history.md #5: C=0.001 was best for the multi-layer ResNet101 features; common.image_model's C=0.03 is DINOv2-only.
RESNET_C = 0.001
VIEWS = [(rot, flip) for flip in (False, True) for rot in range(4)]


class TwoViewDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        d = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * d, len(CLASSES)))

    def embed(self, x):
        out = self.backbone(pixel_values=x)
        return torch.cat([out.pooler_output, out.last_hidden_state[:, 1:].mean(1)], 1)

    def forward(self, clin, derm):
        e = self.embed(torch.cat([clin, derm]))
        return self.head(torch.cat([e[:len(clin)], e[len(clin):]], 1))


def view(x, rot, flip):
    x = torch.flip(x, dims=[3]) if flip else x
    return torch.rot90(x, rot, dims=[2, 3])


@torch.no_grad()
def ft_views(model, clin, derm, views):
    """Per-view probabilities, shape (n_views, batch, 11); exactly the loop of 02_finetune's predict()."""
    with torch.autocast("cuda", dtype=torch.float16):
        return torch.stack([torch.softmax(model(view(clin, r, f), view(derm, r, f)).float(), 1)
                            for r, f in views]).cpu().numpy()


def load_ft(k):
    saved = torch.load(FT_DIR / f"fold{k}.pt", map_location="cuda", weights_only=True)
    model = TwoViewDino().cuda().eval()
    model.load_state_dict(saved["state_dict"])
    return model, saved["config"]


def load_dinov2_large():
    model = AutoModel.from_pretrained("facebook/dinov2-large").cuda().eval()
    embed = lambda out: torch.cat([out.pooler_output, out.last_hidden_state[:, 1:].mean(1)], 1)
    return model, DINO_TRANSFORM, embed


def load_resnet(name):
    arch, weights = RESNETS[name]
    model = create_feature_extractor(arch(weights=weights), return_nodes={l: l for l in LAYERS}).cuda().eval()
    embed = lambda out: torch.cat([out[l].mean(dim=(2, 3)) for l in LAYERS], 1)
    return model, weights.transforms(), embed


def resnet_weights_file(name):
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(RESNETS[name][1].url).name


class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


# ---- features and OOF predictions ----------------------------------------------------------------------------------
@torch.no_grad()
def resnet_features(name, train):
    """[clinical | dermoscopic] multi-layer GAP features, cached like 01_baseline does (X, lesion_id)."""
    cache = CACHE / f"{name}_layers_train.npz"
    ids = lesion_ids(train)
    if cache.exists():
        with np.load(cache) as z:
            assert (z["lesion_id"] == ids).all(), f"stale cache, delete {cache}"
            return z["X"]
    print(f"extracting {name} features ({2 * len(train)} images)...", flush=True)
    model, transform, embed = load_resnet(name)
    loader = DataLoader(ImageDataset(train["clin"].tolist() + train["derm"].tolist(), transform), batch_size=32,
                        num_workers=4)
    F = np.concatenate([embed(model(x.cuda())).cpu().numpy() for x in loader])
    X = np.hstack([F[:len(train)], F[len(train):]])
    np.savez(cache, X=X, lesion_id=ids)
    del model
    torch.cuda.empty_cache()
    return X


def digest(*parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else np.ascontiguousarray(p).tobytes() if isinstance(p, np.ndarray)
                 else str(p).encode())
    return h.hexdigest()


def fold_models(name, make_model, X, y, folds, fold_of, ids):
    """The 5 fitted fold models of a classical classifier and their OOF probabilities. Fitted once, then cached."""
    data = pd.util.hash_pandas_object(X).values if isinstance(X, pd.DataFrame) else X
    key = digest(data, ids, y, fold_of, repr(make_model()), sklearn.__version__, lightgbm.__version__)
    key_file = MCDM_CACHE / f"{name}_key.txt"
    files = [MCDM_CACHE / f"{name}_fold{k}.joblib" for k in range(N_FOLDS)]
    take = (lambda i: X.iloc[i]) if isinstance(X, pd.DataFrame) else (lambda i: X[i])
    if not (key_file.exists() and key_file.read_text() == key and all(f.exists() for f in files)):
        print(f"fitting {name} fold models...", flush=True)
        MCDM_CACHE.mkdir(parents=True, exist_ok=True)
        key_file.unlink(missing_ok=True)
        for (tr, _), f in zip(folds, files):
            joblib.dump(make_model().fit(take(tr), y[tr]), f)
        key_file.write_text(key)
    fitted = [joblib.load(f) for f in files]
    proba = np.zeros((len(y), len(CLASSES)))
    for m, (_, va) in zip(fitted, folds):
        proba[va] = m.predict_proba(take(va))
    return fitted, files, proba


def verify_ft_checkpoints(train, y, folds, n_lesions):
    """Re-predicts the first n validation lesions of every fold from fold{k}.pt and compares with fold{k}.npz."""
    preprocess(list(train["clin"]) + list(train["derm"]))  # no-op when the img256 cache is complete
    checks = []
    for k, (_, va) in enumerate(folds):
        model, config = load_ft(k)
        idx = va[:n_lesions]
        loader = DataLoader(LesionDataset(train["clin"].values[idx], train["derm"].values[idx], y[idx], False),
                            batch_size=32)
        p = np.concatenate([ft_views(model, c.cuda(), d.cuda(), VIEWS) for c, d, _ in loader], axis=1)
        with np.load(FT_DIR / f"fold{k}.npz") as z:
            ref = z["p_va_views"][:, :len(idx)]
        checks.append({"fold": k, "n": len(idx), "train_config": config,
                       "max_abs_diff": float(np.abs(p - ref).max()),
                       "argmax_agreement_notta": float((p[0].argmax(1) == ref[0].argmax(1)).mean()),
                       "argmax_agreement_tta": float((p.mean(0).argmax(1) == ref.mean(0).argmax(1)).mean())})
        del model
        torch.cuda.empty_cache()
    return checks


# ---- latency -------------------------------------------------------------------------------------------------------
def measure_latency(train, sites, folds, classic, n_lesions, repeats, warmup=10):
    """Times every pipeline stage for one lesion at a time (batch size 1), starting from the raw 600x450 JPEGs and
    the lesion's metadata row. Uses the fold-0 models on fold-0 validation lesions. Returns (long table of stage
    times, probabilities produced by the timed pipeline, lesion indices)."""
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(folds[0][1], size=min(n_lesions, len(folds[0][1])), replace=False))

    backbones = {"dinov2l": load_dinov2_large()}
    for name in ("resnet50", "resnet101"):
        if name in classic:
            backbones[name] = load_resnet(name)
    ft, _ = load_ft(0)
    heads = {name: models_[0] for name, models_ in classic.items()}
    produced = {}

    def prep(i, transform):
        return [transform(Image.open(train[v].iat[i]).convert("RGB")).unsqueeze(0).cuda() for v in ("clin", "derm")]

    @torch.no_grad()
    def features(name, clin, derm):
        model, _, embed = backbones[name]
        f = embed(model(torch.cat([clin, derm]))).cpu().numpy()
        return f.reshape(1, -1)  # [clinical | dermoscopic]

    def run(i, record):
        def timed(stage, fn):
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            record[stage] = (time.perf_counter() - t) * 1000
            return result

        clin, derm = timed("image_load_preprocess_dinov2", lambda: prep(i, DINO_TRANSFORM))
        f = timed("dinov2_large_forward", lambda: features("dinov2l", clin, derm))
        out = {"frozen_dinov2l_logreg": timed("dinov2_large_logreg", lambda: heads["dinov2l"].predict_proba(f))[0]}
        out["ft"] = timed("finetuned_forward_1view", lambda: ft_views(ft, clin, derm, VIEWS[:1]))[0, 0]
        out["ft_tta"] = timed("finetuned_forward_8view", lambda: ft_views(ft, clin, derm, VIEWS).mean(0))[0]
        row = timed("tabular_features", lambda: tabular_features(train.iloc[[i]], sites))
        out["tabular_lgbm"] = timed("lightgbm_predict", lambda: heads["lgbm"].predict_proba(row))[0]
        resnets = [n for n in ("resnet50", "resnet101") if n in backbones]
        if resnets:
            c, d = timed("image_load_preprocess_resnet", lambda: prep(i, backbones[resnets[0]][1]))
            for n in resnets:
                fr = timed(f"{n}_forward", lambda: features(n, c, d))
                out[n] = timed(f"{n}_logreg", lambda: heads[n].predict_proba(fr))[0]
        return out

    for i in idx:  # fill the OS file cache so that disk reads do not favour later repeats
        for v in ("clin", "derm"):
            Path(train[v].iat[i]).read_bytes()
    for i in idx[:warmup]:
        run(i, {})
    rows = []
    for rep in range(repeats):
        print(f"latency repeat {rep + 1}/{repeats} over {len(idx)} lesions...", flush=True)
        for i in rng.permutation(idx):
            record = {}
            out = run(i, record)
            if rep == 0:
                produced[int(i)] = out
            rows += [{"repeat": rep, "lesion_index": int(i), "stage": s, "ms": ms} for s, ms in record.items()]
    return pd.DataFrame(rows), produced, idx


# ---- metrics and plots ---------------------------------------------------------------------------------------------
def overall_metrics(y, p):
    pred = p.argmax(1)
    return {"Macro_F1": f1_score(y, pred, average="macro", labels=LABELS, zero_division=0),
            "Accuracy": accuracy_score(y, pred),
            "Balanced_Accuracy": balanced_accuracy_score(y, pred),
            "Weighted_F1": f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0),
            "Macro_ROC_AUC": roc_auc_score(y, p, multi_class="ovr", average="macro", labels=LABELS)}


BLUES = LinearSegmentedColormap.from_list("blue_ramp", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                        "#256abf", "#184f95", "#0d366b"])


def plot_confusion(cm, normalized, title, path):
    ink, ink2, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor=surface)
    share = cm / cm.sum(1, keepdims=True)  # colour is always the row share, so BCC's 2522 lesions do not drown the rest
    im = ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(LABELS, CLASSES, rotation=45, ha="right", color=ink2)
    ax.set_yticks(LABELS, [f"{c} (n={n})" for c, n in zip(CLASSES, cm.sum(1))], color=ink2)
    ax.set_xticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASSES)), minor=True)
    ax.grid(which="minor", color=surface, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in LABELS:
        for j in LABELS:
            if cm[i, j]:
                text = f"{share[i, j]:.2f}" if normalized else f"{cm[i, j]}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="#ffffff" if share[i, j] > 0.5 else ink)
    ax.set_xlabel("Predicted class", color=ink2)
    ax.set_ylabel("True class", color=ink2)
    ax.set_title(f"{title}\n{'row-normalised (recall on the diagonal)' if normalized else 'lesion counts'}, "
                 "out-of-fold", color=ink, fontsize=11)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("share of the true class", color=ink2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=ink2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


# ---- main ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-download", action="store_true", help="let torchvision download missing ResNet weights")
    ap.add_argument("--timing-lesions", type=int, default=200)
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--verify-lesions", type=int, default=64, help="lesions per fold re-predicted from fold{k}.pt")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "a CUDA GPU is required (fp16 inference of TwoViewDino, latency measurement)"

    train = load_split(DATA, "Training")
    ids = lesion_ids(train)
    y = load_labels(DATA, train.index)
    sites = sorted(train["site"].dropna().unique())
    X_tab = tabular_features(train, sites)
    folds = make_folds(y)
    fold_of = np.empty(len(y), int)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k

    # ---- cached OOF predictions --------------------------------------------------------------------------------
    missing = [f for f in [FT_DIR / "reference.npz", CACHE / "dinov2_large_train.npz"]
               + [FT_DIR / f"fold{k}.{e}" for k in range(N_FOLDS) for e in ("npz", "pt")] if not f.exists()]
    assert not missing, f"run 01_baseline and 02_finetune first; missing: {[str(f) for f in missing]}"
    with np.load(FT_DIR / "reference.npz") as z:
        assert (np.array_equal(z["train_id"], ids) and np.array_equal(z["y"], y)
                and np.array_equal(z["fold_of"], fold_of)), "reference.npz belongs to other lesions / labels / folds"
        assert str(z["config"]) == repr(tabular_model()) + repr(image_model()), \
            "reference.npz was made with other model settings; re-run 02_finetune"
        p_tab, p_frozen = z["p_tab"], z["p_frozen"]
    p_ft, p_ft_tta = np.zeros_like(p_tab), np.zeros_like(p_tab)
    for k, (_, va) in enumerate(folds):
        with np.load(FT_DIR / f"fold{k}.npz") as z:
            assert np.array_equal(z["va"], va), f"fold{k}.npz belongs to another split"
            p_ft[va] = z["p_va_views"][0]
            p_ft_tta[va] = z["p_va_views"].mean(0)
    with np.load(CACHE / "dinov2_large_train.npz") as z:
        assert (z["lesion_id"] == ids).all(), "stale DINOv2-large feature cache"
        X_dino = z["X"]

    # ---- classical fold models (for latency / size) and the ResNet alternatives --------------------------------
    classic, classic_files, refit = {}, {}, {}
    classic["lgbm"], classic_files["lgbm"], refit["lgbm"] = fold_models(
        "lgbm", tabular_model, X_tab, y, folds, fold_of, ids)
    classic["dinov2l"], classic_files["dinov2l"], refit["dinov2l"] = fold_models(
        "dinov2l_logreg", image_model, X_dino, y, folds, fold_of, ids)
    resnet_model = lambda: image_model().set_params(logisticregression__C=RESNET_C)
    skipped = []
    for name in RESNETS:
        if not (resnet_weights_file(name).exists() or args.allow_download):
            print(f"SKIPPED {name}: weights {resnet_weights_file(name).name} not found (use --allow-download)")
            skipped.append(name)
            continue
        X = resnet_features(name, train)
        classic[name], classic_files[name], refit[name] = fold_models(
            f"{name}_logreg", resnet_model, X, y, folds, fold_of, ids)

    # ---- model alternatives ------------------------------------------------------------------------------------
    img = ["image_load_preprocess_dinov2"]
    frozen_stages = ["dinov2_large_forward", "dinov2_large_logreg"]
    tab_stages = ["tabular_features", "lightgbm_predict"]
    alts = {  # id: (display name, OOF probabilities, latency stages, size components)
        "tabular_lgbm": ("Tabular only (LightGBM)", p_tab, tab_stages, ["lgbm"]),
        "frozen_dinov2l_logreg": ("Frozen DINOv2-L + LogReg", p_frozen, img + frozen_stages,
                                  ["dinov2l_backbone", "dinov2l_head"]),
    }
    for name, label in (("resnet50", "ResNet50"), ("resnet101", "ResNet101")):
        if name in classic:
            alts[f"frozen_{name}_logreg"] = (f"Frozen {label} + LogReg", refit[name],
                                             ["image_load_preprocess_resnet", f"{name}_forward", f"{name}_logreg"],
                                             [f"{name}_backbone", f"{name}_head"])
    ensemble_parts = ["finetuned", "dinov2l_backbone", "dinov2l_head", "lgbm"]
    alts |= {
        "late_fusion": ("Late fusion (0.5 frozen DINOv2-L + 0.5 LightGBM)", 0.5 * p_frozen + 0.5 * p_tab,
                        img + frozen_stages + tab_stages, ["dinov2l_backbone", "dinov2l_head", "lgbm"]),
        "ft_dinov2b": ("Fine-tuned TwoViewDino (no TTA)", p_ft, img + ["finetuned_forward_1view"], ["finetuned"]),
        "ft_dinov2b_tta": ("Fine-tuned TwoViewDino (8-view TTA)", p_ft_tta, img + ["finetuned_forward_8view"],
                           ["finetuned"]),
        "ensemble": ("Full ensemble (no TTA)", (p_ft + p_frozen + p_tab) / 3,
                     img + ["finetuned_forward_1view"] + frozen_stages + tab_stages, ensemble_parts),
        "ensemble_tta": ("Full ensemble (8-view TTA)", (p_ft_tta + p_frozen + p_tab) / 3,
                         img + ["finetuned_forward_8view"] + frozen_stages + tab_stages, ensemble_parts),
    }

    # ---- consistency checks ------------------------------------------------------------------------------------
    checks = {"refit_vs_reference_npz": {
        name: {"max_abs_diff": float(np.abs(refit[name] - cached).max()),
               "argmax_agreement": float((refit[name].argmax(1) == cached.argmax(1)).mean())}
        for name, cached in (("lgbm", p_tab), ("dinov2l", p_frozen))}}
    recorded = ROOT / "outputs" / "finetune" / "cv_results.csv"
    if recorded.exists():
        table = pd.read_csv(recorded, index_col=0)["macro_f1"]
        rows = {"01: donmuş + tabular": "late_fusion", "fine-tuned": "ft_dinov2b",
                "fine-tuned (TTA)": "ft_dinov2b_tta", "fine-tuned + donmuş + tabular (TTA'sız)": "ensemble"}
        checks["macro_f1_vs_02_cv_results"] = {
            m: {"recorded": float(table[r]), "recomputed": overall_metrics(y, alts[m][1])["Macro_F1"]}
            for r, m in rows.items() if r in table.index}
        for m, c in checks["macro_f1_vs_02_cv_results"].items():
            assert abs(c["recorded"] - c["recomputed"]) < 1e-6, f"{m} does not reproduce 02_finetune: {c}"
    print("verifying fine-tuned checkpoints against cached predictions...", flush=True)
    checks["checkpoint_vs_fold_npz"] = verify_ft_checkpoints(train, y, folds, args.verify_lesions)

    # ---- latency -----------------------------------------------------------------------------------------------
    stage_times, produced, timed_idx = measure_latency(train, sites, folds, classic, args.timing_lesions,
                                                       args.timing_repeats)
    # the timed raw-image pipeline must make (almost) the same decisions as the evaluated OOF predictions
    cached_of = {"frozen_dinov2l_logreg": p_frozen, "ft": p_ft, "ft_tta": p_ft_tta, "tabular_lgbm": p_tab,
                 "resnet50": refit.get("resnet50"), "resnet101": refit.get("resnet101")}
    checks["timed_pipeline_vs_oof_argmax_agreement"] = {
        name: float(np.mean([produced[int(i)][name].argmax() == p[i].argmax() for i in timed_idx]))
        for name, p in cached_of.items() if p is not None and name in produced[int(timed_idx[0])]}

    # ---- size and parameter counts -----------------------------------------------------------------------------
    def n_params(module):
        return sum(p.numel() for p in module.parameters())

    parts = {}  # component: (bytes, parameters, parameters fitted on MILK10k, note)
    dumps = [m.booster_.dump_model()["tree_info"] for m in classic["lgbm"]]
    # a tree with L leaves has L leaf values and L - 1 split thresholds
    lgbm_params = int(round(np.mean([sum(2 * t["num_leaves"] - 1 for t in trees) for trees in dumps])))
    parts["lgbm"] = (np.mean([f.stat().st_size for f in classic_files["lgbm"]]), lgbm_params, lgbm_params,
                     f"{len(dumps[0])} trees; leaf values + split thresholds, mean of 5 fold models (joblib)")
    for name in [n for n in classic if n != "lgbm"]:
        lr = classic[name][0][-1]
        n = lr.coef_.size + lr.intercept_.size
        parts[f"{name}_head"] = (np.mean([f.stat().st_size for f in classic_files[name]]), n, n,
                                 "StandardScaler + LogisticRegression (joblib); scaler statistics not counted")
    dino_file = Path(try_to_load_from_cache("facebook/dinov2-large", "model.safetensors"))
    parts["dinov2l_backbone"] = (dino_file.stat().st_size, n_params(load_dinov2_large()[0]), 0,
                                 "frozen pretrained, model.safetensors")
    for name in RESNETS:
        if name in classic:
            parts[f"{name}_backbone"] = (resnet_weights_file(name).stat().st_size, n_params(load_resnet(name)[0]), 0,
                                         "frozen pretrained, torchvision .pth (file includes the unused fc layer)")
    ft_model = load_ft(0)[0]
    n_ft = n_params(ft_model)
    del ft_model
    parts["finetuned"] = (np.mean([(FT_DIR / f"fold{k}.pt").stat().st_size for k in range(N_FOLDS)]), n_ft, n_ft,
                          "one fold{k}.pt (a single TwoViewDino, as scored by the OOF predictions)")

    # ---- write outputs -----------------------------------------------------------------------------------------
    for sub in ("confusion_matrices", "oof_predictions"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    wide = stage_times.pivot(index=["repeat", "lesion_index"], columns="stage", values="ms")
    matrix, per_class, per_fold, latency, complexity = [], [], [], [], []
    for model_id, (label, p, stages, components) in alts.items():
        p = p.astype(np.float64)
        p = p / p.sum(1, keepdims=True)
        pred = p.argmax(1)
        total = wide[stages].sum(axis=1)
        size = sum(parts[c][0] for c in components) / MB
        params = int(sum(parts[c][1] for c in components))
        head = {"model_id": model_id, "Model": label}
        matrix.append(head | overall_metrics(y, p) | {"Inference_Time_ms_per_lesion": total.mean(),
                                                      "Model_Size_MB": size, "Parameter_Count": params})
        latency.append(head | {"mean_ms": total.mean(), "median_ms": total.median(), "p95_ms": total.quantile(0.95),
                               "std_ms": total.std(),
                               **{f"mean_ms_repeat{r}": v for r, v in total.groupby("repeat").mean().items()},
                               "n_lesions": len(timed_idx), "stages": " + ".join(stages)})
        complexity.append(head | {"Model_Size_MB": size, "Parameter_Count": params,
                                  "Fitted_on_MILK10k_Parameter_Count": int(sum(parts[c][2] for c in components)),
                                  "components": " + ".join(components)})
        for k, (_, va) in enumerate(folds):
            per_fold.append(head | {"fold": k, "n": len(va)} | overall_metrics(y[va], p[va]))

        report = classification_report(y, pred, labels=LABELS, target_names=CLASSES, output_dict=True, zero_division=0)
        for c, name in enumerate(CLASSES):
            r = report[name]
            per_class.append(head | {"class": name, "support": int(r["support"]), "predicted_count": int((pred == c).sum()),
                                     "precision": r["precision"], "recall": r["recall"], "f1": r["f1-score"]})

        cm = confusion_matrix(y, pred, labels=LABELS)
        for kind, values, fmt in (("raw", cm, "%d"), ("normalized", cm / cm.sum(1, keepdims=True), "%.6f")):
            stem = OUT / "confusion_matrices" / f"{model_id}_{kind}"
            pd.DataFrame(values, index=pd.Index(CLASSES, name="true\\predicted"), columns=CLASSES) \
                .to_csv(f"{stem}.csv", float_format=fmt if kind == "normalized" else None)
            plot_confusion(cm, kind == "normalized", label, f"{stem}.png")

        oof = pd.DataFrame({"lesion_id": ids, "fold": fold_of, "y_true": y, "true_label": np.array(CLASSES)[y],
                            "y_pred": pred, "pred_label": np.array(CLASSES)[pred]})
        for col in SUBGROUPS:
            oof[col] = train[col].values
        oof[[f"p_{c}" for c in CLASSES]] = p
        oof.to_csv(OUT / "oof_predictions" / f"{model_id}.csv", index=False, float_format="%.10g")

    pd.DataFrame(matrix).to_csv(OUT / "decision_matrix.csv", index=False)
    pd.DataFrame(per_class).to_csv(OUT / "per_class_metrics.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(latency).to_csv(OUT / "latency_summary.csv", index=False)
    stage_times.groupby("stage")["ms"].describe(percentiles=[0.5, 0.95]).to_csv(OUT / "latency_stages.csv")
    pd.DataFrame(complexity).to_csv(OUT / "model_complexity.csv", index=False)
    pd.DataFrame([{"component": c, "size_MB": b / MB, "parameters": int(n), "fitted_parameters": int(f), "note": note}
                  for c, (b, n, f, note) in parts.items()]).to_csv(OUT / "component_sizes.csv", index=False)
    pd.DataFrame({"criterion": ["Macro_F1", "Accuracy", "Balanced_Accuracy", "Weighted_F1", "Macro_ROC_AUC",
                                "Inference_Time_ms_per_lesion", "Model_Size_MB", "Parameter_Count"],
                  "direction": ["benefit"] * 5 + ["cost"] * 3}).to_csv(OUT / "criteria.csv", index=False)

    info = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": torch.cuda.get_device_name(0),
            "platform": platform.platform(), "processor": platform.processor(),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "transformers": transformers.__version__,
                         "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__},
            "split": "StratifiedKFold(5, shuffle=True, random_state=42), lesion level (common.make_folds)",
            "n_lesions": len(y), "skipped_models": skipped, "resnet_logreg_C": RESNET_C,
            "timing": {"lesions": len(timed_idx), "repeats": args.timing_repeats, "batch_size": 1, "fold_model": 0},
            "checks": checks}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + pd.DataFrame(matrix).drop(columns="Model").round(4).to_string(index=False))
    print("\nchecks:\n" + json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
