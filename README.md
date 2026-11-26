# MILK10k Skin Lesion Classification

Multimodal classifier for the **MILK10k (ISIC 2025)** challenge: given a **clinical close‑up photo**, a **dermoscopic photo** and **patient metadata** for each skin lesion, predict one of 11 diagnoses (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC). The competition metric is **macro‑F1**, so rare classes (MAL_OTH has 9 training lesions) count as much as common ones (BCC has 2522).

| System | Where | CV macro‑F1 (5‑fold) | Leaderboard |
|---|---|---|---|
| **Current best:** fine‑tuned DINOv2‑base (one model trained on all of train, no TTA) + frozen DINOv2‑large/LogReg + LightGBM (equal‑weight average) | `main_code/notebooks/02_finetune.ipynb` (default: `TEST_MODEL = "full"`, `TTA = False`) | **0.544** | **0.511** |
| Same 3 models, but fine‑tuned part = mean of the 5 CV fold models × 8‑view TTA | `02_finetune.ipynb` with `TEST_MODEL = "folds"`, `TTA = True` | 0.560 | 0.474 |
| Frozen DINOv2‑large/LogReg + LightGBM | `main_code/notebooks/01_baseline.ipynb` | 0.487 | 0.421 |

The 0.560 → 0.474 row is why the default went back to the 0.511 setup: CV scores each lesion with a single fold model, so it never measures the 5‑model test average. See `main_code/history.md` #14.

The table reports the runs as logged in `history.md`. The tables in `main_code/outputs/mcdm/` are computed from the #14 fold models, where the three‑model ensemble without TTA scores 0.549 out‑of‑fold.

**Last verification: 2026‑11‑26** (`history.md` #15). A re‑run of `evaluate_models.py` from the existing caches gave the same five classification metrics for all nine models as the previous run, including OOF macro‑F1 0.4867 (frozen + tabular), 0.5492 (ensemble) and 0.5598 (ensemble with TTA). Only latency changed (per‑model means up to 3.8 % lower). The 0.544 run and the leaderboard scores cannot be re‑checked from this checkout.

Every experiment, its result and the decision taken is logged in [`main_code/history.md`](main_code/history.md) (Turkish).

## Tech stack

- **Python 3.13**, **PyTorch** (CUDA), Jupyter notebooks, **matplotlib** (evaluation plots)
- Hugging Face **transformers** (DINOv2), **torchvision** transforms, **scikit‑learn** (LogisticRegression, StratifiedKFold), **LightGBM**

## Project structure

```
.
├── main_code/                     # active project
│   ├── notebooks/
│   │   ├── 01_baseline.ipynb      # frozen DINOv2-large features + LogReg, LightGBM on metadata, late fusion
│   │   ├── 02_finetune.ipynb      # fine-tunes DINOv2-base on both views: 5-fold CV (+ optional 8-view TTA),
│   │   │                          #   full-train fit for the test set, + 2 reference models -> submission
│   │   ├── common.py              # shared by 01, 02 and evaluate_models.py: data loading, labels, tabular
│   │   │                          #   features, folds, reference models, OOF helper
│   │   └── ft_data.py             # Dataset/transforms for 02 (separate file so DataLoader workers work on Windows)
│   ├── evaluate_models.py         # scores 9 model alternatives on the same folds -> outputs/mcdm/
│   ├── outputs/
│   │   ├── baseline/              # CV tables, confusion matrix, submission + probabilities from 01
│   │   ├── finetune/              # CV tables, submission + probabilities from 02  <- file to submit
│   │   │   └── variants/          # alternative submissions for leaderboard A/B (folds_{notta,tta}; full_{notta,tta}
│   │   │                          #   once full.pt exists) + submission_original_lb0511.csv (exact 0.511 file, from git)
│   │   ├── mcdm/                  # decision matrix, per-class/per-fold metrics, latency, size tables,
│   │   │   │                      #   README.md, TECHNICAL_NOTES_FOR_TEAM.md
│   │   │   ├── oof_predictions/   # one CSV per model
│   │   │   └── confusion_matrices/  # raw and row-normalised, CSV + PNG per model
│   │   └── cache/                 # (git-ignored) feature .npz, img256/ resized images, finetune/ (weights, fold
│   │                              #   predictions, reference.npz), mcdm/ (fitted fold models)
│   ├── data/                      # MILK10k_*.csv (tracked) + image folders (git-ignored; you link them, see below)
│   ├── history.md                 # experiment log and decisions
│   └── requirements.txt
├── MILK10k_Training_Input/        # holds MILK10k_Training_Input/<lesion_id>/<isic_id>.jpg (currently committed, see SUGGESTIONS.md)
├── MILK10k_Test_Input/            # test images, same layout (currently committed)
├── README.md · ARCHITECTURE.md · SUGGESTIONS.md
```

## Getting started

No environment variables or secrets are needed. The DINOv2 weights are downloaded from the Hugging Face Hub on first use (internet required once). If `HF_HUB_OFFLINE=1` is set in your environment, unset it (or set it to `0`) for that first run, otherwise `from_pretrained` fails with "couldn't connect to huggingface.co".

### 1. Python environment (Windows / PowerShell)

```powershell
python -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

Install a PyTorch build that matches your GPU (RTX 50‑series needs CUDA ≥ 12.8; the existing venv uses `cu130`):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

```powershell
pip install -r main_code/requirements.txt
```

Check that the interpreter running your Jupyter kernel actually has everything (`torch` with CUDA, `torchvision`, `transformers`, `lightgbm`, `scikit-learn`). On 2026‑11‑26 the root `.venv` had these but no Jupyter or `ipykernel` (the `requirements.txt` step above installs Jupyter); the system Python 3.13 lacked `torchvision` and `lightgbm`:

```powershell
python -c "import torch, torchvision, transformers, lightgbm, sklearn; print(torch.cuda.is_available())"
```

### 2. Data layout for `main_code`

The notebooks look upward from the folder they are opened in for the first directory containing `data/`, and write outputs to `outputs/` next to it. Put `data/` **inside `main_code/`** (the fine‑tuning image cache in `ft_data.py` assumes that). The three metadata / ground‑truth CSVs are tracked in `main_code/data/` (a `.gitignore` exception); only the image folders are git‑ignored. From the repository root, link the existing image downloads instead of copying them:

```powershell
New-Item -ItemType Junction -Path main_code\data\MILK10k_Training_Input -Target (Resolve-Path MILK10k_Training_Input\MILK10k_Training_Input)
```

```powershell
New-Item -ItemType Junction -Path main_code\data\MILK10k_Test_Input -Target (Resolve-Path MILK10k_Test_Input\MILK10k_Test_Input)
```

Resulting layout: `main_code/data/MILK10k_{Training,Test}_Metadata.csv`, `MILK10k_Training_GroundTruth.csv`, `MILK10k_{Training,Test}_Input/<lesion_id>/<isic_id>.jpg`.

### 3. Run

```powershell
jupyter lab main_code/notebooks
```

1. Run all of **`01_baseline.ipynb`** (a few minutes on GPU). It caches DINOv2‑large features to `outputs/cache/` and writes `outputs/baseline/`.
2. Run all of **`02_finetune.ipynb`**. Per fold it takes ≈ 7 min on an RTX 5070 (the saved 5‑fold run took 35 min) or ≈ 30 min on an RTX 4050. The final full‑train fit takes ≈ 38 min on an RTX 4050 and an estimated 9 min on an RTX 5070 (not timed there). The file to upload is `main_code/outputs/finetune/submission.csv`. Alternatives for leaderboard A/B testing are in `outputs/finetune/variants/`.

What 02 keeps in `main_code/outputs/cache/finetune/` (git‑ignored; `full.pt` and `full.npz` appear after the final cell has run):

| File | Content | Effect on re‑run |
|---|---|---|
| `fold{k}.pt` | fine‑tuned `TwoViewDino` weights of fold *k* + the training settings they were trained with | Reloaded with `torch.load(weights_only=True)` instead of retraining. If `EPOCHS`/`BATCH`/LRs/`MAX_STEPS`/`SEED` differ, the fold is retrained. |
| `fold{k}.npz` | per‑view (8 TTA views) probabilities for the validation fold and the test set | Fold is skipped entirely. Delete these (keep the `.pt`) to re‑predict, e.g. after changing TTA. |
| `full.pt` | `TwoViewDino` trained on **all** 5240 training lesions (the 0.511 setup's test model) + its training settings | Reloaded instead of retraining when the settings match. |
| `full.npz` | 8‑view test probabilities of `full.pt` | Reused; switching `TTA` needs no retraining. |
| `reference.npz` | LightGBM + frozen‑LR out‑of‑fold and test probabilities | Reused; recomputed automatically if lesions, folds or model settings change. |

Settings in 02's first cell:
- `TTA = False` (default, as in the 0.511 submission) uses the plain center crop for the main CV row and the submission. Both are always computed and reported.
- `TEST_MODEL = "full"` (default) predicts the test set with the full‑train model; `"folds"` uses the mean of the 5 CV fold models (scored 0.474 with TTA).
- `RUN_FINAL = False` skips writing the submission. `MAX_STEPS = 20` gives a quick smoke run. A `MAX_STEPS` run writes `.pt` files too, but those weights are tagged with `max_steps` and a full run retrains over them.
- Delete `fold*.pt`, `fold*.npz`, `full.pt` and `full.npz` by hand after changing the augmentation (`ft_data.py`) or the model architecture, which the settings tag does not cover.

Shared code: `notebooks/common.py` holds everything both notebooks must agree on (`load_split`, `load_labels`, `tabular_features`, `make_folds`, `tabular_model`, `image_model`/`frozen_model`, `oof_proba`). Change it there, not in a notebook.

### 4. Compare the models (MCDM tables)

After 01 and 02, from the repository root (CUDA GPU required):

```powershell
.venv\Scripts\python.exe main_code\evaluate_models.py --allow-download
```

It scores nine alternatives (tabular only, frozen ResNet50, ResNet101 and DINOv2‑large, late fusion, then the fine‑tuned model and the full ensemble, each with and without TTA) on the shared folds and overwrites the tables in `main_code/outputs/mcdm/`. `--allow-download` lets torchvision fetch missing ResNet weights; without it a ResNet whose weights are missing is skipped. With every cache and weight file present, the run takes about 2 minutes on an RTX 5070 (106 s on 2026‑11‑26, run without the flag and with `HF_HUB_OFFLINE=1`). Latency values depend on the machine and shift by a few percent between runs.

| Read | For |
|---|---|
| [`main_code/outputs/mcdm/README.md`](main_code/outputs/mcdm/README.md) | File list and how each criterion is defined |
| [`main_code/outputs/mcdm/TECHNICAL_NOTES_FOR_TEAM.md`](main_code/outputs/mcdm/TECHNICAL_NOTES_FOR_TEAM.md) | Model inputs, preprocessing, training, MONET scores, metadata check, limitations |

### Tests

There is no automated test suite. `evaluate_models.py` checks cache consistency while it runs (results in `run_info.json`). Validation is the 5‑fold cross‑validation inside each notebook; compare new results against `main_code/history.md` (its acceptance rule assumes fold‑to‑fold noise of ≈ ±0.011–0.015 macro‑F1; in the verified tables the population standard deviation over folds ranges from 0.006 to 0.035 depending on the model). The 479‑lesion leaderboard is much noisier: on random 479‑lesion subsets of the out‑of‑fold predictions, macro‑F1 has σ ≈ 0.045. Changes that move CV by about 0.01 cannot be confirmed on the leaderboard.
