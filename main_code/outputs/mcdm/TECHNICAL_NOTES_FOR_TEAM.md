# Technical notes for the analysis team (statistics / TOPSIS / VIKOR)

Written 2026‑11‑26 from the code, `main_code/history.md`, `main_code/data/MILK10k_*.csv` and the files in this folder.
Statements that cannot be verified from this repository are marked **not established**.
Final verification (2026‑11‑26 20:21): the evaluation was re‑run from the existing caches. All classification metrics, per‑class and per‑fold tables, predictions, sizes and parameter counts are identical to the previous run. The latency figures below are from this final run (`history.md` #15).

Paths are relative to the repository root.

## Read first

1. **The numbers in this folder are not the headline leaderboard run.** The tables here come from the fold models trained on 2026‑11‑23 (`history.md` #14): full ensemble without TTA, OOF macro‑F1 **0.5492**. The leaderboard score **0.511** belongs to an earlier run (`history.md` #13, OOF 0.544) whose test predictions used a fine‑tuned model trained on all 5240 lesions. None of the nine alternatives is that exact system; `ensemble` is the same recipe.
2. **Latency, size and parameter count of every model that uses LightGBM exclude the cost of the MONET scores** (section 3).
3. **No patient identifier exists** in the metadata, so patient‑level separation between folds cannot be verified (section 5).
4. **Only the fine‑tuned model uses the `img256` JPEG cache.** Frozen DINOv2‑L and the ResNets read the raw JPEGs (section 4).
5. **CSV column names:** the 14 MONET features are not 14 columns of the metadata CSV. The CSV has 7 `MONET_*` columns and 2 rows per lesion; the code pivots them into 14 (section 3).

## 1. Model inputs and architecture

`clin` = clinical close‑up image, `derm` = dermoscopic image, demographics = `age_approx`, `sex`, `skin_tone_class`, `site`.

| `model_id` | Model | `clin` | `derm` | Demographics | MONET | Architecture |
|---|---|:-:|:-:|:-:|:-:|---|
| `tabular_lgbm` | Tabular only | – | – | ✓ | ✓ | LightGBM on 18 features |
| `frozen_resnet50_logreg` | Frozen ResNet50 | ✓ | ✓ | – | – | ImageNet V1 ResNet50, pooled `layer2`+`layer3`+`layer4` (3584 per image, 7168 per lesion) → StandardScaler → LogisticRegression (C = 0.001) |
| `frozen_resnet101_logreg` | Frozen ResNet101 | ✓ | ✓ | – | – | Same as above with ResNet101 |
| `frozen_dinov2l_logreg` | Frozen DINOv2‑L | ✓ | ✓ | – | – | `facebook/dinov2-large`, CLS ‖ mean patch token (2048 per image, 4096 per lesion) → StandardScaler → LogisticRegression (C = 0.03) |
| `late_fusion` | Frozen + tabular | ✓ | ✓ | ✓ | ✓ | `0.5·frozen_dinov2l_logreg + 0.5·tabular_lgbm` |
| `ft_dinov2b` | Fine‑tuned, no TTA | ✓ | ✓ | – | – | `TwoViewDino`: one shared `facebook/dinov2-base` encodes both images; CLS ‖ mean patch per image, concatenated (4 × 768) → Dropout 0.2 → Linear(11) |
| `ft_dinov2b_tta` | Fine‑tuned, 8‑view TTA | ✓ | ✓ | – | – | Mean of the softmax outputs of `ft_dinov2b` over 8 views |
| `ensemble` | Full ensemble | ✓ | ✓ | ✓ | ✓ | `(ft_dinov2b + frozen_dinov2l_logreg + tabular_lgbm) / 3` |
| `ensemble_tta` | Full ensemble, TTA | ✓ | ✓ | ✓ | ✓ | `(ft_dinov2b_tta + frozen_dinov2l_logreg + tabular_lgbm) / 3` |

MONET scores are derived from the images, so `tabular_lgbm` receives image information indirectly even though it reads no pixels.

### How fusion works

There are two different levels of fusion:

- **Between the two images (inside an image model): feature concatenation.** The `clin` and `derm` feature vectors are placed side by side and one classifier sees both.
- **Between models: late fusion.** Each model outputs 11 class probabilities for a lesion. The probability vectors are averaged with fixed, equal weights, and the predicted class is the `argmax`. No meta‑learner is trained and no weight is fitted per fold. `evaluate_models.py` rescales each row to sum to 1 before scoring.

TTA views are the 0/90/180/270° rotations, each with and without a horizontal flip, of the 224 px center crop. Both images of a lesion get the same transform in each view. Only the fine‑tuned part uses TTA.

Equal weights were kept because the alternatives did not pass the project's acceptance rule (gain of about ≥ +0.01 and better on most folds): `history.md` #6 and #12, and the nested‑CV table under #14.

### Results (copy of `decision_matrix.csv`, rounded)

| `model_id` | Macro‑F1 | Accuracy | Balanced acc. | Weighted F1 | Macro ROC‑AUC | ms / lesion | Size MB | Parameters |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `tabular_lgbm` | 0.3334 | 0.6120 | 0.3265 | 0.6044 | 0.8423 | 2.8 | 5.8 | 95,699 |
| `frozen_resnet50_logreg` | 0.4274 | 0.6435 | 0.4658 | 0.6571 | 0.8635 | 11.4 | 103.0 | 23,586,891 |
| `frozen_resnet101_logreg` | 0.4269 | 0.6391 | 0.4641 | 0.6517 | 0.8483 | 16.1 | 179.3 | 42,579,019 |
| `frozen_dinov2l_logreg` | 0.4821 | 0.7082 | 0.4732 | 0.7086 | 0.9080 | 33.0 | 1217.8 | 304,413,707 |
| `late_fusion` | 0.4867 | 0.7273 | 0.4615 | 0.7190 | 0.9099 | 35.8 | 1223.6 | 304,509,406 |
| `ft_dinov2b` | 0.5350 | 0.7578 | 0.5379 | 0.7581 | 0.9088 | 15.6 | 346.5 | 86,614,283 |
| `ft_dinov2b_tta` | 0.5543 | 0.7687 | 0.5479 | 0.7678 | 0.9146 | 58.4 | 346.5 | 86,614,283 |
| `ensemble` | 0.5492 | 0.7781 | 0.5244 | 0.7712 | 0.9304 | 44.4 | 1570.1 | 391,123,689 |
| `ensemble_tta` | 0.5598 | 0.7821 | 0.5308 | 0.7746 | 0.9322 | 87.2 | 1570.1 | 391,123,689 |

Criterion directions are in `criteria.csv` (five benefit criteria, three cost criteria). How latency, size and parameters are counted is described in `README.md` in this folder.

**Pooled score versus fold mean.** `decision_matrix.csv` scores the pooled out‑of‑fold (OOF) predictions. `fold_metrics.csv` has one score per fold. For macro‑F1 the two differ (for example `ensemble`: pooled 0.5492, fold mean 0.5449, sample standard deviation 0.0313). State which one the report uses. The `fold_std` columns in `outputs/{baseline,finetune}/cv_results.csv` are population standard deviations (`numpy.std`).

## 2. Data preprocessing and training

### Data layout

- Training: 5240 lesions = 10,480 images (one `clin` and one `derm` per lesion). Test: 479 lesions = 958 images.
- All training images are 600 × 450 px.
- `common.load_split` turns the two metadata rows of a lesion into one row.
- Labels: one‑hot ground truth → one class index (0–10), single‑label.

| Class | AKIEC | BCC | BEN_OTH | BKL | DF | INF | MAL_OTH | MEL | NV | SCCKA | VASC |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Lesions | 303 | 2522 | 44 | 544 | 52 | 50 | 9 | 450 | 746 | 473 | 47 |

### Image preprocessing

All paths end with a 224 × 224 center crop (or random crop in training) and ImageNet normalisation (mean `0.485, 0.456, 0.406`, std `0.229, 0.224, 0.225`).

| Used by | Steps before the crop |
|---|---|
| `TwoViewDino` training and OOF evaluation | Raw JPEG → short side 256 px (bicubic; 600×450 → 341×256) → **saved as JPEG quality 95** in `outputs/cache/img256/` → decoded again |
| Frozen DINOv2‑L (features and latency) | Raw JPEG → short side 256 px (bicubic), in memory |
| Frozen ResNet50/101 (features and latency) | Raw JPEG → short side 256 px (bilinear), in memory (`torchvision` `IMAGENET1K_V1` transforms) |
| `TwoViewDino` latency measurement only | Raw JPEG → short side 256 px (bicubic), in memory (no JPEG re‑save) |

### Augmentation (fine‑tuning only; `ft_data.TRAIN_TF`)

Applied to the `img256` image, drawn **independently** for the clinical and the dermoscopic image, in this order:

1. `RandomResizedCrop(224)`, area scale 0.35–1.0, default aspect ratio 3/4–4/3, bilinear.
2. `RandomHorizontalFlip` (p = 0.5).
3. `RandomVerticalFlip` (p = 0.5).
4. `RandomRot90`: 0, 90, 180 or 270°, uniformly.
5. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)`.
6. `ToTensor` and normalisation.

Evaluation uses only the center crop and normalisation (`EVAL_TF`). The frozen models and LightGBM use no augmentation.

### Missing values

| Where | Handling |
|---|---|
| `age_approx` (20 training lesions) | Left as `NaN`. LightGBM handles it natively in its tree splits. |
| `site` (31 training, 6 test lesions) | Left empty. `site` is a pandas categorical with the 7 training categories, so an empty value reaches LightGBM as missing. The test set has no category that is absent from training. |
| `sex`, `skin_tone_class`, MONET columns | No missing values in either split. |
| Image models (frozen and fine‑tuned) | They do not read metadata, so there is nothing to impute. Every lesion has both images. |

No imputation, scaling or feature engineering is applied to the tabular features. `age_approx` and `skin_tone_class` are numeric, `sex` and `site` categorical. StandardScaler is applied only to the image features in front of LogisticRegression, and is fitted on the training folds only.

### Class imbalance

All three trained model types use inverse‑frequency ("balanced") weights computed on the training folds: `w_c = N / (11 · n_c)`.

| Model | Mechanism |
|---|---|
| LightGBM | `class_weight="balanced"` |
| LogisticRegression | `class_weight="balanced"` |
| `TwoViewDino` | `nn.CrossEntropyLoss(weight=w)` |

There is no over‑ or under‑sampling, and the "effective number of samples" class‑balanced loss is not used. `history.md` #4 records the effect of removing the weights (fusion macro‑F1 0.436 → 0.351 on the ResNet101 system of that date).

### Hyperparameters

| Model | Settings |
|---|---|
| LightGBM | `n_estimators=300`, `learning_rate=0.03`, `num_leaves=15`, `min_child_samples=10`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.8`, `random_state=42` (300 rounds × 11 classes = 3300 trees) |
| LogisticRegression | `C=0.03` (DINOv2‑L) or `0.001` (ResNet), `max_iter=3000`, other scikit‑learn defaults |
| `TwoViewDino` optimiser | AdamW; learning rate 1e‑5 (backbone) and 1e‑3 (head); weight decay 0.05 on all parameters |
| `TwoViewDino` schedule | Stepped every batch: linear warm‑up over the first epoch, multiplied by a cosine decay to 0 over all 10 epochs |
| `TwoViewDino` loop | 10 epochs; batch 16 lesions (32 images); incomplete last batch dropped (262 steps per epoch in CV); fp16 autocast with GradScaler; gradient norm clipped at 1.0; seed 42 (`torch`, `numpy`) |
| Memory | Gradient checkpointing switches on only if less than 5.3 GB of GPU memory is free (off in the 2026‑11‑23 run) |
| Inference | `TwoViewDino` fp16 autocast; frozen backbones fp32 |

### Checkpoint and epoch selection

- Fixed at **10 epochs**. **No early stopping** and no best‑epoch selection: the weights after the last epoch are saved.
- Only the training loss is logged per epoch. The validation fold is predicted once, after training.
- The saved file `fold{k}.pt` stores the training settings; weights trained with different settings are not reused.
- Seeds are fixed, but training is not guaranteed to be bit‑for‑bit repeatable. Two training runs with the same settings gave fine‑tuned OOF macro‑F1 0.527 (#13) and 0.535 (#14).

## 3. MONET scores

### What is in the data

`MILK10k_Training_Metadata.csv` (and the test file) has **7 columns**, one value per image:

| Column | Concept named by the column |
|---|---|
| `MONET_ulceration_crust` | ulceration / crust |
| `MONET_hair` | hair |
| `MONET_vasculature_vessels` | vasculature / vessels |
| `MONET_erythema` | erythema (redness) |
| `MONET_pigmented` | pigmentation |
| `MONET_gel_water_drop_fluid_dermoscopy_liquid` | gel, water drop or dermoscopy fluid |
| `MONET_skin_markings_pen_ink_purple_pen` | skin markings / pen ink |

`common.load_split` pivots them by image type into **14 per‑lesion features**: `clin_MONET_*` and `derm_MONET_*`.

In the training file all values lie between 0 and 1, and the seven values of an image do not sum to 1 (row sums 0.37–4.62), so each behaves as a separate concept score.

### Origin

- The scores **arrive ready‑made in the dataset metadata**. No code in this repository computes or modifies them.
- The prefix refers to MONET, a dermatology image–text model. `history.md` #7 mentions `suinleelab/monet` in a separate experiment that used it as an image feature extractor; that experiment did not produce these columns.
- Model version, prompts, image preprocessing and calibration behind the metadata scores are **not established** from this repository. Cite the MILK10k dataset documentation for them.

### How they are used

They are 14 of the 18 numeric/categorical inputs of LightGBM, passed as raw numeric values (no scaling). No image model reads them.

### Why their cost is not in latency or size

`evaluate_models.py` starts the tabular timing at the metadata row (`tabular_features` + `lightgbm_predict`) and counts only the LightGBM files, because this pipeline never runs MONET: the values are already in the CSV.

Consequence for MCDM: `tabular_lgbm`, `late_fusion`, `ensemble` and `ensemble_tta` all depend on MONET. A deployment that has only images would also need to run MONET on both images, so the cost criteria of these four models are **lower bounds**, most visibly for `tabular_lgbm` (2.8 ms, 5.8 MB). MONET's latency and size were not measured here. A sensitivity analysis on the cost criteria of these four models is advisable.

## 4. Preprocessing consistency and limitations

### Cached `img256` evaluation versus in‑memory latency pipeline

- The classification metrics of the fine‑tuned model come from the **cached JPEG‑95 `img256`** images, the same preprocessing the model was trained on.
- The latency measurement resizes the raw JPEG **in memory** and skips the JPEG re‑save.
- On the 200 fold‑0 lesions used for timing, the in‑memory path gave a different `TwoViewDino` class for **3.5 %** (7 lesions; agreement 0.965) without TTA and **3 %** (6 lesions; 0.970) with TTA (`run_info.json`, `timed_pipeline_vs_oof_argmax_agreement`).
- **Cause confirmed** by a one‑off check on 2026‑11‑26 (same 200 lesions, fold‑0 model, batch size 1): reading the cache reproduced the OOF classes for 100 %; resizing in memory reproduced the 96.5 % / 97 % above; resizing in memory **and re‑encoding as JPEG 95 in memory** gave outputs identical to the cached path (maximum probability difference 0.0). The mean pixel difference between the two inputs was about 1.5 grey levels out of 255 (clinical images).
- The effect on macro‑F1 over all 5240 lesions with in‑memory preprocessing was **not measured**.
- The other timed models agreed with their OOF classes on 100 % (frozen DINOv2‑L, LightGBM, ResNet50) and 99.5 % (ResNet101, 1 lesion; cause not investigated). The ensembles were not compared separately.

### Why the cached OOF pipeline is kept, and what it does and does not guarantee

- It matches training: switching the evaluation to in‑memory resizing would score the model on inputs prepared differently from those it learned on.
- On 64 validation lesions per fold, predictions recomputed from `fold{k}.pt` matched the cached `fold{k}.npz` exactly (`run_info.json`, `checkpoint_vs_fold_npz`: maximum difference 0.0).
- All **nine** OOF files in `oof_predictions/` are tracked in git, and recomputing the five classification metrics from them reproduces `decision_matrix.csv` (checked 2026‑11‑26, differences about 1e‑16). The statistical and MCDM analysis can therefore be reproduced exactly from this folder.
- It does **not** mean all nine models share the `img256` input: only `ft_dinov2b`, `ft_dinov2b_tta`, `ensemble` and `ensemble_tta` contain a model that uses it.
- It does **not** guarantee bit‑exact results from scratch. Feature caches and fold weights are git‑ignored; retraining changes the fine‑tuned results, re‑extracted DINOv2 features differ slightly (`history.md` #12, #14), and latency changes with hardware and from run to run.
- The LogisticRegression fold models refitted for latency and size agree with the cached frozen DINOv2‑L predictions on 99.8 % of lesions (macro‑F1 0.4822 against the reported 0.4821). Metrics use the cached predictions. LightGBM refits are identical.

### Limitations to state in the report

| Limitation | Detail |
|---|---|
| No patient grouping | See section 5. |
| Selection on the evaluation folds | LogReg `C`, the ResNet multi‑layer features and the ensemble composition (fixed on a fold‑0 pilot) were chosen using these data, so scores are mildly optimistic, and not equally so for all models. The ResNet50 row uses `C` and features selected on ResNet101. |
| One split, one seed | A single 5‑fold split (seed 42) and one training run per fold. |
| Rare classes | MAL_OTH has 9 lesions (1–2 per fold); its F1 is 0 for all nine models (`per_class_metrics.csv`). BEN_OTH, DF, INF and VASC have 44–52. Per‑class and subgroup figures are noisy. |
| OOF is not the leaderboard | In #14 the higher‑CV test setup (5‑model average + TTA, CV 0.560) scored 0.474, below 0.511. OOF measures one fold model per lesion. The 479‑lesion leaderboard has a macro‑F1 standard deviation of about 0.045 (`history.md` #14). |
| Folds are not independent samples | Training sets of the five folds overlap; paired tests over five fold scores have little power and their assumptions are only approximately met. |
| Latency is machine‑specific | RTX 5070, batch size 1, warm file cache, fold‑0 models, 200 lesions × 3 repeats. Between two runs on this machine on 2026‑11‑26, per‑model means differed by 0.0–3.8 % (the final run was the faster one) with the same ranking. `std_ms` in `latency_summary.csv` is sensitive to single outliers and changed much more. |

## 5. Metadata and patient‑level grouping check

### Columns of `MILK10k_Training_Metadata.csv` (10,480 rows × 17 columns)

`lesion_id`, `image_type`, `isic_id`, `attribution`, `copyright_license`, `image_manipulation`, `age_approx`, `sex`, `skin_tone_class`, `site`, and the 7 `MONET_*` columns.

`image_type` is `clinical: close-up` or `dermoscopic`. `image_manipulation` is `instrument only` (10,145 images) or `altered` (335) and is **not used** by any model. `attribution` and `copyright_license` are constant.

### Demographic columns (counted per lesion; values are identical on both rows of every lesion)

| Column | Coding | Missing (representation: empty CSV cell → `NaN`) |
|---|---|---|
| `age_approx` | Number, 5‑year steps from 5 to 85 | Train 20 lesions (40 rows); test 0 |
| `sex` | `female`, `male` | None |
| `skin_tone_class` | Integer 0–5 in train, 1–5 in test | None |
| `site` | `foot`, `genital`, `hand`, `head_neck_face`, `lower_extremity`, `trunk`, `upper_extremity` | Train 31 lesions (62 rows); test 6 lesions (12 rows) |

51 training lesions miss `age_approx` or `site`; none misses both. The direction and clinical meaning of the `skin_tone_class` scale are **not established** from this repository; take them from the dataset documentation before interpreting subgroups.

| `age_approx` | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 85 | missing |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Train lesions | 24 | 26 | 27 | 38 | 53 | 107 | 132 | 227 | 276 | 429 | 483 | 595 | 734 | 753 | 607 | 370 | 339 | 20 |

| `sex` | female | male |
|---|--:|--:|
| Train | 2088 | 3152 |
| Test | 181 | 298 |

| `skin_tone_class` | 0 | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|--:|
| Train | 6 | 105 | 506 | 3174 | 1066 | 383 |
| Test | 0 | 19 | 33 | 269 | 113 | 45 |

| `site` | foot | genital | hand | head_neck_face | lower_extremity | trunk | upper_extremity | missing |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Train | 23 | 8 | 72 | 1491 | 1023 | 1925 | 667 | 31 |
| Test | 1 | 2 | 22 | 146 | 106 | 133 | 63 | 6 |

In `oof_predictions/*.csv` the four columns are copied per lesion; missing values are empty.

### Patient identifier

- **There is no `patient_id` or any equivalent column.** The only identifiers are `lesion_id` (5240 unique) and `isic_id` (10,480 unique, one per image).
- Therefore **whether one patient contributes several lesions cannot be determined** from this metadata. Matching on age, sex, skin tone and site would not identify patients and was not attempted.

### Cross‑validation level

Confirmed: `common.make_folds` runs `StratifiedKFold(5, shuffle=True, random_state=42)` on one row per `lesion_id`, stratified by class label only. Both images of a lesion are always in the same fold. All nine models use this split.

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|--:|--:|--:|--:|--:|
| Validation lesions | 1048 | 1048 | 1048 | 1048 | 1048 |
| MAL_OTH among them | 2 | 2 | 2 | 1 | 2 |

Suggested wording: *"Cross‑validation was stratified by diagnosis and split at lesion level. The public metadata contain no patient identifier, so lesions from the same patient, if any, may appear in different folds; the reported performance may therefore be optimistic with respect to unseen patients."*

## 6. Code and experiment log to share

| File | Content |
|---|---|
| `main_code/notebooks/common.py` | Data loading, labels, tabular features, folds, LightGBM and LogReg definitions, OOF helper |
| `main_code/notebooks/ft_data.py` | `img256` preprocessing, augmentation, lesion dataset |
| `main_code/notebooks/01_baseline.ipynb` | Frozen DINOv2‑L features, LogReg, LightGBM, late fusion |
| `main_code/notebooks/02_finetune.ipynb` | `TwoViewDino` training, TTA, ensembles, submissions |
| `main_code/evaluate_models.py` | Produces the tables in this folder (ResNet alternatives, latency, size) |
| `main_code/history.md` | Experiment log and decisions (Turkish) |

Useful alongside them: `main_code/requirements.txt`, this folder (`main_code/outputs/mcdm/`), and `run_info.json` for hardware and library versions.

Before sending:

- Code comments, notebook text and `history.md` are in **Turkish**.
- The code does not run on its own: it needs the MILK10k images (CC‑BY‑NC, downloaded separately) and, for `evaluate_models.py`, the git‑ignored caches produced by notebooks 01 and 02.
- The saved outputs of both notebooks contain local file paths. The Windows user name in them was replaced with `user` on 2026‑11‑26; drive paths such as `D:\…` remain. Earlier git commits are unchanged, so send files, not the repository history.
- The OOF files carry `lesion_id` and demographic columns from MILK10k (CC‑BY‑NC); pass the dataset attribution and licence terms on with them.
