# MCDM evaluation outputs

Produced by `main_code/evaluate_models.py`. Run from the repository root (GPU required):

```powershell
.venv\Scripts\python.exe main_code\evaluate_models.py --allow-download
```

All models are scored **out-of-fold (OOF)** on the 5240 training lesions with the same split:
`StratifiedKFold(5, shuffle=True, random_state=42)` at lesion level (`common.make_folds`).

## Last verified run

2026‑11‑26 20:21, RTX 5070, from the existing caches (106 s, nothing fine‑tuned). Compared with the previous run of the same day (11:09):

- **Identical:** `criteria.csv`, `per_class_metrics.csv`, `fold_metrics.csv`, `model_complexity.csv`, `component_sizes.csv`, all 36 files in `confusion_matrices/`, all 9 files in `oof_predictions/`, and in `decision_matrix.csv` every column except inference time.
- **Changed:** latency only (`latency_summary.csv`, `latency_stages.csv`, the inference‑time column) and the timestamp in `run_info.json`. Per‑model means are 0.0–3.8 % lower; the speed ranking is the same.
- **Checks in `run_info.json`** have the same values as before. Not all are exact matches: see "Frozen DINOv2‑L" and "Fine‑tuned model and image preprocessing" below.

## Files

| File | Content |
|---|---|
| `decision_matrix.csv` | One row per model, one column per criterion |
| `criteria.csv` | Whether each criterion is a benefit or a cost |
| `per_class_metrics.csv` | Precision, recall, F1, support and predicted count for each model × class |
| `fold_metrics.csv` | The five classification metrics per model × fold (for paired tests) |
| `confusion_matrices/` | `{model_id}_{raw,normalized}.{csv,png}`; rows are true classes. Normalised = by row |
| `oof_predictions/{model_id}.csv` | `lesion_id`, `fold`, true and predicted label, `age_approx`, `sex`, `skin_tone_class`, `site`, `p_<class>` × 11 |
| `latency_summary.csv`, `latency_stages.csv` | Latency per model (mean, median, p95, std, per-repeat mean) and per pipeline stage |
| `model_complexity.csv`, `component_sizes.csv` | Size and parameter counts per model and per component |
| `run_info.json` | Hardware, library versions, settings and cache consistency checks |
| `TECHNICAL_NOTES_FOR_TEAM.md` | Hand‑written Q&A for the analysis team: inputs, preprocessing, training, MONET, metadata, limitations. Not produced by the script; update it by hand after a new run |

## How the criteria are defined

- **Classification metrics** use the pooled OOF predictions, `argmax` decision, all 11 classes
  (`zero_division=0`). Macro ROC-AUC is one-vs-rest, unweighted over classes.
- **Inference time** is the mean wall-clock time for **one lesion at a time** (batch size 1), from the raw JPEGs
  and the metadata row to the 11 probabilities, with a warm OS file cache. It is the sum of the stages a model
  needs, so the image decode/resize shared by DINOv2-large and TwoViewDino is counted once. Frozen backbones run
  in fp32 (as in 01), TwoViewDino in fp16 autocast with the 8 TTA views run one after another (as in 02).
  The cost of averaging probabilities is not counted. Absolute values depend on this machine (see `run_info.json`)
  and shift by a few percent between runs.
- **Model size** (decimal MB) is the disk size of the files one prediction needs: pretrained backbone weights,
  fitted classifier (`joblib`, mean over the 5 fold models) and/or one `fold{k}.pt`.
- **Parameter count**: all network parameters plus LogReg coefficients and intercepts. For LightGBM it is the
  number of leaf values plus split thresholds, which is not directly comparable with neural network weights.
  `model_complexity.csv` also lists how many parameters were fitted on MILK10k.

## Read before analysing

- **Tabular latency and size exclude MONET.** The 14 MONET scores are read from the dataset metadata. A deployment
  that has to compute them from images would add that model's time and size.
- **OOF is not the leaderboard.** In `history.md` #14 the TTA + 5-model test average had the higher CV score
  (0.560) but scored 0.474 on the leaderboard, below the 0.511 of the full-train model without TTA. Latency and
  size here describe the single-model-per-lesion system that the OOF predictions measure.
- **Some settings were chosen on these folds**, so scores are mildly optimistic and not equally so: LogReg `C`
  (0.03 for DINOv2-large, selected on this CV; 0.001 for ResNet, taken from the ResNet101 experiment and also
  applied to ResNet50), and the ensemble composition (fixed on a fold-0 pilot). `history.md` reports ResNet50
  with single-layer features and C=0.003; the ResNet50 row here uses the multi-layer features of the current
  `BACKBONE` switch, so the two are not the same experiment.
- **Fine-tuned model and image preprocessing.** OOF predictions come from the `img256` cache (resized, then re-saved
  as JPEG quality 95). The latency pipeline resizes the raw image in memory instead, and on 200 fold-0 lesions
  that changed the TwoViewDino class for 3.5% of them (no TTA; 3% with TTA). From the cache the agreement is 100%.
  The metrics here describe the cached preprocessing.
- **Frozen DINOv2-L:** metrics use the cached `reference.npz`. The fold models refitted for latency and size differ
  slightly from it (argmax agreement 99.8%, macro-F1 0.4822 vs 0.4821). LightGBM refits are identical.
- **Rare classes:** MAL_OTH has 9 lesions, BEN_OTH, DF, INF and VASC about 50 each. Per-class and subgroup
  figures for them are very noisy.
- **Missing values** in `age_approx` / `site` are left empty in the prediction files.
