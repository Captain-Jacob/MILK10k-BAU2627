# Engineering Suggestions

Findings from a full audit of the repository (2026‑11‑23). Items are ordered by impact within each section.

## 1. Status (2026‑11‑23)

- **Legacy pipeline retired.** `OLD_Test_Folder/` (EfficientNet‑B0 scripts, Docker) was deleted from the working tree; it remains in git history (commit `91f9cf46`). Its metadata/ground‑truth CSVs were byte‑identical to `main_code/data/MILK10k_*.csv`, which are now tracked through a `.gitignore` exception (`data/*` + `!data/MILK10k_*.csv`). The legacy‑only findings from the earlier audit (wrong tabular column names, KFold, multi‑label framing, resume, black‑image fallback, slow path walk) were removed with it.
- **Done since the audit:** fine‑tuned weights are saved (`fold{k}.pt`, `full.pt`); reference‑model predictions are cached (`reference.npz`); validation is predicted once per fold instead of every epoch; shared code lives in `notebooks/common.py`; `main_code/requirements.txt` exists.
- **Leaderboard regression 0.511 → 0.474 diagnosed** (`main_code/history.md` #14). It was not a code bug and not a CV drop (CV rose to 0.560). The test path changed from one full‑train model to the mean of 5 fold models + TTA, which CV never measures. `02_finetune` defaults are back to the 0.511 setup (`TEST_MODEL = "full"`, `TTA = False`). The other combinations are written to `outputs/finetune/variants/` for leaderboard A/B testing.
- **Environment:** neither the root `.venv` (no `transformers`, `lightgbm`) nor the system Python 3.13 (no `torchvision`, `lightgbm`) can run `02_finetune` end to end today. Install `main_code/requirements.txt` into the interpreter your Jupyter kernel uses.

## 2. Security & reliability

| # | Issue | Where | Fix |
|---|---|---|---|
| R1 | **Dataset and model weights are committed to git.** 10,480 train + 958 test JPEGs and the retired pipeline's 5 × `checkpoints/*.pth` (169 MB) are tracked; the pack is ≈ 500 MB and `.git` also holds **4.2 GB of orphaned `tmp_obj_*` files** from interrupted operations. MILK10k is CC‑BY‑NC; redistributing it via a (possibly public) GitHub remote needs attribution and a license check. | repo root | `git rm -r --cached MILK10k_*_Input checkpoints`, add them to the root `.gitignore`, then `git gc --prune=now`. To shrink history, use `git filter-repo` (rewrites history, so coordinate with collaborators). |
| R2 | **Test inference path is not what CV measures.** CV scores each lesion with one fold model; the submission uses a different model (full‑train, or a 5‑fold mean). This is how 0.560 CV became 0.474 on the leaderboard (`history.md` #14). | `02_finetune.ipynb` | Keep `TEST_MODEL = "full"` + `TTA = False` (the leaderboard‑verified setup). Treat any change to the test path as unverified until it holds on the leaderboard. |
| R3 | **Unpinned dependencies and no working environment.** Every package is `>=`, and no interpreter on the machine currently has the full set. | `main_code/requirements.txt` | Add a lock file (`pip freeze` or `uv lock`) from the kernel that produced the cached results. |

## 3. Performance & scalability

| # | Bottleneck | Impact | Suggestion |
|---|---|---|---|
| P1 | `01_baseline` extracts features with `num_workers=0` (notebook‑defined Dataset can't be pickled on Windows). | Slower first run | Move `ImageDataset` into a module (as was done for `ft_data.py`). |

## 4. Future refactoring & tech debt

### High impact
1. **Commit the reorganisation and fix the repo (R1).** `capstone-Gökhan/ → main_code/` currently shows as 17 deletions plus an untracked folder, and `OLD_Test_Folder/` as deletions. Commit it as a rename so history is kept (`git add -A capstone-Gökhan main_code OLD_Test_Folder`), and untrack the image data and `checkpoints/`.
2. **Headless runs.** On the RTX 5070 a full `02` run is ≈ 40 min, but it can only be driven from Jupyter. Use `jupytext`/`papermill`, or extract a `train_finetune.py`, so runs log to a file and can be reproduced from the command line.

### Medium impact
3. **Smoke test.** Add a `MAX_STEPS=2`, 2‑fold run on about 50 lesions plus data invariants (one row per lesion, both views present, label/feature index alignment). There is no automated test of any kind today.
4. **Single config.** Hyper‑parameters are spread across notebook cells. A small `config.yaml` (or a dataclass in `common.py`) would make experiments in `history.md` traceable to exact settings.
5. **Pin the environment (R3)** and document the GPU/CUDA requirement in one place.

### Low impact
6. Remove the legacy ResNet layer‑by‑layer cache fallback in `01.image_features` (`np.hstack([z[l] for l in LAYERS])`) once old caches are gone.
7. Unify documentation language (code comments and `history.md` are Turkish; the top‑level docs are English).
8. Add a `Dockerfile` for `main_code` if containerised runs are needed.

## 5. Beating 0.511 — prioritized modelling plan

Constraints from `history.md`: every gain so far came from the image model (lesson 1). Class weights, LR alternatives, stacking, fusion weights and per‑class thresholds all plateaued or overfit (#3, #4, #6, #10, #12). The leaderboard has σ ≈ 0.045 on 479 lesions, so decide on CV with the acceptance rule (≥ +0.01, majority of folds) and use the leaderboard only to confirm.

**Post‑processing, re‑checked on 2026‑11‑23 with nested CV on the cached OOF (no training).** Nothing passes the acceptance rule, so these are not the lever:

| Idea | 3‑model, no TTA (0.549) | 3‑model, TTA (0.560) |
|---|---|---|
| Learned blend weights (simplex, step 0.1) | +0.003, 3/5 | +0.009, 4/5 (stable ≈ 0.4 FT / 0.3 frozen / 0.3 tab) |
| Prior adjustment `p / prior^τ` (1 parameter) | −0.002, 2/5 | +0.006, 3/5 |
| Shared temperature before averaging | +0.008, 4/5 | −0.010, 0/5 |
| Per‑class log‑bias (11 parameters, #10 redo) | +0.002, 3/5 (in‑sample 0.566) | −0.008, 1/5 (in‑sample 0.573) |

**Ordered by expected macro‑F1 gain per GPU hour:**

1. **Leaderboard A/B of what already exists (no training, ~1 submission each).** Submit `variants/submission_full_notta.csv` (the re‑run 0.511 setup) and `variants/submission_full_tta.csv`. This separates TTA from fold‑averaging, which the 0.474 result confounds, and shows how much a mere re‑training moves the leaderboard (two fold models agree on only 88 % of test lesions).
2. **Fine‑tune DINOv2‑large instead of base.** Frozen large beat frozen base by +0.014 (#7), and fine‑tuning base added +0.04 over frozen. On the 12 GB RTX 5070, ViT‑L/14 at 224 px fits with fp16 and gradient checkpointing (≈ 3× base time, ~20 min per fold). Use layer‑wise LR decay (≈ 0.75) so the extra depth doesn't destabilise the 1e‑5 backbone LR. This is the single most likely +0.02–0.04.
3. **A second, architecturally different fine‑tuned backbone** (ConvNeXt‑V2‑B or EVA‑02‑B; optionally a dermatology‑pretrained ViT). Fine‑tuned + frozen were complementary (+0.02 over fine‑tuned alone, #13), and a CNN adds texture bias that the ViTs lack. Average it into the 1/3 slot of LightGBM only if CV says so; LightGBM barely contributes (#13) but halves fold variance.
4. **Rare‑class training signal** (BEN_OTH 44, DF 52, INF 50, VASC 47, MAL_OTH 9 make up 5/11 of the metric).
   - **Logit‑adjusted CE** (`logits + τ·log prior`, τ ≈ 1) instead of weighted CE. It gives the same balanced decision rule with better‑calibrated probabilities, so averaging with the other models doesn't wash out rare‑class peaks, the exact failure seen in #14.
   - **Class‑aware sampling** (square‑root frequency) so every batch sees rare lesions; history #4 shows "sqrt" weighting was competitive for the image model alone.
   - **External data** for the rare classes (ISIC archive DF/VASC/BEN_OTH dermoscopy), after checking the competition rules (already an open item in `history.md`).
5. **Higher input resolution for fine‑tuning** (resize 384 → crop 336; DINOv2 was pre‑trained at 518). #8's negative result was for removing the crop on a frozen ResNet, not for higher‑res crops on a fine‑tuned ViT. Cost ≈ 2.3× compute.
6. **Multi‑seed full‑train models** (2–3 seeds of `full.pt`) to reduce run‑to‑run variance on the leaderboard. Each model sees 100 % of the rare classes, unlike the 5 fold models. Validate on the leaderboard, since CV cannot score it either (R2).
7. **Learned blend weights 0.4 / 0.3 / 0.3** (table above): only if (2)–(4) leave the ensemble unchanged, and only after a leaderboard check; the CV gain is at the noise floor.

Lower priority or already disproven: OOF stacking with a meta‑learner (#6: −0.022; ~5k lesions are not enough for a balanced meta‑learner), per‑class thresholds (#10 and the redo above), tabular feature engineering (#11), and moving metadata into the network (tabular F1 is 0.33 and adds ≈ 0 to the ensemble).
