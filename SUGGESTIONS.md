# Engineering Suggestions

Findings from a full audit of the repository (2026‑11‑23). Repository status was updated on 2026‑11‑26. Items are ordered by impact within each section.

## 1. Status (2026‑11‑26)

- **Done: final verification.** `evaluate_models.py` was re‑run from the existing caches (`main_code/history.md` #15). All classification metrics, per‑class and per‑fold tables, confusion matrices, OOF files, sizes and parameter counts are identical to the previous run; only latency moved (per‑model means up to 3.8 % lower). No model was fine‑tuned and `outputs/finetune/` is unchanged. The notebooks were not re‑executed.
- **Done: Windows user name removed** from the saved outputs of `01_baseline.ipynb` and `02_finetune.ipynb` (replaced with `user`; local `D:\…` paths remain, and earlier git commits are unchanged).
- **Open: commit.** At the time of the verification, the 2026‑11‑26 changes (including the weight‑file deletions and the new `TECHNICAL_NOTES_FOR_TEAM.md`) were not yet committed.
- **Open from this audit:** R1, R3, P1 and refactoring items 2–7. R2 is mitigated by the defaults but remains true of the pipeline. Item 1 is the same task as R1.
- **Legacy pipeline removed.** The EfficientNet‑B0 scripts left the working tree on 2026‑11‑23 and their five weight files on 2026‑11‑26; both remain in git history (commit `91f9cf46`). The metadata/ground‑truth CSVs in `main_code/data/` are tracked through a `.gitignore` exception (`data/*` + `!data/MILK10k_*.csv`).
- **Cleanup 2026‑11‑26.** The unused ResNet layer‑by‑layer cache fallback was removed from `01_baseline.ipynb` and `evaluate_models.py`.
- **Model comparison added.** `main_code/evaluate_models.py` writes the MCDM tables to `main_code/outputs/mcdm/`.
- **Done since the audit:** fine‑tuned weights are saved (`fold{k}.pt`, `full.pt`); reference‑model predictions are cached (`reference.npz`); validation is predicted once per fold instead of every epoch; shared code lives in `notebooks/common.py`; `main_code/requirements.txt` exists.
- **Leaderboard regression 0.511 → 0.474 diagnosed** (`main_code/history.md` #14). It was not a code bug and not a CV drop (CV rose to 0.560). The test path changed from one full‑train model to the mean of 5 fold models + TTA, which CV never measures. `02_finetune` defaults are back to the 0.511 setup (`TEST_MODEL = "full"`, `TTA = False`). The other combinations are written to `outputs/finetune/variants/` for leaderboard A/B testing.
- **Environment (2026‑11‑26):** the root `.venv` has the model libraries (it can import `evaluate_models.py`) but no Jupyter or `ipykernel`; the system Python 3.13 lacks `torchvision` and `lightgbm`. Install `main_code/requirements.txt` into the interpreter your Jupyter kernel uses.

## 2. Security & reliability

| # | Issue | Where | Fix |
|---|---|---|---|
| R1 | **Dataset is committed to git, and old weights remain in its history.** 10,480 train + 958 test JPEGs are tracked, and the removed legacy `.pth` files (169 MB) stay in history; the pack is ≈ 500 MB and `.git` also holds **≈ 4.2 GiB of orphaned `tmp_obj_*` files** from interrupted operations. MILK10k is CC‑BY‑NC; redistributing it via a (possibly public) GitHub remote needs attribution and a license check. | repo root | `git rm -r --cached MILK10k_*_Input`, add them to the root `.gitignore`, then `git gc --prune=now`. To shrink history, use `git filter-repo` (rewrites history, so coordinate with collaborators). |
| R2 | **Test inference path is not what CV measures.** CV scores each lesion with one fold model; the submission uses a different model (full‑train, or a 5‑fold mean). This is how 0.560 CV became 0.474 on the leaderboard (`history.md` #14). | `02_finetune.ipynb` | Keep `TEST_MODEL = "full"` + `TTA = False` (the leaderboard‑verified setup). Treat any change to the test path as unverified until it holds on the leaderboard. |
| R3 | **Unpinned dependencies.** Every package is `>=` or unpinned and there is no lock file (`outputs/mcdm/run_info.json` records the versions of the evaluation run only). | `main_code/requirements.txt` | Add a lock file (`pip freeze` or `uv lock`) from the kernel that produced the cached results. |

## 3. Performance & scalability

| # | Bottleneck | Impact | Suggestion |
|---|---|---|---|
| P1 | `01_baseline` extracts features with `num_workers=0` (notebook‑defined Dataset can't be pickled on Windows). | Slower first run | Move `ImageDataset` into a module (as was done for `ft_data.py`). |

## 4. Future refactoring & tech debt

### High impact
1. **Untrack the image data (R1).**
2. **Headless runs.** On the RTX 5070 the 5‑fold part of `02` takes ≈ 35 min (≈ 45 min estimated with the final full‑train fit), but it can only be driven from Jupyter. Use `jupytext`/`papermill`, or extract a `train_finetune.py`, so runs log to a file and can be reproduced from the command line.

### Medium impact
3. **Smoke test.** Add a `MAX_STEPS=2`, 2‑fold run on about 50 lesions plus data invariants (one row per lesion, both views present, label/feature index alignment). There is no automated test of any kind today.
4. **Single config.** Hyper‑parameters are spread across notebook cells. A small `config.yaml` (or a dataclass in `common.py`) would make experiments in `history.md` traceable to exact settings.
5. **Pin the environment (R3)** and document the GPU/CUDA requirement in one place.

### Low impact
6. Unify documentation language (code comments and `history.md` are Turkish; the top‑level docs are English).
7. Add a `Dockerfile` for `main_code` if containerised runs are needed.

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

1. **Leaderboard A/B of what already exists (~1 submission each).** Submit `variants/submission_full_notta.csv` (the re‑run 0.511 setup) and `variants/submission_full_tta.csv`. Neither exists in the current checkout: first run the final cell of `02_finetune` to train `full.pt` (this also overwrites `submission.csv`). This separates TTA from fold‑averaging, which the 0.474 result confounds, and shows how much a mere re‑training moves the leaderboard (three‑model ensembles that differ only in which fold model supplies the fine‑tuned part agree on 88 % of test lesions on average without TTA; the fine‑tuned models alone agree on 79 %).
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
