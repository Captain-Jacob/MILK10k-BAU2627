import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import Config
from src.dataset import extract_zip_if_needed, prepare_paired_dataframe, MILKDataset, get_transforms
from src.model import MultimodalMILKNet
from src.loss import AsymmetricLoss

# Enforce GPU Execution
if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available! Training requires a GPU.")

print(f"--> [OVERNIGHT ENGINE INITIALIZED] Target GPU: {torch.cuda.get_device_name(0)}")
torch.backends.cudnn.benchmark = True

def compute_macro_f1(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs >= threshold).astype(int)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def train_overnight():
    # 1. Unpack Datasets
    extract_zip_if_needed(Config.TRAIN_ZIP, Config.TRAIN_IMG_DIR)
    extract_zip_if_needed(Config.TEST_ZIP, Config.TEST_IMG_DIR)

    df_full = prepare_paired_dataframe(Config.TRAIN_METADATA_CSV, Config.TRAIN_GROUND_TRUTH_CSV)
    
    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
    fold_splits = list(kf.split(df_full))

    # Resumption tracking variables
    start_fold = 1
    start_epoch = 0
    scores = []
    checkpoint_state = None

    # Check for power-loss recovery checkpoint
    if os.path.exists(Config.RESUME_CHECKPOINT):
        print(f"--> [AUTO-RESUME FOUND] Recovering state from '{Config.RESUME_CHECKPOINT}'...")
        checkpoint_state = torch.load(Config.RESUME_CHECKPOINT, map_location=Config.DEVICE)
        start_fold = checkpoint_state["fold"]
        start_epoch = checkpoint_state["epoch"] + 1
        scores = checkpoint_state.get("scores", [])
        print(f"--> Resuming at Fold {start_fold}/{Config.NUM_FOLDS}, Epoch {start_epoch + 1}/{Config.EPOCHS}")

    # Initialize Dataset
    dummy_dataset = MILKDataset(df_full, Config.TRAIN_IMG_DIR, transform=get_transforms(Config.IMAGE_SIZE, is_train=False))
    tab_dim = dummy_dataset[0][2].shape[0]

    for fold_idx in range(start_fold - 1, Config.NUM_FOLDS):
        fold = fold_idx + 1
        print(f"\n==================== Training Fold {fold}/{Config.NUM_FOLDS} ====================")
        
        train_idx, val_idx = fold_splits[fold_idx]
        train_df = df_full.iloc[train_idx]
        val_df = df_full.iloc[val_idx]

        train_dataset = MILKDataset(train_df, Config.TRAIN_IMG_DIR, transform=get_transforms(Config.IMAGE_SIZE, is_train=True))
        val_dataset = MILKDataset(val_df, Config.TRAIN_IMG_DIR, transform=get_transforms(Config.IMAGE_SIZE, is_train=False))

        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=(Config.NUM_WORKERS > 0)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=Config.NUM_WORKERS, 
            pin_memory=True,
            persistent_workers=(Config.NUM_WORKERS > 0)
        )

        model = MultimodalMILKNet(
            backbone_name=Config.BACKBONE,
            tabular_input_dim=tab_dim,
            num_classes=Config.NUM_CLASSES,
            pretrained=True
        ).to(Config.DEVICE)

        criterion = AsymmetricLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
        scaler = torch.amp.GradScaler('cuda')  # AMP for RTX Tensor Cores

        best_macro_f1 = 0.0

        # Load weights if resuming within the same fold
        if checkpoint_state is not None and checkpoint_state["fold"] == fold:
            model.load_state_dict(checkpoint_state["model_state"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state"])
            scheduler.load_state_dict(checkpoint_state["scheduler_state"])
            scaler.load_state_dict(checkpoint_state["scaler_state"])
            best_macro_f1 = checkpoint_state["best_macro_f1"]
            checkpoint_state = None  # Reset state container after applying
        else:
            start_epoch = 0

        for epoch in range(start_epoch, Config.EPOCHS):
            model.train()
            running_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1:02d}/{Config.EPOCHS} [Train]", leave=False)
            for img_clin, img_derm, tab, labels in train_pbar:
                img_clin = img_clin.to(Config.DEVICE, non_blocking=True)
                img_derm = img_derm.to(Config.DEVICE, non_blocking=True)
                tab = tab.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)

                optimizer.zero_grad()

                # FP16 Automatic Mixed Precision
                with torch.amp.autocast('cuda'):
                    logits = model(img_clin, img_derm, tab)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()

            # Validation Loop
            model.eval()
            val_targets, val_preds = [], []
            val_pbar = tqdm(val_loader, desc=f"Fold {fold} | Epoch {epoch+1:02d}/{Config.EPOCHS} [Val]", leave=False)
            
            with torch.no_grad():
                for img_clin, img_derm, tab, labels in val_pbar:
                    img_clin = img_clin.to(Config.DEVICE, non_blocking=True)
                    img_derm = img_derm.to(Config.DEVICE, non_blocking=True)
                    tab = tab.to(Config.DEVICE, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        logits = model(img_clin, img_derm, tab)
                        probs = torch.sigmoid(logits)

                    val_targets.append(labels.numpy())
                    val_preds.append(probs.cpu().numpy())

            val_targets = np.vstack(val_targets)
            val_preds = np.vstack(val_preds)
            val_f1 = compute_macro_f1(val_targets, val_preds, threshold=0.5)

            avg_loss = running_loss / len(train_loader)
            print(f"Fold {fold} | Epoch {epoch+1:02d}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}")

            # Save best fold model
            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                torch.save(model.state_dict(), f"{Config.OUTPUT_DIR}/model_fold_{fold}.pth")

            # --- AUTOSAVE CHECKPOINT FOR POWER LOSS / REBOOT RECOVERY ---
            save_checkpoint = {
                "fold": fold,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_macro_f1": best_macro_f1,
                "scores": scores
            }
            torch.save(save_checkpoint, Config.RESUME_CHECKPOINT)

        scores.append(best_macro_f1)
        start_epoch = 0  # Reset epoch start index for subsequent folds

    # Cleanup temporary recovery file upon full completion
    if os.path.exists(Config.RESUME_CHECKPOINT):
        os.remove(Config.RESUME_CHECKPOINT)

    print(f"\n==================================================")
    print(f"ALL 5 FOLDS COMPLETE! Mean CV Macro F1: {np.mean(scores):.4f}")
    print(f"You can now run 'python inference_and_submission.py' to build your submission file.")
    print(f"==================================================")

if __name__ == "__main__":
    train_overnight()