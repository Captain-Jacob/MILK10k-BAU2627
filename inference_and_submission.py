import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from config import Config
from src.dataset import extract_zip_if_needed, prepare_paired_dataframe, MILKDataset, get_transforms
from src.model import MultimodalMILKNet

def run_inference():
    extract_zip_if_needed(Config.TEST_ZIP, Config.TEST_IMG_DIR)
    
    test_df = prepare_paired_dataframe(Config.TEST_METADATA_CSV)
    test_dataset = MILKDataset(test_df, Config.TEST_IMG_DIR, transform=get_transforms(Config.IMAGE_SIZE, is_train=False), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    tab_dim = test_dataset[0][2].shape[0]
    fold_preds = []

    for fold in range(1, Config.NUM_FOLDS + 1):
        ckpt = f"{Config.OUTPUT_DIR}/model_fold_{fold}.pth"
        if os.path.exists(ckpt):
            model = MultimodalMILKNet(
                backbone_name=Config.BACKBONE,
                tabular_input_dim=tab_dim,
                num_classes=Config.NUM_CLASSES,
                pretrained=False
            ).to(Config.DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=Config.DEVICE))
            model.eval()

            preds_list = []
            with torch.no_grad():
                for img_clin, img_derm, tab, _ in test_loader:
                    img_clin, img_derm = img_clin.to(Config.DEVICE), img_derm.to(Config.DEVICE)
                    tab = tab.to(Config.DEVICE)

                    logits = model(img_clin, img_derm, tab)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds_list.append(probs)

            fold_preds.append(np.vstack(preds_list))

    # Compute average predictions across folds
    ensemble_preds = np.mean(fold_preds, axis=0)

    # Construct final ISIC submission file
    sub_df = pd.DataFrame()
    lesion_col = "lesion_id" if "lesion_id" in test_df.columns else "lesion"
    sub_df["lesion"] = test_df[lesion_col]

    for i, class_name in enumerate(Config.CLASSES):
        sub_df[class_name] = ensemble_preds[:, i]

    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission generated successfully: '{Config.SUBMISSION_PATH}'")
    print(sub_df.head())

if __name__ == "__main__":
    run_inference()