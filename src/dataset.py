import os
import zipfile
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

def extract_zip_if_needed(zip_path, extract_to):
    """Auto-extract zip archives if image folder does not exist."""
    if not os.path.exists(extract_to):
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path} to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            # Check for downloaded variations (e.g. .duckload extension)
            duckload_path = zip_path + ".duckload"
            if os.path.exists(duckload_path):
                print(f"Renaming {duckload_path} to {zip_path}...")
                os.rename(duckload_path, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

def prepare_paired_dataframe(metadata_path, ground_truth_path=None):
    """
    Pairs clinical close-up and dermoscopic images per lesion_id from the metadata file.
    """
    df_meta = pd.read_csv(metadata_path)
    
    # Standardize Lesion ID Column Name
    lesion_col = "lesion_id" if "lesion_id" in df_meta.columns else "lesion"
    image_id_col = "isic_id" if "isic_id" in df_meta.columns else "image_id"
    type_col = "image_type" if "image_type" in df_meta.columns else "type"

    # Separate Clinical and Dermoscopic Image Entries
    clinical_df = df_meta[df_meta[type_col].str.lower().str.contains("close|clin")].copy()
    dermo_df = df_meta[df_meta[type_col].str.lower().str.contains("dermo")].copy()

    clinical_df = clinical_df.rename(columns={image_id_col: "img_clinical"}).drop_duplicates(subset=[lesion_col])
    dermo_df = dermo_df.rename(columns={image_id_col: "img_dermoscopic"}).drop_duplicates(subset=[lesion_col])

    # Merge on Lesion ID
    paired_df = pd.merge(clinical_df, dermo_df[[lesion_col, "img_dermoscopic"]], on=lesion_col, how="inner")

    # Merge Ground Truth Labels if available
    if ground_truth_path and os.path.exists(ground_truth_path):
        df_gt = pd.read_csv(ground_truth_path)
        gt_lesion_col = "lesion" if "lesion" in df_gt.columns else "lesion_id"
        paired_df = pd.merge(paired_df, df_gt, left_on=lesion_col, right_on=gt_lesion_col, how="inner")

    return paired_df

def get_transforms(img_size=224, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

class MILKDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.tabular_features = self._extract_tabular_features(self.df)

    def _extract_tabular_features(self, df):
        # Identify MONET concept columns dynamically
        monet_cols = [c for c in df.columns if any(kw in c.lower() for kw in Config.MONET_KEYWORDS)]
        monet_vals = df[monet_cols].fillna(0.0).values.astype(np.float32) if len(monet_cols) > 0 else np.zeros((len(df), 7), dtype=np.float32)

        age = (df["age_group"].fillna(50) / 100.0).values.astype(np.float32)[:, None] if "age_group" in df.columns else np.zeros((len(df), 1), dtype=np.float32)
        skin = (df["skin_tone"].fillna(2.5) / 5.0).values.astype(np.float32)[:, None] if "skin_tone" in df.columns else np.zeros((len(df), 1), dtype=np.float32)

        sex = pd.get_dummies(df.get("sex", pd.Series(["unknown"]*len(df))), prefix="sex", dummy_na=True).astype(np.float32).values
        site = pd.get_dummies(df.get("anatomical_site", pd.Series(["unknown"]*len(df))), prefix="site", dummy_na=True).astype(np.float32).values

        tabular = np.hstack([monet_vals, age, skin, sex, site])
        return torch.tensor(tabular, dtype=torch.float32)

    def _load_image(self, img_name):
        # Support multiple image extension formats (.jpg, .jpeg, .png)
        for ext in ["", ".jpg", ".jpeg", ".png"]:
            path = os.path.join(self.img_dir, f"{img_name}{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Recursive check inside subdirectories
        for root, _, files in os.walk(self.img_dir):
            for file in files:
                if img_name in file:
                    img = cv2.imread(os.path.join(root, file))
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return np.zeros((224, 224, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_clin = self._load_image(str(row["img_clinical"]))
        img_derm = self._load_image(str(row["img_dermoscopic"]))

        if self.transform:
            img_clin = self.transform(image=img_clin)["image"]
            img_derm = self.transform(image=img_derm)["image"]

        tab_vector = self.tabular_features[idx]

        if not self.is_test:
            labels = torch.tensor(row[Config.CLASSES].values.astype(np.float32), dtype=torch.float32)
            return img_clin, img_derm, tab_vector, labels
        else:
            lesion_col = "lesion_id" if "lesion_id" in row else "lesion"
            return img_clin, img_derm, tab_vector, str(row[lesion_col])