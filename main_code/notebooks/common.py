"""01_baseline.ipynb ve 02_finetune.ipynb'nin ortak parçaları.

İki notebook'un sonuçları ancak aynı lezyonlar, aynı tabular özellikler ve aynı 5 fold kullanıldığında
kıyaslanabilir. Bu kod eskiden iki notebook'ta ayrı ayrı kopyalanmıştı; biri değişip diğeri değişmezse
kıyas sessizce bozulurdu. Artık tek kopya burada.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
IMAGE_TYPES = {"clinical: close-up": "clin", "dermoscopic": "derm"}
N_FOLDS = 5
SEED = 42


def load_split(data, split):
    """Metadata'daki lezyon başına iki satırı (klinik + dermoskopik) tek satıra indirir: hasta bilgileri,
    iki görüntünün MONET skorları (`clin_…`, `derm_…`) ve iki görüntünün dosya yolu (`clin`, `derm`)."""
    data = Path(data)
    meta = pd.read_csv(data / f"MILK10k_{split}_Metadata.csv")
    meta["kind"] = meta["image_type"].map(IMAGE_TYPES)
    meta["path"] = [str(data / f"MILK10k_{split}_Input" / l / f"{i}.jpg") for l, i in zip(meta["lesion_id"], meta["isic_id"])]
    monet_cols = [c for c in meta.columns if c.startswith("MONET_")]

    patient = meta.drop_duplicates("lesion_id").set_index("lesion_id")[["age_approx", "sex", "skin_tone_class", "site"]]
    monet = meta.pivot(index="lesion_id", columns="kind", values=monet_cols)
    monet.columns = [f"{kind}_{col}" for col, kind in monet.columns]
    paths = meta.pivot(index="lesion_id", columns="kind", values="path")
    return patient.join(monet).join(paths)


def lesion_ids(df):
    """Lezyon id'leri numpy metin dizisi olarak (cache'lerdeki lesion_id kontrolü için).
    `df.index.values.astype(str)` pandas 3'te numpy değil pandas dizisi veriyor; np.savez onu pickle'lıyor ve
    np.load (allow_pickle=False) geri okuyamıyor. np.asarray(..., dtype=str) pandas 2 ve 3'te aynı sonucu verir."""
    return np.asarray(df.index, dtype=str)


def load_labels(data, index):
    """One-hot GroundTruth -> `index` sırasıyla sınıf indeksi (0–10)."""
    gt = pd.read_csv(Path(data) / "MILK10k_Training_GroundTruth.csv").set_index("lesion_id").loc[index]
    assert list(gt.columns) == CLASSES
    return gt.values.argmax(1)


def tabular_features(df, sites):
    """18 tabular özellik. Görüntü yolları çıkarılır; sex/site sabit kategorili (train/test aynı kodlanır).
    Eksik değerlere dokunulmaz, LightGBM bunları kendisi işliyor."""
    X = df.drop(columns=list(IMAGE_TYPES.values()))
    X["sex"] = pd.Categorical(X["sex"], categories=["female", "male"])
    X["site"] = pd.Categorical(X["site"], categories=sites)
    return X


def make_folds(y):
    """Bütün deneylerin ortak fold'ları: lezyon bazında StratifiedKFold(5, shuffle, seed 42).
    Bölünme sadece y'ye (ve örnek sayısına) bağlı, özellik matrisine değil."""
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def tabular_model():
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          class_weight="balanced", random_state=SEED, verbose=-1)


def image_model():
    """Donmuş görüntü özellikleri (DINOv2-large) üzerinde LR. C=0.03 DINOv2-large için seçildi (history.md #7, #12)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=3000))


frozen_model = image_model  # 02_finetune'daki adı


def take(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def oof_proba(make_model, X, y, folds):
    """Out-of-fold olasılıklar: her fold'da model kalan fold'larla eğitilip o fold'u tahmin eder."""
    proba = np.zeros((len(y), len(CLASSES)))
    for tr, va in folds:
        proba[va] = make_model().fit(take(X, tr), y[tr]).predict_proba(take(X, va))
    return proba
