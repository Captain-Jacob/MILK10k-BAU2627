"""02_finetune.ipynb için veri yükleme.

Bu kod notebook yerine ayrı dosyada duruyor: Windows'ta DataLoader'ın paralel worker'ları
notebook içinde tanımlanan sınıfları yükleyemiyor. Dosyada olunca num_workers > 0 çalışıyor
ve eğitim belirgin şekilde hızlanıyor.
"""
import random
from pathlib import Path

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

SMALL = Path(__file__).resolve().parents[1] / "outputs" / "cache" / "img256"
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def small_path(path):
    """Orijinal görüntünün küçültülmüş (kısa kenar 256 px) kopyasının yolu."""
    path = Path(path)
    return SMALL / f"{path.parent.name}_{path.name}"


def preprocess(paths):
    """Görüntüleri bir kez küçültüp JPEG olarak kaydeder; her epoch'ta 600x450 JPEG çözmekten çok daha hızlı."""
    SMALL.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not small_path(p).exists()]
    for p in tqdm(todo, desc="küçültme"):
        T.functional.resize(Image.open(p).convert("RGB"), 256, interpolation=T.InterpolationMode.BICUBIC) \
            .save(small_path(p), quality=95)


class RandomRot90:
    """0/90/180/270 derece döndürme. Dermoskopide 'yukarı' yönü yok; kare görüntüde kayıpsız."""

    def __call__(self, img):
        return img.rotate(90 * random.randint(0, 3))


TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      RandomRot90(), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), NORM])
EVAL_TF = T.Compose([T.CenterCrop(224), T.ToTensor(), NORM])


class LesionDataset(Dataset):
    """Her örnek bir lezyon: (klinik görüntü, dermoskopik görüntü, etiket)."""

    def __init__(self, clin_paths, derm_paths, labels, train):
        self.clin = [small_path(p) for p in clin_paths]
        self.derm = [small_path(p) for p in derm_paths]
        self.labels = labels
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        clin = self.tf(Image.open(self.clin[i]).convert("RGB"))
        derm = self.tf(Image.open(self.derm[i]).convert("RGB"))
        return clin, derm, self.labels[i]
