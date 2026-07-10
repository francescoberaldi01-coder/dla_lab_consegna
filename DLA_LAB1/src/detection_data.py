"""
Modulo per il caricamento del dataset di Object Detection GTSRB.

I dati provengono dal dataset Hugging Face `keremberke/german-traffic-sign-detection`,
che fornisce fotogrammi stradali completi con una o piu' bounding box per immagine
(a differenza del dataset di classificazione GTSRB, gia' ritagliato attorno a un
singolo segnale per immagine).
"""

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

_HF_DATASET_NAME = "keremberke/german-traffic-sign-detection"
_HF_CONFIG_NAME = "full"

# Mappa tra gli split richiesti dalla pipeline e i nomi degli split su Hugging Face.
_SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}


class GTSRBDetectionDataset(Dataset):
    """
    Dataset di Object Detection per GTSRB, basato sul dataset Hugging Face
    `keremberke/german-traffic-sign-detection`. Ogni immagine puo' contenere
    piu' segnali stradali, con bounding box fornite in formato COCO (x, y, width, height).
    """

    def __init__(self, split="train", cache_dir=None, transforms=None):
        if split not in _SPLIT_MAP:
            raise ValueError(f"Split non supportato: {split}")

        self.transforms = transforms
        self.dataset = load_dataset(
            _HF_DATASET_NAME,
            name=_HF_CONFIG_NAME,
            split=_SPLIT_MAP[split],
            cache_dir=cache_dir,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset[idx]

        img = record["image"].convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        objects = record["objects"]
        # reshape(-1, 4) garantisce la forma (0, 4), e non (0,), quando l'immagine
        # non contiene alcun segnale annotato (scena di sfondo/negativa).
        boxes = torch.tensor(
            [self._xywh_to_xyxy(box) for box in objects["bbox"]],
            dtype=torch.float32,
        ).reshape(-1, 4)

        # +1 per riservare l'indice 0 allo sfondo, come richiesto da Faster R-CNN.
        labels = torch.tensor([c + 1 for c in objects["category"]], dtype=torch.int64)
        area = torch.tensor(objects["area"], dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        image_id = torch.tensor([record["image_id"]])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        return img, target

    @staticmethod
    def _xywh_to_xyxy(box):
        """
        Converte una bounding box dal formato COCO [x, y, width, height]
        al formato [x1, y1, x2, y2] richiesto da Faster R-CNN.
        """
        x, y, w, h = box
        return [x, y, x + w, y + h]


def detection_collate_fn(batch):
    """
    Raggruppa le tuple del dataset in modo compatibile con Faster R-CNN.
    Restituisce: tuple(images), tuple(targets).
    """
    return tuple(zip(*batch))


class LitGTSRBDetectionDataModule(pl.LightningDataModule):
    """
    DataModule Lightning per l'Object Detection su GTSRB, basato sugli split
    ufficiali (train/validation/test) del dataset Hugging Face.

    Note:
        Istanziata in 03_object_detection_dataset, 04_faster_rcnn_training e
        05_evaluation_and_inference (Esercizio 3).
    """

    def __init__(self, cache_dir: str = None, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Faster R-CNN gestisce resize e normalizzazione interna; qui basta il tensore in [0, 1].
        self.transform = T.ToTensor()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = GTSRBDetectionDataset(
                split="train",
                cache_dir=self.cache_dir,
                transforms=self.transform,
            )
            self.val_dataset = GTSRBDetectionDataset(
                split="val",
                cache_dir=self.cache_dir,
                transforms=self.transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = GTSRBDetectionDataset(
                split="test",
                cache_dir=self.cache_dir,
                transforms=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=detection_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )
