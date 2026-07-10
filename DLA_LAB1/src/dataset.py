"""
Modulo per la gestione del dataset, delle trasformazioni e dei Dataloader.
"""

import torch
from torchvision.transforms import v2
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(data_dir='../_data'):
    """
    Calcola i pesi di classe sullo split di training ufficiale GTSRB. La
    successiva divisione train/validation è stratificata, quindi mantiene le
    stesse proporzioni di classe senza usare immagini o metriche di validation.

    Args:
        data_dir (str): Percorso della cartella contenente i dati.

    Returns:
        torch.Tensor: Un peso per classe, ordinato per class_id crescente.

    Note:
        Richiamata in 05_pipeline_experiments (Esercizio 2) e
        02_resnet50_pretraining (Esercizio 3).
    """
    data_path = Path(data_dir)
    dataset = datasets.GTSRB(root=data_path, split='train', download=True)
    # Accesso diretto a _samples per leggere le label senza dover decodificare le immagini.
    targets = [s[1] for s in dataset._samples]
    classes = np.unique(targets)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=targets)
    return torch.tensor(weights, dtype=torch.float32)

def get_transforms(augmentation_strategy="baseline", normalization="gtsrb"):
    """
    Definisce e restituisce le trasformazioni di data augmentation per il training
    e le trasformazioni deterministiche per il testing.
    
    Args:
        augmentation_strategy (str): Strategia di augmentation del training set.
        normalization (str): Statistiche di normalizzazione da applicare. Usa
            ``"imagenet"`` per un estrattore ImageNet congelato e ``"gtsrb"``
            per gli esperimenti di fine-tuning del laboratorio.

    Returns:
        tuple: (train_transform, test_transform)
    """
    normalization_stats = {
        "gtsrb": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }
    if normalization not in normalization_stats:
        raise ValueError(f"Normalizzazione non supportata: {normalization}")
    image_mean, image_std = normalization_stats[normalization]

    # Trasformazione per il Training: include augmentation (RandomCrop)
    if augmentation_strategy == "none":
        # Nessuna augmentation: identica alla trasformazione di test, deterministica.
        train_transform = v2.Compose([
            v2.Resize((68, 68)),
            v2.CenterCrop(64),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std)
        ])
    elif augmentation_strategy == "extreme":
        train_transform = v2.Compose([
            v2.Resize((68, 68)),
            v2.RandomCrop(64),
            v2.RandomRotation(15),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # Simulazione dashcam
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2), # Motion blur
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Ostacoli sul cartello
            v2.Normalize(mean=image_mean, std=image_std)
        ])
    elif augmentation_strategy == "heavy_blur":
        train_transform = v2.Compose([
            v2.Resize((68, 68)),
            v2.RandomCrop(64),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2), # Solo il rumore Gaussiano aggiunto alla heavy
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std)
        ])
    elif augmentation_strategy == "heavy":
        train_transform = v2.Compose([
            v2.Resize((68, 68)),
            v2.RandomCrop(64),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std)
        ])
    else:
        train_transform = v2.Compose([
        v2.Resize((68, 68)),           # Resize leggermente più grande per permettere il crop
        v2.RandomCrop(64),             # Ritaglio casuale per data augmentation
        v2.ToImage(),                  # Converte PIL Image o array numpy in Tensor
        v2.ToDtype(torch.float32, scale=True), # Scala i valori in [0, 1]
        v2.Normalize(mean=image_mean, std=image_std)
    ])

    # Trasformazione per il Test: deterministica (CenterCrop)
    test_transform = v2.Compose([
        v2.Resize((68, 68)),
        v2.CenterCrop(64),             # Ritaglio centrale esatto, nessuna casualità
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=image_mean, std=image_std)
    ])
    
    return train_transform, test_transform

def get_dataloaders(
    data_dir="../_data",
    batch_size=128,
    num_workers=4,
    device=None,
    augmentation_strategy="baseline",
    normalization="gtsrb",
):
    """
    Inizializza i dataset GTSRB e crea i relativi DataLoader pronti per l'uso.
    
    Args:
        data_dir (str): Percorso della cartella contenente i dati.
        batch_size (int): Numero di campioni per batch.
        num_workers (int): Numero di processi per il caricamento dei dati.
        device (torch.device): Se specificato come cuda, abilita il pin_memory.
        augmentation_strategy (str): Strategia di augmentation per il training set,
            vedi get_transforms per le opzioni disponibili.
        normalization (str): Statistiche di normalizzazione, ``"gtsrb"`` oppure
            ``"imagenet"``.

    Returns:
        tuple: (train_loader, val_loader, test_loader)

    Note:
        Richiamata in 02_preprocessing, 03_baseline_features_svm,
        04_finetuning_baseline, 05_pipeline_experiments (Esercizio 2) e
        02_resnet50_pretraining (Esercizio 3).
    """
    train_transform, test_transform = get_transforms(augmentation_strategy, normalization)
    
    data_path = Path(data_dir)
    
    # Caricamento del dataset di training (due istanze per avere trasformazioni separate)
    train_dataset_full = datasets.GTSRB(
        root=data_path,
        split='train',
        download=True,
        transform=train_transform
    )
    
    val_dataset_full = datasets.GTSRB(
        root=data_path,
        split='train',
        download=True,
        transform=test_transform
    )
    
    # Divisione 80% / 20% con stratificazione
    dataset_size = len(train_dataset_full)
    indices = list(range(dataset_size))
    targets = [s[1] for s in train_dataset_full._samples]
    
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets
    )
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_idx)

    # Caricamento del dataset di test puro
    test_dataset = datasets.GTSRB(
        root=data_path,
        split='test',
        download=True,
        transform=test_transform
    )
    
    # Determinare l'uso di pin_memory in base al device
    pin_memory = False
    if device is not None and device.type == 'cuda':
        pin_memory = True

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,          # Shuffle attivato solo per il training
        num_workers=num_workers,
        pin_memory=pin_memory,
        # I worker restano vivi tra un'epoca e l'altra invece di essere ricreati da zero;
        # ha effetto solo se num_workers > 0.
        persistent_workers=(num_workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # I worker restano vivi tra un'epoca e l'altra invece di essere ricreati da zero;
        # ha effetto solo se num_workers > 0.
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,         # Shuffle disattivato per il test
        num_workers=num_workers,
        pin_memory=pin_memory,
        # I worker restano vivi tra un'epoca e l'altra invece di essere ricreati da zero;
        # ha effetto solo se num_workers > 0.
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader
