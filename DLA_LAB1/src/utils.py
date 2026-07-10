"""
Modulo di utilità generali per l'addestramento e la valutazione.
"""

import torch
import numpy as np
import pandas as pd
import random
import re
from pathlib import Path
from tqdm import tqdm

def set_seed(seed=42):
    """
    Imposta il seed per la riproducibilità degli esperimenti in numpy e PyTorch.

    Args:
        seed (int): Valore del seed (default: 42).

    Note:
        Richiamata nei notebook con training o estrazione feature stocastica:
        03_baseline_features_svm, 04_finetuning_baseline, 05_pipeline_experiments
        (Esercizio 2); 02_resnet50_pretraining, 04_faster_rcnn_training (Esercizio 3).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Rende le operazioni convoluzionali deterministiche, ma potrebbe impattare sulle performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def extract_features(model, loader, device, desc="Estraendo feature"):
    """
    Funzione helper per estrarre le feature in blocco da un dataloader usando un modello.
    
    Args:
        model (torch.nn.Module): Modello (Feature Extractor).
        loader (torch.utils.data.DataLoader): DataLoader contenente le immagini e le label.
        device (torch.device): Device di elaborazione (es. 'cuda' o 'cpu').
        desc (str): Stringa da mostrare nella barra di progresso (tqdm).

    Returns:
        tuple: (features_tensor, classes_tensor) restituiti sulla CPU pronti per algoritmi come SVM.

    Note:
        Richiamata in 03_baseline_features_svm (Esercizio 2).
    """
    # Mette il modello in modalità valutazione (disabilita Dropout, aggiornamenti BatchNorm, ecc.)
    model.eval() 
    
    all_feats = []
    all_classes = []
    
    # Non dobbiamo calcolare gradienti, risparmiando tempo e VRAM
    with torch.no_grad():
        for ims, cls in tqdm(loader, desc=desc):
            ims = ims.to(device)
            # Otteniamo le feature: shape (batch_size, 512)
            feats = model(ims)
            
            # Spostiamo i risultati su CPU e li accumuliamo
            all_feats.append(feats.cpu())
            all_classes.append(cls)
            
    # Concateniamo tutte le liste in due grandi tensori 2D/1D
    return torch.vstack(all_feats), torch.concat(all_classes)

def load_epoch_metrics(csv_path, columns=("train/loss_epoch", "val/loss", "val_f1_score")):
    """
    Legge un file metrics.csv prodotto da PyTorch Lightning CSVLogger e lo aggrega
    per epoca, cosi' da ottenere una riga per epoca con le colonne richieste.
    Le colonne non presenti nel file (ad es. metriche non loggate) vengono ignorate.

    Args:
        csv_path (str): Percorso del file metrics.csv.
        columns (tuple): Nomi delle colonne da estrarre e aggregare per epoca.

    Returns:
        pandas.DataFrame: DataFrame indicizzato per epoca con le colonne disponibili.

    Note:
        Richiamata in 05_pipeline_experiments (Esercizio 2); 02_resnet50_pretraining
        e 04_faster_rcnn_training (Esercizio 3).
    """
    df = pd.read_csv(csv_path)
    available_columns = [c for c in columns if c in df.columns]
    return df.groupby("epoch")[available_columns].mean()

def load_final_test_metrics(csv_path, columns=("test/accuracy", "test/f1_score")):
    """
    Legge un file metrics.csv prodotto da PyTorch Lightning CSVLogger ed estrae
    l'ultima riga valida per le metriche di test (loggate una sola volta da trainer.test()).

    Args:
        csv_path (str): Percorso del file metrics.csv.
        columns (tuple): Nomi delle colonne di test da estrarre.

    Returns:
        pandas.Series: Ultimi valori disponibili per le colonne richieste.

    Note:
        Richiamata in 05_pipeline_experiments (Esercizio 2).
    """
    df = pd.read_csv(csv_path)
    available_columns = [c for c in columns if c in df.columns]
    return df[available_columns].dropna(how="all").iloc[-1]


def find_best_metric_checkpoint(checkpoint_dir, filename_prefix, metric_name):
    """
    Trova il checkpoint con il valore di metrica piu' alto codificato nel nome.

    Args:
        checkpoint_dir (str | pathlib.Path): Cartella che contiene i checkpoint.
        filename_prefix (str): Prefisso dei file da considerare.
        metric_name (str): Nome della metrica incluso nel filename.

    Returns:
        pathlib.Path: Percorso del checkpoint con la metrica piu' alta.

    Raises:
        FileNotFoundError: Se non esistono checkpoint compatibili.

    Note:
        Richiamata in 04_faster_rcnn_training e 05_evaluation_and_inference
        (Esercizio 3).
    """
    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(
        rf"{re.escape(metric_name)}=([0-9]+(?:\.[0-9]+)?)(?:-v\d+)?\.ckpt$"
    )

    candidates = []
    for checkpoint_path in checkpoint_dir.rglob(f"{filename_prefix}*.ckpt"):
        match = pattern.search(checkpoint_path.name)
        if match:
            candidates.append((float(match.group(1)), checkpoint_path))

    if not candidates:
        raise FileNotFoundError(
            f"Nessun checkpoint '{filename_prefix}*.ckpt' con metrica "
            f"'{metric_name}' trovato in {checkpoint_dir}."
        )

    return max(candidates, key=lambda item: (item[0], item[1].name))[1]
