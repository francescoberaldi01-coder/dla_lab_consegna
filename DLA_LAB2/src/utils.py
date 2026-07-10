"""Utilita' generali per gli esperimenti del Lab 2."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Imposta il seed per rendere gli esperimenti piu' riproducibili.

    Note:
        Richiamata in 01_eda_dataset_and_tokenizer e 02_feature_extraction_baseline
        (Esercizio 1); in 01_distilbert_finetuning (Esercizio 2).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True, force_cpu: bool = False) -> torch.device:
    """Restituisce il miglior device PyTorch disponibile su Windows, macOS o CPU.

    Stessa logica di `DLA_LAB3/src/utils.py`, per avere un comportamento
    coerente su tutti i laboratori quando si esegue su Mac con MPS.

    Note:
        Richiamata in 01_eda_dataset_and_tokenizer e 02_feature_extraction_baseline
        (Esercizio 1); in 01_distilbert_finetuning (Esercizio 2); in
        02_clip_retrieval_demo (Esercizio 3).
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
