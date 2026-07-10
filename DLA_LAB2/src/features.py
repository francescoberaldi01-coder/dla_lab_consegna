"""Feature extraction con DistilBERT per la baseline dell'Esercizio 1."""

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "distilbert/distilbert-base-uncased"


def load_distilbert_components(model_name: str = MODEL_NAME, device: torch.device | None = None):
    """Carica tokenizer e modello DistilBERT base.

    Note:
        Richiamata in 01_eda_dataset_and_tokenizer e 02_feature_extraction_baseline
        (Esercizio 1).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is not None:
        model = model.to(device)
    return tokenizer, model


def extract_cls_features(
    texts: Iterable[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
):
    """Estrae il vettore [CLS] dell'ultimo layer per una lista di testi.

    Note:
        Richiamata in 02_feature_extraction_baseline (Esercizio 1).
    """
    model.eval()
    loader = DataLoader(list(texts), batch_size=batch_size, shuffle=False)
    features = []

    with torch.no_grad():
        for batch_texts in tqdm(loader, desc="Extracting CLS features"):
            encoded = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            cls_vectors = outputs.last_hidden_state[:, 0, :]
            features.append(cls_vectors.cpu())

    return torch.cat(features, dim=0)
