"""Componenti riutilizzabili per il retrieval text-to-image con CLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import (
    Dataset,
    Features,
    Image as DatasetImage,
    Sequence,
    Value,
    get_dataset_split_names,
    load_dataset,
)
from PIL import Image as PILImage
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


PREFERRED_IMAGE_COLUMNS = ("image", "img", "picture", "photo")
PREFERRED_CAPTION_COLUMNS = ("caption", "captions", "text", "sentence", "sentences", "description")
PREFERRED_ID_COLUMNS = ("image_id", "id", "filename", "file_name")


@dataclass(slots=True)
class ImageRecord:
    """Rappresenta una singola immagine indicizzata con la relativa descrizione."""

    item_id: str
    split: str
    caption: str
    image: PILImage.Image


@dataclass(slots=True)
class SearchResult:
    """Contiene il risultato di una ricerca testuale sull'indice visuale."""

    record: ImageRecord
    score: float
    rank: int


@dataclass(slots=True)
class ClipIndex:
    """Indice CLIP in memoria con embedding normalizzati e metadati essenziali."""

    records: list[ImageRecord]
    image_embeddings: torch.Tensor
    metadata: dict[str, Any]


class ClipRetrievalSystem:
    """Sistema di retrieval che confronta embedding testuali e visuali CLIP."""

    def __init__(
        self,
        index: ClipIndex,
        model: CLIPModel,
        processor: CLIPProcessor,
        device: torch.device,
    ) -> None:
        """Inizializza il motore di ricerca a partire da indice, modello e processor."""
        self.index = index
        self.model = model
        self.processor = processor
        self.device = device
        self.model.eval()

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Restituisce le immagini piu' simili semanticamente alla query testuale."""
        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return []

        text_embedding = encode_text_embedding(
            text=cleaned_query,
            processor=self.processor,
            model=self.model,
            device=self.device,
        )
        # Prodotto scalare tra embedding gia' normalizzati L2: equivale alla cosine similarity.
        similarities = text_embedding @ self.index.image_embeddings.T
        k = min(top_k, len(self.index.records))
        scores, indices = torch.topk(similarities, k=k)

        results = []
        for rank, (score, record_index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
            results.append(
                SearchResult(
                    record=self.index.records[record_index],
                    score=float(score),
                    rank=rank,
                )
            )
        return results


def resolve_split(dataset_name: str, requested_split: str = "auto") -> str:
    """Seleziona uno split disponibile privilegiando `train` quando richiesto in automatico.

    Note:
        Richiamata in 01_flickr8k_eda (Esercizio 3).
    """
    splits = get_dataset_split_names(dataset_name)
    if requested_split != "auto":
        if requested_split not in splits:
            raise ValueError(f"Split '{requested_split}' non disponibile. Split trovati: {splits}")
        return requested_split

    if "train" in splits:
        return "train"
    if not splits:
        raise ValueError(f"Nessuno split trovato per il dataset {dataset_name}.")
    return splits[0]


def load_image_dataset_subset(
    dataset_name: str,
    split: str = "auto",
    max_images: int = 1000,
    seed: int = 42,
) -> tuple[Dataset, str]:
    """Carica un subset deterministico del dataset immagini indicato."""
    selected_split = resolve_split(dataset_name, requested_split=split)
    dataset = load_dataset(dataset_name, split=selected_split)
    sample_size = min(max_images, len(dataset))
    subset = dataset.shuffle(seed=seed).select(range(sample_size))
    return subset, selected_split


def detect_image_column(features: Features) -> str:
    """Individua la colonna immagine usando prima i nomi attesi e poi il tipo feature.

    Note:
        Richiamata in 01_flickr8k_eda (Esercizio 3).
    """
    for column in PREFERRED_IMAGE_COLUMNS:
        if column in features:
            return column

    for column, feature in features.items():
        if isinstance(feature, DatasetImage):
            return column

    raise ValueError(f"Colonna immagine non trovata. Colonne disponibili: {list(features)}")


def detect_caption_column(features: Features) -> str:
    """Individua la colonna testuale con caption o descrizioni.

    Note:
        Richiamata in 01_flickr8k_eda (Esercizio 3).
    """
    for column in PREFERRED_CAPTION_COLUMNS:
        if column in features:
            return column

    for column, feature in features.items():
        if isinstance(feature, Value) and feature.dtype == "string":
            return column
        if isinstance(feature, Sequence):
            inner_feature = getattr(feature, "feature", None)
            if isinstance(inner_feature, Value) and inner_feature.dtype == "string":
                return column

    raise ValueError(f"Colonna caption non trovata. Colonne disponibili: {list(features)}")


def detect_id_column(features: Features) -> str | None:
    """Trova una colonna identificativa se il dataset ne espone una.

    Note:
        Richiamata in 01_flickr8k_eda (Esercizio 3).
    """
    for column in PREFERRED_ID_COLUMNS:
        if column in features:
            return column
    return None


def extract_caption(value: Any) -> str:
    """Converte il valore della colonna caption in una stringa leggibile."""
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                nested = extract_caption(item)
                if nested:
                    return nested
        return ""

    if isinstance(value, dict):
        for nested_value in value.values():
            nested = extract_caption(nested_value)
            if nested:
                return nested

    return "" if value is None else str(value)


def normalize_image(image: Any) -> PILImage.Image:
    """Converte un'immagine del dataset in PIL RGB, formato atteso da CLIPProcessor."""
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    raise TypeError(f"Formato immagine non supportato: {type(image)!r}")


def build_image_records(
    dataset: Dataset,
    split: str,
    image_column: str | None = None,
    caption_column: str | None = None,
    id_column: str | None = None,
) -> list[ImageRecord]:
    """Trasforma il dataset Hugging Face in record immagini pronti per l'indicizzazione.

    Note:
        Richiamata in 01_flickr8k_eda (Esercizio 3).
    """
    selected_image_column = image_column or detect_image_column(dataset.features)
    selected_caption_column = caption_column or detect_caption_column(dataset.features)
    selected_id_column = id_column or detect_id_column(dataset.features)

    records = []
    for index, row in enumerate(tqdm(dataset, desc="Preparing image records")):
        item_id = str(row[selected_id_column]) if selected_id_column is not None else f"{split}-{index}"
        records.append(
            ImageRecord(
                item_id=item_id,
                split=split,
                caption=extract_caption(row[selected_caption_column]),
                image=normalize_image(row[selected_image_column]),
            )
        )
    return records


def load_clip_components(model_name: str, device: torch.device) -> tuple[CLIPModel, CLIPProcessor]:
    """Carica modello CLIP e processor sul device richiesto."""
    processor = CLIPProcessor.from_pretrained(model_name)
    try:
        model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    except OSError:
        # Il checkpoint ufficiale OpenAI puo' essere disponibile solo come .bin.
        # In quel caso Transformers recenti richiedono questa scelta esplicita con torch<2.6.
        model = CLIPModel.from_pretrained(model_name, weights_only=False).to(device)
    model.eval()
    return model, processor


def compute_image_embeddings(
    records: list[ImageRecord],
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Calcola embedding visuali normalizzati per una lista di record immagine."""
    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(records), batch_size), desc="Encoding images"):
            batch_records = records[start : start + batch_size]
            inputs = processor(
                images=[record.image for record in batch_records],
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            batch_embeddings = extract_projected_features(model.get_image_features(**inputs))
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def encode_text_embedding(
    text: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: torch.device,
) -> torch.Tensor:
    """Codifica una query testuale in uno spazio compatibile con gli embedding immagine."""
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embedding = extract_projected_features(model.get_text_features(**inputs))
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.cpu()[0]


def extract_projected_features(output: Any) -> torch.Tensor:
    """Estrae il tensore proiettato da versioni diverse delle API CLIP."""
    if isinstance(output, torch.Tensor):
        return output

    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    if isinstance(output, tuple):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2:
                return item

    raise TypeError(f"Output CLIP non supportato per l'estrazione feature: {type(output)!r}")


def make_index_metadata(
    dataset_name: str,
    split: str,
    model_name: str,
    max_images: int,
    seed: int,
    records: list[ImageRecord],
) -> dict[str, Any]:
    """Crea i metadati minimi per validare una cache di embedding."""
    return {
        "dataset_name": dataset_name,
        "split": split,
        "model_name": model_name,
        "max_images": max_images,
        "seed": seed,
        "num_records": len(records),
        "item_ids": [record.item_id for record in records],
    }


def load_cached_embeddings(cache_path: Path, expected_metadata: dict[str, Any]) -> torch.Tensor | None:
    """Carica embedding da cache solo se i metadati coincidono con la configurazione attuale."""
    if not cache_path.exists():
        return None

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Versioni di torch precedenti al parametro weights_only sollevano TypeError.
        payload = torch.load(cache_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    comparable_keys = ("dataset_name", "split", "model_name", "max_images", "seed", "num_records", "item_ids")
    if any(metadata.get(key) != expected_metadata.get(key) for key in comparable_keys):
        return None

    embeddings = payload.get("image_embeddings")
    if (
        not isinstance(embeddings, torch.Tensor)
        or embeddings.ndim != 2
        or embeddings.shape[0] != expected_metadata["num_records"]
    ):
        return None
    return embeddings


def save_embeddings_cache(cache_path: Path, embeddings: torch.Tensor, metadata: dict[str, Any]) -> None:
    """Salva embedding e metadati in una cache locale ignorata da Git."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_embeddings": embeddings, "metadata": metadata}, cache_path)


def build_clip_index(
    dataset_name: str,
    split: str,
    model_name: str,
    max_images: int,
    seed: int,
    cache_path: Path,
    device: torch.device,
    image_batch_size: int = 32,
    force_rebuild: bool = False,
) -> tuple[ClipIndex, CLIPModel, CLIPProcessor]:
    """Costruisce o ricarica un indice CLIP completo per il retrieval."""
    dataset, selected_split = load_image_dataset_subset(
        dataset_name=dataset_name,
        split=split,
        max_images=max_images,
        seed=seed,
    )
    records = build_image_records(dataset=dataset, split=selected_split)
    if not records:
        raise ValueError("Il subset selezionato non contiene immagini da indicizzare.")
    metadata = make_index_metadata(
        dataset_name=dataset_name,
        split=selected_split,
        model_name=model_name,
        max_images=max_images,
        seed=seed,
        records=records,
    )

    model, processor = load_clip_components(model_name=model_name, device=device)
    embeddings = None if force_rebuild else load_cached_embeddings(cache_path, expected_metadata=metadata)

    if embeddings is None:
        embeddings = compute_image_embeddings(
            records=records,
            processor=processor,
            model=model,
            device=device,
            batch_size=image_batch_size,
        )
        save_embeddings_cache(cache_path=cache_path, embeddings=embeddings, metadata=metadata)

    return ClipIndex(records=records, image_embeddings=embeddings, metadata=metadata), model, processor


def build_clip_retrieval_system(
    dataset_name: str,
    split: str,
    model_name: str,
    max_images: int,
    seed: int,
    cache_path: Path,
    device: torch.device,
    image_batch_size: int = 32,
    force_rebuild: bool = False,
) -> ClipRetrievalSystem:
    """Costruisce il sistema completo pronto per rispondere a query testuali.

    Note:
        Richiamata in 02_clip_retrieval_demo (Esercizio 3) e in app.py.
    """
    index, model, processor = build_clip_index(
        dataset_name=dataset_name,
        split=split,
        model_name=model_name,
        max_images=max_images,
        seed=seed,
        cache_path=cache_path,
        device=device,
        image_batch_size=image_batch_size,
        force_rebuild=force_rebuild,
    )
    return ClipRetrievalSystem(index=index, model=model, processor=processor, device=device)
