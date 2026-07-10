"""Funzioni per caricare e ispezionare il dataset Rotten Tomatoes."""

from datasets import DatasetDict, get_dataset_split_names, load_dataset


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
LABEL_NAMES = {0: "negative", 1: "positive"}


def get_split_names(dataset_name: str = DATASET_NAME) -> list[str]:
    """Restituisce gli split disponibili per il dataset indicato."""
    return get_dataset_split_names(dataset_name)


def load_rotten_tomatoes(dataset_name: str = DATASET_NAME) -> DatasetDict:
    """Carica tutti gli split del dataset Rotten Tomatoes in un DatasetDict.

    Note:
        Richiamata in 01_eda_dataset_and_tokenizer e 02_feature_extraction_baseline
        (Esercizio 1); in 01_distilbert_finetuning (Esercizio 2).
    """
    splits = get_split_names(dataset_name)
    return DatasetDict({split: load_dataset(dataset_name, split=split) for split in splits})


def make_debug_subset(dataset_dict: DatasetDict, num_examples: int = 256) -> DatasetDict:
    """Crea subset piccoli per verificare rapidamente la pipeline."""
    subsets = {}
    for split, dataset in dataset_dict.items():
        size = min(num_examples, len(dataset))
        subsets[split] = dataset.select(range(size))
    return DatasetDict(subsets)
