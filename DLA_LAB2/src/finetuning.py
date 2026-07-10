"""Funzioni riutilizzabili per il fine-tuning di DistilBERT con HuggingFace Trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert/distilbert-base-uncased"


def load_tokenizer(model_name: str = MODEL_NAME):
    """Carica il tokenizer associato al modello scelto.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    return AutoTokenizer.from_pretrained(model_name)


def load_sequence_classifier(
    model_name: str = MODEL_NAME,
    num_labels: int = 2,
    label_names: dict[int, str] | None = None,
):
    """Carica DistilBERT con testa di classificazione per sentiment analysis.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    id2label = label_names or {0: "negative", 1: "positive"}
    label2id = {label: idx for idx, label in id2label.items()}

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def tokenize_dataset_dict(
    dataset_dict: DatasetDict,
    tokenizer,
    text_column: str = "text",
    max_length: int = 128,
) -> DatasetDict:
    """Tokenizza tutti gli split preservando label e metadati utili.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """

    def preprocess_function(examples):
        """Applica tokenizzazione batched alle frasi dello split corrente."""
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )

    # Dataset.map non e' in-place: restituisce sempre una nuova DatasetDict.
    return dataset_dict.map(preprocess_function, batched=True)


def remove_text_column(dataset):
    """Rimuove il testo libero dagli split passati al modello.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    if "text" in dataset.column_names:
        return dataset.remove_columns(["text"])
    return dataset


def compute_classification_metrics_from_predictions(y_true, y_pred) -> dict[str, float]:
    """Calcola metriche sklearn a partire da label reali e predizioni."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Funzione di valutazione richiesta da `Trainer`: converte i logits in metriche sklearn."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return compute_classification_metrics_from_predictions(labels, predictions)


def build_training_arguments(
    output_dir: str | Path,
    learning_rate: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    num_train_epochs: int = 5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
) -> TrainingArguments:
    """Configura `TrainingArguments` con selezione del checkpoint migliore su F1 macro.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=seed,
    )


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    training_args: TrainingArguments,
) -> Trainer:
    """Istanzia un `Trainer` con collator a padding dinamico e metriche sklearn.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )


def evaluate_predictions(trainer: Trainer, dataset) -> dict:
    """Esegue `trainer.predict` e restituisce metriche, label vere e predizioni.

    Note:
        Richiamata in 01_distilbert_finetuning (Esercizio 2).
    """
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    metrics = compute_classification_metrics_from_predictions(labels, predictions)

    return {
        "loss": float(output.metrics.get("test_loss", np.nan)),
        "metrics": metrics,
        "labels": labels,
        "predictions": predictions,
    }
