"""Metriche per classificazione binaria e report dei risultati."""

from sklearn.metrics import accuracy_score, classification_report, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcola accuracy e F1 macro.

    Note:
        Richiamata in 02_feature_extraction_baseline (Esercizio 1).
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def build_classification_report(y_true, y_pred, target_names=None) -> str:
    """Restituisce il classification report in formato testuale.

    Note:
        Richiamata in 02_feature_extraction_baseline (Esercizio 1) e
        01_distilbert_finetuning (Esercizio 2).
    """
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
