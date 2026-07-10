# DLA Lab 2 - Transformer e multimodal retrieval

Il laboratorio utilizza l'ecosistema Hugging Face per sentiment analysis con DistilBERT e retrieval text-to-image con CLIP.

## Struttura

- `exercise_1_feature_baseline/`: EDA di Rotten Tomatoes, analisi del tokenizer e baseline con DistilBERT congelato più `LinearSVC`;
- `exercise_2_finetuning/`: fine-tuning end-to-end di DistilBERT tramite Hugging Face `Trainer`;
- `exercise_3_3_clip_retrieval/`: indicizzazione CLIP di Flickr8k, verifica qualitativa del ranking e applicazione Gradio;
- `src/`: caricamento dati, feature extraction, metriche, fine-tuning e retrieval riutilizzabili;
- `professor/`: materiale originale del laboratorio, mantenuto separato dalla soluzione.

## Ordine di esecuzione

1. `exercise_1_feature_baseline/01_eda_dataset_and_tokenizer.ipynb`;
2. `exercise_1_feature_baseline/02_feature_extraction_baseline.ipynb`;
3. `exercise_2_finetuning/01_distilbert_finetuning.ipynb`;
4. `exercise_3_3_clip_retrieval/01_flickr8k_eda.ipynb`;
5. `exercise_3_3_clip_retrieval/02_clip_retrieval_demo.ipynb`.

L'applicazione si avvia dalla root della repository con:

```text
.venv-windows\Scripts\python.exe DLA_LAB2\exercise_3_3_clip_retrieval\app.py
```

Su macOS va usato l'eseguibile equivalente in `.venv-mac/bin/python`.

## Risultati sentiment analysis

| Modello | Validation accuracy | Validation F1 macro | Test accuracy | Test F1 macro |
|---|---:|---:|---:|---:|
| DistilBERT congelato + LinearSVC | 0,8218 | 0,8217 | 0,7983 | 0,7983 |
| DistilBERT fine-tuned | 0,8490 | 0,8490 | 0,8424 | 0,8424 |

Il fine-tuning migliora il test set di circa `+0,044` in accuracy e F1 macro. Il validation set viene usato per selezionare il checkpoint, mentre il test set resta separato fino alla valutazione finale.

## Retrieval CLIP

Il sistema indicizza un subset deterministico di 1000 immagini Flickr8k con seed 42. Gli embedding immagine e testo sono normalizzati L2 e confrontati tramite prodotto scalare, equivalente alla cosine similarity. L'interfaccia restituisce fino a 10 risultati con score e caption; la cache locale viene riutilizzata solo quando dataset, split, modello, seed e identificativi delle immagini coincidono.

## Riproducibilità

- dataset sentiment: `cornell-movie-review-data/rotten_tomatoes` con split ufficiali;
- modello NLP: `distilbert/distilbert-base-uncased`;
- modello multimodale: `openai/clip-vit-base-patch32`;
- configurazioni in `config.yaml` per fine-tuning e retrieval;
- seed principale: 42;
- padding dinamico tramite `DataCollatorWithPadding`;
- checkpoint migliore del fine-tuning selezionato su F1 macro di validation;
- output pesanti e cache salvati nelle cartelle `outputs/`, escluse dal versionamento.

La run finale Windows usa Python 3.11, PyTorch 2.5.1, Transformers 5.13.0, Datasets 3.6.0, Accelerate 1.14.0, scikit-learn 1.9.0, OmegaConf 2.3.1 e Gradio 6.20.0.

## Fonti e attribuzioni

- dataset Rotten Tomatoes: `cornell-movie-review-data/rotten_tomatoes` su Hugging Face;
- dataset Flickr8k: `jxie/flickr8k` su Hugging Face;
- modelli DistilBERT e CLIP distribuiti tramite Hugging Face Transformers;
- training: Hugging Face `Trainer` e `TrainingArguments`;
- interfaccia: Gradio;
- metriche: scikit-learn.
