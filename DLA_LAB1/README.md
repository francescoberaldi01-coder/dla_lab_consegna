# DLA Lab 1 - Transfer learning su GTSRB

Il laboratorio studia l'uso di modelli preaddestrati per classificazione e object detection di segnali stradali. Sono inclusi una baseline con feature congelate, il fine-tuning, una pipeline sperimentale consolidata e la detection con Faster R-CNN.

## Struttura degli esercizi

| Esercizio | Implementazione |
|---|---|
| Esercizio 1.1 - EDA | `exercise_2_gtsrb_classification/01_eda_gtsrb.ipynb` |
| Esercizio 1.2 - baseline stabile | `exercise_2_gtsrb_classification/03_baseline_features_svm.ipynb` |
| Esercizio 1.3 - fine-tuning | `exercise_2_gtsrb_classification/04_finetuning_baseline.ipynb` |
| Esercizio 2 - pipeline riproducibile | `exercise_2_gtsrb_classification/05_pipeline_experiments.ipynb` |
| Esercizio 3.1 - miglioramento | confronto tra notebook `04` e `05` della classificazione |
| Esercizio 3.3 - detection | notebook `01`-`05` in `exercise_3_gtsrb_detection/` |

Il notebook `02_preprocessing.ipynb` della classificazione documenta e verifica la preparazione dei dati condivisa dagli esperimenti successivi.

## Ordine di esecuzione

Per la classificazione:

1. `01_eda_gtsrb.ipynb`;
2. `02_preprocessing.ipynb`;
3. `03_baseline_features_svm.ipynb`;
4. `04_finetuning_baseline.ipynb`;
5. `05_pipeline_experiments.ipynb`.

Per la detection i notebook vanno eseguiti da `01` a `05`. In particolare, `02_resnet50_pretraining.ipynb` produce il checkpoint ResNet-50 fine-tunato su GTSRB che viene trasferito nel backbone Faster R-CNN dal notebook `04`.

I risultati locali sono salvati nelle cartelle `logs/`, i checkpoint in `checkpoints/` e le immagini destinate ai README in `assets/`. Il tracking cloud degli esperimenti consolidati usa Weights & Biases.

## Riproducibilità

- seed principale: 42;
- split train/validation della classificazione: 80/20 stratificato con `random_state=42`;
- test set ufficiale mantenuto separato fino alla valutazione finale;
- configurazioni principali in file YAML gestiti con OmegaConf;
- metriche salvate localmente tramite `CSVLogger`;
- checkpoint selezionati sulle metriche di validation, mai sul test set.

Le versioni delle dipendenze principali e le istruzioni per creare gli ambienti Windows e macOS sono riportate nel README principale della repository e nei file `requirements-*.txt`.

La run finale Windows è stata eseguita con Python 3.11, PyTorch 2.5.1, torchvision 0.20.1, PyTorch Lightning 2.6.5, TorchMetrics 1.9.0, datasets 3.6.0, OmegaConf 2.3.1, Weights & Biases 0.28.0 e faster-coco-eval 1.7.2.

## Fonti e attribuzioni

- Houben, S. et al., *Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark*, IJCNN 2013.
- Dataset di classificazione: `torchvision.datasets.GTSRB`.
- Dataset detection: `keremberke/german-traffic-sign-detection` su Hugging Face.
- Modelli preaddestrati: ResNet e Faster R-CNN forniti da `torchvision` con pesi ImageNet/COCO.
- Training e metriche: PyTorch Lightning, TorchMetrics e `faster-coco-eval`.
- Configurazione e tracking: OmegaConf e Weights & Biases.

Le implementazioni esterne sono state adattate tramite moduli locali in `src/`; le modifiche principali riguardano preprocessing, classificatori finali, pipeline Lightning e sostituzione del backbone ResNet-50 nel detector.
