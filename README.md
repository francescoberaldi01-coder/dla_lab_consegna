# Deep Learning Applications — Francesco Beraldi

Repository dei laboratori del corso di Deep Learning Applications. Ogni
laboratorio contiene notebook eseguiti con output salvati, configurazioni
riproducibili, moduli Python riutilizzabili e un README dedicato.

## Laboratori

### [DLA Lab 1](DLA_LAB1/README.md)

Computer vision con PyTorch e PyTorch Lightning:

- classificazione delle immagini GTSRB;
- confronto tra baseline e fine-tuning;
- object detection con Faster R-CNN;
- valutazione quantitativa e inferenza qualitativa.

### [DLA Lab 2](DLA_LAB2/README.md)

Transformer e modelli multimodali:

- analisi del dataset Rotten Tomatoes e del tokenizer DistilBERT;
- baseline su feature `[CLS]` con `LinearSVC`;
- fine-tuning di DistilBERT tramite Hugging Face `Trainer`;
- retrieval testo-immagine su Flickr8k con CLIP e interfaccia Gradio.

### [DLA Lab 3](DLA_LAB3/README.md)

Deep Reinforcement Learning:

- REINFORCE su CartPole;
- confronto tra standardizzazione dei ritorni e baseline appresa `V(s)`;
- Advantage Actor-Critic su CartPole e LunarLander;
- Deep Q-Learning con replay buffer e target network;
- confronto finale tra gli algoritmi.

Le cartelle `professor/` conservano esclusivamente i notebook originali delle
consegne, separati dalle soluzioni.

## Ambiente

Sono disponibili due file di dipendenze nella root:

- `requirements-windows.txt`: Windows con PyTorch CUDA 12.1;
- `requirements-mac.txt`: macOS con PyTorch standard e supporto MPS quando
  disponibile.

Gli ambienti locali non sono versionati.

### Windows

```text
uv venv .venv-windows --python 3.11
uv pip install --python .venv-windows\Scripts\python.exe -r requirements-windows.txt
```

Verifica di PyTorch e CUDA:

```text
.venv-windows\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### macOS

```text
uv venv .venv-mac --python 3.11
uv pip install --python .venv-mac/bin/python -r requirements-mac.txt
```

Verifica di PyTorch e MPS:

```text
.venv-mac/bin/python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

## Esecuzione

1. creare l'ambiente adatto al proprio sistema operativo;
2. aprire la repository dalla cartella principale;
3. selezionare il kernel dell'ambiente locale;
4. seguire l'ordine indicato nel README del laboratorio interessato.

I notebook individuano automaticamente la root della repository e importano i
moduli del relativo package `src`. I dataset e gli artefatti pesanti vengono
scaricati o generati localmente e sono esclusi da Git tramite `.gitignore`.

## Riproducibilità

- gli iperparametri sono raccolti in file YAML;
- i seed sono esplicitati nei notebook o nelle configurazioni;
- gli split di validazione e test sono mantenuti separati;
- i notebook includono gli output delle esecuzioni finali;
- figure e tabelle riportano le metriche usate nelle conclusioni;
- i dettagli specifici di ogni pipeline sono documentati nei README dei lab.

I risultati di Reinforcement Learning possono variare leggermente tra sistemi e
versioni delle librerie anche usando gli stessi seed, a causa della natura
stocastica degli ambienti e degli algoritmi.
