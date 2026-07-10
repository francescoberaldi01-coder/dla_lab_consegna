"""
Modulo contenente le definizioni dei LightningModule per addestrare i modelli
tramite PyTorch Lightning in modo pulito e modulare.
"""

import torch
import pytorch_lightning as pl
import torchmetrics

from .models import get_resnet18_classifier, get_resnet50_classifier

class LitResNet(pl.LightningModule):
    """
    LightningModule che racchiude una ResNet-18 o ResNet-50 per la classificazione
    del dataset GTSRB. Gestisce il forward pass, la CrossEntropy e le metriche di
    accuracy e F1 macro per training, validazione e test.

    Note:
        Istanziata in 04_finetuning_baseline e 05_pipeline_experiments (Esercizio 2);
        in 02_resnet50_pretraining (Esercizio 3) tramite model_name="resnet50".
    """
    def __init__(
        self,
        num_classes=43,
        pretrained=True,
        lr=1e-4,
        class_weights=None,
        optimizer_name="adamw",
        model_name="resnet18",
        label_smoothing=0.0,
        use_onecycle_lr=False,
    ):
        """
        Inizializza il modulo Lightning.
        
        Args:
            num_classes (int): Numero di classi in output (es. 43 per GTSRB).
            pretrained (bool): Se True, carica i pesi preaddestrati su ImageNet.
            lr (float): Learning rate dell'ottimizzatore.
            class_weights (torch.Tensor): Pesi opzionali per bilanciare la loss.
            optimizer_name (str): Nome dell'ottimizzatore ('adam', 'adamw' o 'sgd').
            model_name (str): Nome del modello da usare ('resnet18' o 'resnet50').
            label_smoothing (float): Intensita' del label smoothing nella CrossEntropy.
            use_onecycle_lr (bool): Se True, applica OneCycleLR a ogni batch.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.model_name = model_name
        self.label_smoothing = label_smoothing
        self.use_onecycle_lr = use_onecycle_lr
        
        # Inizializza il modello prendendolo dal modulo src/models.py
        if self.model_name == "resnet50":
            self.model = get_resnet50_classifier(
                num_classes=num_classes, 
                pretrained=pretrained
            )
        else:
            self.model = get_resnet18_classifier(
                num_classes=num_classes, 
                pretrained=pretrained
            )
        
        # Inizializzazione delle metriche
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        
        # La F1 macro assegna lo stesso peso a ogni classe ed evidenzia le
        # prestazioni sulle categorie meno rappresentate.
        self.train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        """
        Propagazione in avanti dei dati attraverso la rete.
        """
        return self.model(x)

    def _compute_loss(self, logits, targets):
        """
        Calcola la CrossEntropy applicando, se presenti, i pesi di classe.
        """
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        return torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        """
        Logica per un singolo step di addestramento.
        """
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.train_acc(logits, y)
        self.train_f1(logits, y)
        
        # Logging delle metriche per tensorboard / wandb
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/f1_score', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Logica per un singolo step di validazione.
        """
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        
        # Logging della val_loss, val_accuracy e val_f1
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1_score', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Alias senza slash: PyTorch Lightning non risolve correttamente i nomi di
        # metrica con "/" all'interno del template filename di ModelCheckpoint.
        self.log('val_f1_score', self.val_f1, on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        """
        Logica per un singolo step di test. (Chiamato solo da trainer.test())
        """
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        self.test_acc(logits, y)
        self.test_f1(logits, y)
        
        # Logging della test_loss, test_accuracy e test_f1
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/accuracy', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/f1_score', self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configura l'ottimizzatore scelto e, se richiesto, OneCycleLR.
        """
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError(f"Ottimizzatore non supportato: {self.optimizer_name}")
        
        if not self.use_onecycle_lr:
            return optimizer

        # OneCycleLR raggiunge un picco di max_lr e poi scende verso zero.
        # interval='step' significa che il LR viene aggiornato a ogni singolo batch, non a fine epoca.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10, # Il picco sarà 10 volte l'LR iniziale (es. 1e-3)
            total_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


class LitFasterRCNN(pl.LightningModule):
    """
    LightningModule per l'addestramento e la validazione di Faster R-CNN.
    Gestisce il calcolo della loss multi-task e la metrica di valutazione mAP.

    Note:
        Istanziata in 04_faster_rcnn_training e 05_evaluation_and_inference (Esercizio 3).
    """
    def __init__(
        self,
        num_classes=44,
        pretrained=True,
        lr=0.005,
        optimizer_name="sgd",
        min_size=256,
        max_size=256,
        backbone_checkpoint_path=None,
        backbone_image_mean=None,
        backbone_image_std=None,
    ):
        """
        Inizializza il modulo.
        
        Args:
            num_classes (int): Numero di classi incluso il background (default 44 per GTSRB).
            pretrained (bool): Se True, carica i pesi preaddestrati.
            lr (float): Learning rate dell'ottimizzatore.
            optimizer_name (str): Nome dell'ottimizzatore ('sgd' o 'adamw').
            min_size (int): Dimensione minima a cui riscalare le immagini internamente.
            max_size (int): Dimensione massima a cui riscalare le immagini internamente.
            backbone_checkpoint_path (str): Checkpoint ResNet-50 fine-tunato su GTSRB.
            backbone_image_mean (list[float]): Media di normalizzazione del backbone GTSRB.
            backbone_image_std (list[float]): Deviazione standard del backbone GTSRB.
        """
        super().__init__()
        # Il percorso viene salvato per ricostruire esplicitamente la stessa
        # architettura GTSRB-FPN prima di ripristinare lo state_dict del detector.
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer_name = optimizer_name
        
        # Importiamo la factory per il modello
        from .models import get_faster_rcnn_resnet50
        self.model = get_faster_rcnn_resnet50(
            num_classes=num_classes, 
            pretrained=pretrained,
            min_size=min_size,
            max_size=max_size,
            backbone_checkpoint_path=backbone_checkpoint_path,
            backbone_image_mean=backbone_image_mean,
            backbone_image_std=backbone_image_std,
        )
        
        # Inizializziamo la metrica di validazione mAP usando faster_coco_eval per compatibilita' Windows
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        self.val_map = MeanAveragePrecision(backend="faster_coco_eval")

    def forward(self, images, targets=None):
        """
        Passaggio in avanti. In modalità train richiede targets e restituisce le loss.
        In modalità eval, targets è opzionale e restituisce le predizioni.
        """
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        # Faster R-CNN in train mode calcola e restituisce il dizionario delle loss
        loss_dict = self(images, targets)
        loss = sum(l for l in loss_dict.values())
        
        # Loggiamo la loss totale e le singole componenti
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True, batch_size=len(images))
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(images)
        targets = [{k: v for k, v in t.items()} for t in targets]

        # La validazione usa esclusivamente la modalita' di inferenza. Evitare il
        # passaggio train/eval nello stesso batch rende la mAP riproducibile dopo
        # il caricamento del checkpoint.
        self.model.eval()
        with torch.no_grad():
            preds = self(images)
        self.val_map.update(preds, targets)

    def on_validation_epoch_end(self):
        # Calcoliamo le metriche finali mAP sull'intera epoca di validazione
        map_metrics = self.val_map.compute()
        
        self.log("val/mAP", map_metrics["map"], prog_bar=True)
        self.log("val/mAP_50", map_metrics["map_50"], prog_bar=True)

        # Alias senza slash: PyTorch Lightning non risolve correttamente i nomi di
        # metrica con "/" all'interno del template filename di ModelCheckpoint.
        self.log("val_mAP_50", map_metrics["map_50"], prog_bar=False)

        # Resettiamo la metrica per l'epoca successiva
        self.val_map.reset()

    def configure_optimizers(self):
        """
        Configura l'ottimizzatore e il learning rate scheduler.
        """
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        elif optimizer_name == "sgd":
            # SGD con momentum e weight decay è una scelta comune per Faster R-CNN.
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        else:
            raise ValueError(f"Ottimizzatore non supportato: {self.optimizer_name}")
            
        # Scheduler StepLR per dimezzare o scalare il LR nel corso delle epoche
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
