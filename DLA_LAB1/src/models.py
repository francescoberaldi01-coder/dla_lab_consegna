"""
Modulo per la definizione e l'inizializzazione dei modelli.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

def get_resnet18_feature_extractor(device=None):
    """
    Inizializza una ResNet18 preaddestrata su ImageNet e ne rimuove 
    il layer finale (fully connected) per utilizzarla come estrattore di feature.
    
    Args:
        device (torch.device): Device su cui caricare il modello (cpu o cuda).

    Returns:
        torch.nn.Module: Il modello ResNet18 modificato per restituire feature (512 dimensioni).

    Note:
        Richiamata in 03_baseline_features_svm (Esercizio 2).
    """
    # Carica il modello preaddestrato
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Rimpiazza l'ultimo strato (fc) con un'identità per estrarre le feature (512 dimensioni)
    model.fc = nn.Identity()
    
    # Sposta il modello sul device corretto se specificato
    if device is not None:
        model = model.to(device)
        
    return model

def get_resnet18_classifier(num_classes=43, pretrained=True, device=None):
    """
    Inizializza una ResNet18 preaddestrata su ImageNet e ne sostituisce
    il layer finale (fully connected) con un classificatore lineare per num_classes.
    
    Args:
        num_classes (int): Il numero di classi di output (default 43 per GTSRB).
        pretrained (bool): Se True, carica i pesi preaddestrati su ImageNet.
        device (torch.device): Device su cui caricare il modello (cpu o cuda).
        
    Returns:
        torch.nn.Module: Il modello ResNet18 modificato per il fine-tuning.
    """
    # Carica il modello
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    
    # Sostituiamo l'ultimo strato (fc) con un classificatore lineare con num_classes output
    model.fc = nn.Linear(512, num_classes)
    
    # Sposta il modello sul device corretto se specificato
    if device is not None:
        model = model.to(device)
        
    return model

def get_resnet50_feature_extractor(device=None):
    """
    Inizializza una ResNet50 preaddestrata su ImageNet e ne rimuove 
    il layer finale (fully connected) per utilizzarla come estrattore di feature.
    
    Args:
        device (torch.device): Device su cui caricare il modello (cpu o cuda).
        
    Returns:
        torch.nn.Module: Il modello ResNet50 modificato per restituire feature (2048 dimensioni).
    """
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    if device is not None:
        model = model.to(device)
    return model

def get_resnet50_classifier(num_classes=43, pretrained=True, device=None):
    """
    Inizializza una ResNet50 preaddestrata su ImageNet e ne sostituisce
    il layer finale (fully connected) con un classificatore lineare per num_classes.
    
    Args:
        num_classes (int): Il numero di classi di output (default 43 per GTSRB).
        pretrained (bool): Se True, carica i pesi preaddestrati su ImageNet.
        device (torch.device): Device su cui caricare il modello (cpu o cuda).
        
    Returns:
        torch.nn.Module: Il modello ResNet50 modificato per il fine-tuning.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(2048, num_classes)
    if device is not None:
        model = model.to(device)
    return model

def _build_gtsrb_resnet50_fpn(checkpoint_path, fpn_state_dict):
    """Costruisce un backbone ResNet-50 FPN dai pesi fine-tunati su GTSRB."""
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.ops import FrozenBatchNorm2d
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint backbone non trovato: {checkpoint_path}")

    # Il checkpoint Lightning e' prodotto localmente dal notebook 02 e contiene
    # lo state_dict della ResNet-50 sotto il prefisso "model.".
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    source_state = {
        key.removeprefix("model."): value
        for key, value in source_state.items()
        if key.startswith("model.") and not key.startswith("model.fc.")
    }

    # I batch del detector sono piccoli: FrozenBatchNorm mantiene le statistiche
    # apprese durante il fine-tuning di classificazione senza aggiornarle.
    resnet = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    target_state = resnet.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    }
    missing_keys, _ = resnet.load_state_dict(compatible_state, strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    if set(missing_keys) != expected_missing:
        raise RuntimeError(
            "Il checkpoint GTSRB non copre completamente la ResNet-50. "
            f"Pesi mancanti: {missing_keys}"
        )

    backbone = BackboneWithFPN(
        resnet,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )
    # La FPN mantiene l'inizializzazione COCO, mentre il corpo ResNet proviene
    # esclusivamente dal checkpoint fine-tunato su GTSRB.
    backbone.fpn.load_state_dict(fpn_state_dict)
    return backbone


def get_faster_rcnn_resnet50(
    num_classes=44,
    pretrained=True,
    device=None,
    min_size=256,
    max_size=256,
    backbone_checkpoint_path=None,
    backbone_image_mean=None,
    backbone_image_std=None,
):
    """
    Inizializza una Faster R-CNN con backbone ResNet-50 FPN.
    Sostituisce il classificatore finale con uno adatto al numero di classi desiderato.
    
    Args:
        num_classes (int): Numero totale di classi incluso il background (default 44 per GTSRB).
        pretrained (bool): Se True, carica i pesi preaddestrati su COCO.
        device (torch.device): Device su cui spostare il modello (cpu o cuda).
        min_size (int): Dimensione minima a cui riscalare le immagini internamente.
        max_size (int): Dimensione massima a cui riscalare le immagini internamente.
        backbone_checkpoint_path (str | pathlib.Path): Checkpoint della ResNet-50
            fine-tunata su GTSRB da trasferire nel backbone del detector.
        backbone_image_mean (list[float]): Media usata per normalizzare le immagini
            durante il fine-tuning del backbone GTSRB.
        backbone_image_std (list[float]): Deviazione standard usata per normalizzare
            le immagini durante il fine-tuning del backbone GTSRB.
        
    Returns:
        torch.nn.Module: Il modello Faster R-CNN configurato.
    """
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    # Carica il modello preaddestrato o casuale
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size
    )

    if (backbone_image_mean is None) != (backbone_image_std is None):
        raise ValueError("Media e deviazione standard devono essere specificate insieme.")
    if backbone_checkpoint_path is not None and backbone_image_mean is None:
        raise ValueError("La normalizzazione del backbone GTSRB deve essere specificata.")

    if backbone_checkpoint_path is not None:
        model.backbone = _build_gtsrb_resnet50_fpn(
            backbone_checkpoint_path,
            model.backbone.fpn.state_dict(),
        )

    # Questi attributi non fanno parte dello state_dict: vengono salvati negli
    # iperparametri Lightning per ricostruire correttamente il detector in inferenza.
    if backbone_image_mean is not None:
        model.transform.image_mean = list(backbone_image_mean)
        model.transform.image_std = list(backbone_image_std)
    
    # Rileviamo il numero di feature in ingresso al predittore di box esistente
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Sostituiamo il box predictor con uno nuovo tarato sul nostro numero di classi
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Sposta il modello sul device corretto se specificato
    if device is not None:
        model = model.to(device)
        
    return model
