import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    """
    Classe implémentant le modèle UNet++ pour la segmentation multiclasses
    avec une branche additionnelle pour la classification
    """
    def __init__(self, n_classes=6, encoder='resnet34', pretrained=True, with_classification=False):
        """
        Initialise le modèle UNet++
        
        Args:
            n_classes (int): Nombre de classes pour la segmentation
            encoder (str): Encodeur à utiliser (resnet34, efficientnet-b0, etc.)
            pretrained (bool): Utiliser un encodeur préentraîné sur ImageNet
            with_classification (bool): Ajouter une branche de classification
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.with_classification = with_classification
        
        # Initialiser le modèle UNet++ avec l'encodeur spécifié
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=n_classes,
            activation=None  # Pas d'activation ici car on l'applique après
        )
        
        # Si on utilise la classification, ajouter une branche classificateur
        if with_classification:
            # Récupérer les caractéristiques de l'encodeur
            if 'resnet' in encoder:
                self.classification_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(512 if '34' in encoder else 2048, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_classes)
                )
            elif 'efficientnet' in encoder:
                # Pour EfficientNet, la taille dépend de la version (b0, b1, etc.)
                self.classification_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(1280, 256),  # 1280 pour efficientnet-b0
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_classes)
                )
            else:
                # Configuration par défaut
                self.classification_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, n_classes)
                )
    
    def forward(self, x):
        """
        Passe avant du modèle
        
        Args:
            x (torch.Tensor): Batch d'images [B, 3, H, W]
            
        Returns:
            tuple ou torch.Tensor:
                - Si with_classification=True: (segmentation_logits, classification_logits)
                - Sinon: segmentation_logits
        """
        # Extraction des features avec l'encodeur
        features = self.model.encoder(x)
        
        # Segmentation avec le UNet++ complet
        segmentation_logits = self.model.decoder(*features)
        segmentation_logits = self.model.segmentation_head(segmentation_logits)
        
        if self.with_classification:
            # Utiliser les features les plus profondes pour la classification
            classification_logits = self.classification_head(features[-1])
            return segmentation_logits, classification_logits
        else:
            return segmentation_logits
    
    def predict_masks(self, x):
        """
        Prédit les masques de segmentation pour un batch d'images
        
        Args:
            x (torch.Tensor): Batch d'images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Masques de segmentation [B, H, W]
        """
        with torch.no_grad():
            output = self(x)
            if self.with_classification:
                segmentation_logits, _ = output
            else:
                segmentation_logits = output
                
            return torch.argmax(F.softmax(segmentation_logits, dim=1), dim=1)
    
    def predict_class(self, x):
        """
        Prédit la classe pour un batch d'images
        
        Args:
            x (torch.Tensor): Batch d'images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Classes prédites [B]
        """
        if not self.with_classification:
            raise ValueError("Le modèle n'a pas été initialisé avec la classification")
            
        with torch.no_grad():
            _, classification_logits = self(x)
            return torch.argmax(classification_logits, dim=1)
    
    def predict_tiles(self, tiles):
        """
        Prédit les masques de segmentation pour des tuiles d'images
        
        Args:
            tiles (list): Liste de tuiles d'images (torch.Tensor)
            
        Returns:
            list: Liste de masques prédits pour chaque tuile
        """
        results = []
        for tile in tiles:
            # Ajouter une dimension de batch si nécessaire
            if len(tile.shape) == 3:
                tile = tile.unsqueeze(0)
            
            # Prédire le masque
            pred = self.predict_masks(tile)
            results.append(pred)
        
        return results

class CombinedLoss(nn.Module):
    """
    Fonction de perte combinant CrossEntropy et Dice Loss
    """
    def __init__(self, weight_ce=0.5, weight_dice=0.5, class_weights=None):
        """
        Initialise la fonction de perte combinée
        
        Args:
            weight_ce (float): Poids pour la loss CrossEntropy
            weight_dice (float): Poids pour la Dice Loss
            class_weights (torch.Tensor, optional): Poids par classe pour CrossEntropy
        """
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, logits, targets):
        """
        Calcule la perte combinée
        
        Args:
            logits (torch.Tensor): Logits prédits [B, n_classes, H, W]
            targets (torch.Tensor): Masques cibles [B, H, W]
            
        Returns:
            torch.Tensor: Valeur de la perte
        """
        # CrossEntropy Loss
        ce_loss = self.ce_loss(logits, targets)
        
        # Dice Loss
        dice_loss = self.dice_loss(logits, targets)
        
        # Combiner les pertes
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

class MultiTaskLoss(nn.Module):
    """
    Fonction de perte pour l'apprentissage multi-tâches
    (segmentation + classification)
    """
    def __init__(self, segmentation_loss, weight_seg=0.7, weight_cls=0.3):
        """
        Initialise la fonction de perte multi-tâches
        
        Args:
            segmentation_loss (nn.Module): Fonction de perte pour la segmentation
            weight_seg (float): Poids pour la perte de segmentation
            weight_cls (float): Poids pour la perte de classification
        """
        super(MultiTaskLoss, self).__init__()
        self.segmentation_loss = segmentation_loss
        self.classification_loss = nn.CrossEntropyLoss()
        self.weight_seg = weight_seg
        self.weight_cls = weight_cls
    
    def forward(self, pred, target):
        """
        Calcule la perte multi-tâches
        
        Args:
            pred (tuple): Prédictions (segmentation_logits, classification_logits)
            target (tuple): Cibles (segmentation_targets, classification_targets)
            
        Returns:
            torch.Tensor: Valeur de la perte
        """
        seg_pred, cls_pred = pred
        seg_target, cls_target = target
        
        # Calculer la perte pour chaque tâche
        seg_loss = self.segmentation_loss(seg_pred, seg_target)
        cls_loss = self.classification_loss(cls_pred, cls_target)
        
        # Combiner les pertes
        return self.weight_seg * seg_loss + self.weight_cls * cls_loss

class DiceLoss(nn.Module):
    """
    Implémentation de la Dice Loss pour la segmentation
    """
    def __init__(self, smooth=1.0):
        """
        Initialise la Dice Loss
        
        Args:
            smooth (float): Facteur de lissage pour éviter la division par zéro
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Calcule la Dice Loss
        
        Args:
            logits (torch.Tensor): Logits prédits [B, n_classes, H, W]
            targets (torch.Tensor): Masques cibles [B, H, W]
            
        Returns:
            torch.Tensor: Valeur de la Dice Loss
        """
        # Calculer les probabilités avec softmax
        probs = F.softmax(logits, dim=1)
        
        # Récupérer les dimensions
        batch_size, n_classes, h, w = probs.size()
        
        # S'assurer que les valeurs des targets sont dans la plage [0, n_classes-1]
        targets = torch.clamp(targets, 0, n_classes-1)
        
        # Convertir les cibles en one-hot encoding
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculer l'intersection et l'union
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        
        # Calculer le score Dice par classe
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Moyenne sur toutes les classes (sauf le fond si spécifié)
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss 