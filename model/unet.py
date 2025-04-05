import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    """
    Classe implémentant le modèle UNet++ pour la segmentation multiclasses
    """
    def __init__(self, n_classes=6, encoder='resnet34', pretrained=True):
        """
        Initialise le modèle UNet++
        
        Args:
            n_classes (int): Nombre de classes pour la segmentation
            encoder (str): Encodeur à utiliser (resnet34, efficientnet-b0, etc.)
            pretrained (bool): Utiliser un encodeur préentraîné sur ImageNet
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes
        
        # Initialiser le modèle UNet++ avec l'encodeur spécifié
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=n_classes,
            activation=None  # Pas d'activation ici car on l'applique après
        )
    
    def forward(self, x):
        """
        Passe avant du modèle
        
        Args:
            x (torch.Tensor): Batch d'images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Logits de segmentation [B, n_classes, H, W]
        """
        return self.model(x)
    
    def predict_masks(self, x):
        """
        Prédit les masques de segmentation pour un batch d'images
        
        Args:
            x (torch.Tensor): Batch d'images [B, 3, H, W]
            
        Returns:
            torch.Tensor: Masques de segmentation [B, H, W]
        """
        with torch.no_grad():
            logits = self(x)
            return torch.argmax(F.softmax(logits, dim=1), dim=1)
    
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
        
        # Convertir les cibles en one-hot encoding
        batch_size, n_classes, h, w = probs.size()
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