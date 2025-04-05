import torch
import numpy as np
import math

class TileMerger:
    """
    Classe pour fusionner des tuiles prédites en une image complète
    """
    def __init__(self, height, width, n_classes, step_size, tile_size):
        """
        Initialise le fusionneur de tuiles
        
        Args:
            height (int): Hauteur de l'image cible
            width (int): Largeur de l'image cible
            n_classes (int): Nombre de classes
            step_size (int): Taille du pas entre les tuiles
            tile_size (int): Taille des tuiles
        """
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.step_size = step_size
        self.tile_size = tile_size
        
        # Initialiser les accumulateurs
        self.result = np.zeros((n_classes, height, width), dtype=np.float32)
        self.counts = np.zeros((height, width), dtype=np.int32)
    
    def add_tile(self, prediction, x, y):
        """
        Ajoute une prédiction de tuile à l'accumulateur
        
        Args:
            prediction (torch.Tensor): Prédiction du modèle (B, C, H, W)
            x (int): Position x de la tuile dans l'image
            y (int): Position y de la tuile dans l'image
        """
        # Convertir la prédiction en probabilités
        prediction = torch.softmax(prediction, dim=1)
        prediction = prediction.squeeze().cpu().numpy()
        
        # Calculer les limites effectives de la tuile
        y2 = min(y + self.tile_size, self.height)
        x2 = min(x + self.tile_size, self.width)
        tile_h = y2 - y
        tile_w = x2 - x
        
        # Ajouter la prédiction à l'accumulateur
        self.result[:, y:y2, x:x2] += prediction[:, :tile_h, :tile_w]
        self.counts[y:y2, x:x2] += 1
    
    def merge(self):
        """
        Fusionne toutes les prédictions de tuiles
        
        Returns:
            np.ndarray: Prédiction fusionnée (C, H, W)
        """
        # Éviter la division par zéro
        self.counts = np.maximum(self.counts, 1)
        
        # Normaliser par le nombre de prédictions par pixel
        for c in range(self.n_classes):
            self.result[c, :, :] /= self.counts
        
        return self.result 