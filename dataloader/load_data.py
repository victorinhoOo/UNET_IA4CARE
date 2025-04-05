import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CytologyDataset(Dataset):
    """
    Dataset pour la segmentation des images de cytoponction thyroïdienne
    """
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, n_classes=6):
        """
        Initialisation du dataset
        
        Args:
            img_dir (str): Chemin vers le dossier des images
            mask_dir (str): Chemin vers le dossier des masques
            transform: Transformations à appliquer aux images
            mask_transform: Transformations à appliquer aux masques
            n_classes (int): Nombre de classes pour la segmentation
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.n_classes = n_classes
        
        # Transformations par défaut
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = mask_transform or transforms.ToTensor()
        
        # Trouver tous les fichiers d'images et de masques correspondants
        self.img_files = []
        self.mask_files = []
        
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(img_dir, img_file)
            mask_file = img_file  # Supposons que les noms de fichiers sont identiques
            mask_path = os.path.join(mask_dir, mask_file)
            
            if os.path.exists(mask_path):
                self.img_files.append(img_path)
                self.mask_files.append(mask_path)
        
        print(f"Chargé {len(self.img_files)} paires d'images/masques")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        # Charger l'image et le masque
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')  # Mode L pour les niveaux de gris
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            
            # Convertir le masque en tensor d'entiers
            mask = np.array(mask)
            
            # Limiter les valeurs du masque au nombre de classes
            mask = np.clip(mask, 0, self.n_classes - 1)
            
            # Convertir en tensor
            mask = torch.from_numpy(mask).long()
            
            return image, mask
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path} ou du masque {mask_path}: {e}")
            # En cas d'erreur, retourner une image noire et un masque vide
            dummy_img = torch.zeros(3, 224, 224)
            dummy_mask = torch.zeros(224, 224, dtype=torch.long)
            return dummy_img, dummy_mask

def get_data_loaders(img_dir, mask_dir, batch_size=8, val_split=0.2, test_split=0.1, 
                     workers=4, seed=42, n_classes=6):
    """
    Crée des DataLoaders pour les données de segmentation
    
    Args:
        img_dir (str): Chemin vers le dossier des images
        mask_dir (str): Chemin vers le dossier des masques
        batch_size (int): Taille des batches
        val_split (float): Proportion des données pour la validation
        test_split (float): Proportion des données pour le test
        workers (int): Nombre de workers pour le chargement des données
        seed (int): Seed pour la reproductibilité
        n_classes (int): Nombre de classes pour la segmentation
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Définir les transformations avec augmentation de données pour l'entraînement
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transformations pour validation et test (sans augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset complet
    full_dataset = CytologyDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        n_classes=n_classes
    )
    
    # Calculer les tailles des ensembles
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - test_size - val_size
    
    # Diviser en train, val, test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Mettre à jour les transformations pour val et test
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 