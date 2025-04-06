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
        
        # Créer un dictionnaire des masques disponibles pour une recherche plus rapide
        mask_dict = {}
        for mask_file in os.listdir(mask_dir):
            if mask_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not mask_file.startswith('._'):
                mask_dict[mask_file] = os.path.join(mask_dir, mask_file)
        
        for img_file in os.listdir(img_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')) or img_file.startswith('._'):
                continue
                
            img_path = os.path.join(img_dir, img_file)
            
            # Convertir le nom d'image avec préfixe ann en nom de masque avec préfixe mask
            # Format: X.svs_ann_Y.png -> X.svs_mask_Y.png
            if "_ann_" in img_file:
                mask_file = img_file.replace("_ann_", "_mask_")
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

class CategorizedCytologyDataset(Dataset):
    """
    Dataset pour la classification des images de cytoponction thyroïdienne
    organisées en dossiers par catégorie
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialisation du dataset
        
        Args:
            root_dir (str): Chemin vers le dossier racine contenant les sous-dossiers de catégories
            transform: Transformations à appliquer aux images
        """
        self.root_dir = root_dir
        
        # Transformations par défaut
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Récupérer toutes les catégories (noms des dossiers)
        self.categories = [d for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))]
        
        # Créer un dictionnaire pour mapper les catégories aux indices de classe
        self.class_to_idx = {cat: i for i, cat in enumerate(sorted(self.categories))}
        self.idx_to_class = {i: cat for i, cat in enumerate(sorted(self.categories))}
        
        # Récupérer tous les fichiers images avec leur catégorie
        self.samples = []
        for category in self.categories:
            category_dir = os.path.join(root_dir, category)
            for img_file in os.listdir(category_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_file.startswith('._'):
                    img_path = os.path.join(category_dir, img_file)
                    self.samples.append((img_path, self.class_to_idx[category]))
        
        print(f"Chargé {len(self.samples)} images dans {len(self.categories)} catégories")
        print(f"Catégories: {self.categories}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Charger l'image
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            # En cas d'erreur, retourner une image noire
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, class_idx

class CategorizedMaskDataset(Dataset):
    """
    Dataset pour la segmentation des images de cytoponction thyroïdienne organisées par catégorie
    """
    def __init__(self, img_root_dir, mask_root_dir, transform=None, mask_transform=None):
        """
        Initialisation du dataset
        
        Args:
            img_root_dir (str): Chemin vers le dossier racine des images catégorisées
            mask_root_dir (str): Chemin vers le dossier racine des masques catégorisés
            transform: Transformations à appliquer aux images
            mask_transform: Transformations à appliquer aux masques
        """
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir
        
        # Transformations par défaut
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = mask_transform or transforms.ToTensor()
        
        # Obtenir toutes les catégories (dossiers)
        self.categories = [d for d in os.listdir(img_root_dir) 
                          if os.path.isdir(os.path.join(img_root_dir, d)) and not d.startswith('.')]
        
        # Créer une liste de tuples (chemin_image, chemin_masque, catégorie)
        self.samples = []
        
        for category in self.categories:
            img_category_dir = os.path.join(img_root_dir, category)
            mask_category_dir = os.path.join(mask_root_dir, category)
            
            # Vérifier que le dossier de masques existe
            if not os.path.exists(mask_category_dir):
                print(f"[AVERTISSEMENT] Dossier de masques manquant pour la catégorie '{category}'")
                continue
                
            # Liste tous les fichiers d'image dans cette catégorie
            for img_file in os.listdir(img_category_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')) or img_file.startswith('.'):
                    continue
                    
                img_path = os.path.join(img_category_dir, img_file)
                mask_path = os.path.join(mask_category_dir, img_file)
                
                # Vérifier que le masque existe
                if not os.path.exists(mask_path):
                    print(f"[AVERTISSEMENT] Masque manquant pour l'image '{img_path}'")
                    continue
                    
                self.samples.append((img_path, mask_path, category))
        
        # Lire le fichier de mapping des classes pour déterminer le nombre de classes
        self.n_classes = 0
        class_mapping_path = os.path.join(mask_root_dir, 'class_mapping.txt')
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path, 'r') as f:
                lines = f.readlines()
                # La première ligne est un en-tête, donc on compte les lignes - 1
                self.n_classes = len(lines) - 1
        else:
            # Nombre de catégories + 1 pour le fond (classe 0)
            self.n_classes = len(self.categories) + 1
            print(f"[AVERTISSEMENT] Fichier class_mapping.txt non trouvé. Utilisation de {self.n_classes} classes par défaut.")
        
        print(f"Chargé {len(self.samples)} paires d'images/masques dans {len(self.categories)} catégories")
        print(f"Nombre de classes: {self.n_classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path, category = self.samples[idx]
        
        try:
            # Charger l'image et le masque
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

def get_categorized_data_loaders(root_dir, batch_size=8, val_split=0.2, test_split=0.1, 
                               workers=4, seed=42):
    """
    Crée des DataLoaders pour les données de classification organisées en dossiers par catégorie
    
    Args:
        root_dir (str): Chemin vers le dossier racine contenant les sous-dossiers de catégories
        batch_size (int): Taille des batches
        val_split (float): Proportion des données pour la validation
        test_split (float): Proportion des données pour le test
        workers (int): Nombre de workers pour le chargement des données
        seed (int): Seed pour la reproductibilité
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
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
    full_dataset = CategorizedCytologyDataset(
        root_dir=root_dir,
        transform=train_transform
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
    
    return train_loader, val_loader, test_loader, full_dataset.categories

def get_categorized_mask_loaders(img_root_dir, mask_root_dir, batch_size=8, val_split=0.2, test_split=0.1, 
                               workers=4, seed=42):
    """
    Crée des DataLoaders pour les données de segmentation organisées par catégories
    
    Args:
        img_root_dir (str): Chemin vers le dossier racine des images catégorisées
        mask_root_dir (str): Chemin vers le dossier racine des masques catégorisés
        batch_size (int): Taille des batches
        val_split (float): Proportion des données pour la validation
        test_split (float): Proportion des données pour le test
        workers (int): Nombre de workers pour le chargement des données
        seed (int): Seed pour la reproductibilité
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, n_classes)
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
    full_dataset = CategorizedMaskDataset(
        img_root_dir=img_root_dir,
        mask_root_dir=mask_root_dir,
        transform=train_transform
    )
    
    # Récupérer le nombre de classes
    n_classes = full_dataset.n_classes
    
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
    
    return train_loader, val_loader, test_loader, n_classes 