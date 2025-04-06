import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import glob
from pathlib import Path

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

# Classe pour le dataset de classification
class CategorizedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = [d for d in os.listdir(root_dir) 
                          if os.path.isdir(os.path.join(root_dir, d))]
        self.categories.sort()  # Trier pour assurer la cohérence
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        self.image_paths = []
        self.labels = []
        
        for category in self.categories:
            category_path = os.path.join(root_dir, category)
            image_list = [f for f in os.listdir(category_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_list:
                img_path = os.path.join(category_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.category_to_idx[category])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Classe pour le dataset de segmentation avec masques
class MaskDataset(Dataset):
    def __init__(self, img_root_dir, mask_root_dir, transform=None, mask_transform=None):
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.categories = [d for d in os.listdir(img_root_dir) 
                          if os.path.isdir(os.path.join(img_root_dir, d))]
        self.categories.sort()  # Trier pour assurer la cohérence
        
        self.image_paths = []
        self.mask_paths = []
        
        for category in self.categories:
            img_category_path = os.path.join(img_root_dir, category)
            mask_category_path = os.path.join(mask_root_dir, category)
            
            # Vérifier si le dossier de masques correspondant existe
            if not os.path.exists(mask_category_path):
                print(f"Attention: Le dossier de masques pour la catégorie '{category}' n'existe pas.")
                continue
            
            # Récupérer les images et les masques correspondants
            image_list = [f for f in os.listdir(img_category_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_list:
                img_path = os.path.join(img_category_path, img_name)
                mask_path = os.path.join(mask_category_path, img_name)
                
                # Vérifier si le masque correspondant existe
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Charger en niveaux de gris
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convertir le masque en tensor sans normalisation
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask

# Classe pour le dataset multitask (segmentation + classification)
class MultitaskDataset(Dataset):
    def __init__(self, img_root_dir, mask_root_dir, transform=None, mask_transform=None):
        self.img_root_dir = img_root_dir
        self.mask_root_dir = mask_root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.categories = [d for d in os.listdir(img_root_dir) 
                          if os.path.isdir(os.path.join(img_root_dir, d))]
        self.categories.sort()  # Trier pour assurer la cohérence
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        for category in self.categories:
            img_category_path = os.path.join(img_root_dir, category)
            mask_category_path = os.path.join(mask_root_dir, category)
            
            # Vérifier si le dossier de masques correspondant existe
            if not os.path.exists(mask_category_path):
                print(f"Attention: Le dossier de masques pour la catégorie '{category}' n'existe pas.")
                continue
            
            # Récupérer les images et les masques correspondants
            image_list = [f for f in os.listdir(img_category_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_list:
                img_path = os.path.join(img_category_path, img_name)
                mask_path = os.path.join(mask_category_path, img_name)
                
                # Vérifier si le masque correspondant existe
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(self.category_to_idx[category])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Charger en niveaux de gris
        
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convertir le masque en tensor sans normalisation
            mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask, label

# Définir la classe TransformedMultitaskSubset en dehors de la fonction
class TransformedMultitaskSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        dataset = self.subset.dataset
        orig_idx = self.subset.indices[idx]
        
        img_path = dataset.image_paths[orig_idx]
        mask_path = dataset.mask_paths[orig_idx]
        label = dataset.labels[orig_idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Charger en niveaux de gris
        
        if self.transform:
            image = self.transform(image)
        
        # Convertir le masque en tensor sans normalisation
        mask_array = np.array(mask, dtype=np.int64)
        # Limiter les valeurs à l'intervalle [0, n_classes-1]
        n_classes = len(dataset.categories)
        mask_array = np.clip(mask_array, 0, n_classes-1)
        mask = torch.from_numpy(mask_array)
        
        return image, mask, label

# À ajouter avec les autres classes en dehors des fonctions
class TransformedMaskSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        img_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
        mask_path = self.subset.dataset.mask_paths[self.subset.indices[idx]]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Charger en niveaux de gris
        
        if self.transform:
            image = self.transform(image)
        
        # Convertir le masque en tensor sans normalisation
        mask_array = np.array(mask, dtype=np.int64)
        # Limiter les valeurs à l'intervalle [0, n_classes-1]
        n_classes = len(self.subset.dataset.categories)
        mask_array = np.clip(mask_array, 0, n_classes-1)
        mask = torch.from_numpy(mask_array)
        
        return image, mask

# Fonction pour obtenir les data loaders pour l'apprentissage multitâche
def get_multitask_loaders(img_root_dir, mask_root_dir, batch_size=8, workers=4, seed=42, train_split=0.7, val_split=0.15):
    # Définir le seed pour la reproductibilité
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Transformations pour les images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset multitâche
    full_dataset = MultitaskDataset(img_root_dir, mask_root_dir, transform=None)
    
    # Calculer les tailles des splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Diviser le dataset en train, val, test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Compter le nombre unique de classes (à la fois pour la segmentation et la classification)
    n_classes = len(full_dataset.categories)
    
    # Appliquer les transformations
    train_dataset_transformed = TransformedMultitaskSubset(train_dataset, train_transform)
    val_dataset_transformed = TransformedMultitaskSubset(val_dataset, val_transform)
    test_dataset_transformed = TransformedMultitaskSubset(test_dataset, val_transform)
    
    # Créer les data loaders
    train_loader = DataLoader(
        train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    
    return train_loader, val_loader, test_loader, n_classes

# Fonction pour obtenir les data loaders de classification
def get_data_loaders(root_dir, batch_size=32, workers=4, train_split=0.7, val_split=0.15):
    # Transformations pour les images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset
    dataset = CategorizedDataset(root_dir, transform=train_transform)
    
    # Récupérer les classes
    categories = dataset.categories
    
    # Calculer les tailles des splits
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Diviser le dataset en train, val, test
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Appliquer des transformations différentes pour val et test
    val_dataset.dataset = CategorizedDataset(root_dir, transform=val_transform)
    test_dataset.dataset = CategorizedDataset(root_dir, transform=val_transform)
    
    # Créer les data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    
    return train_loader, val_loader, test_loader, categories

# Fonction pour obtenir les data loaders de classification à partir de données catégorisées
def get_categorized_data_loaders(root_dir, batch_size=32, workers=4, seed=42, train_split=0.7, val_split=0.15):
    # Définir le seed pour la reproductibilité
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Transformations pour les images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset
    full_dataset = CategorizedDataset(root_dir, transform=None)
    
    # Récupérer les catégories
    categories = full_dataset.categories
    
    # Calculer les tailles des splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Diviser le dataset en train, val, test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Créer des datasets avec les transformations appropriées
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __len__(self):
            return len(self.subset)
            
        def __getitem__(self, idx):
            img_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
            label = self.subset.dataset.labels[self.subset.indices[idx]]
            
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    # Appliquer les transformations
    train_dataset_transformed = TransformedSubset(train_dataset, train_transform)
    val_dataset_transformed = TransformedSubset(val_dataset, val_transform)
    test_dataset_transformed = TransformedSubset(test_dataset, val_transform)
    
    # Créer les data loaders
    train_loader = DataLoader(
        train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    
    return train_loader, val_loader, test_loader, categories

# Fonction pour obtenir les data loaders de segmentation
def get_categorized_mask_loaders(img_root_dir, mask_root_dir, batch_size=8, workers=4, seed=42, train_split=0.7, val_split=0.15):
    # Définir le seed pour la reproductibilité
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Transformations pour les images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Créer le dataset
    full_dataset = MaskDataset(img_root_dir, mask_root_dir, transform=None)
    
    # Calculer les tailles des splits
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Diviser le dataset en train, val, test
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Compter le nombre unique de classes dans les masques
    # Charger quelques masques et déterminer le nombre de classes
    n_classes = 0
    # Parcourir un petit nombre d'échantillons pour trouver le nombre de classes
    for i in range(min(100, len(full_dataset))):
        _, mask = full_dataset[i]
        n_classes = max(n_classes, torch.max(mask).item() + 1)
    
    # Créer des datasets avec les transformations appropriées
    train_dataset_transformed = TransformedMaskSubset(train_dataset, train_transform)
    val_dataset_transformed = TransformedMaskSubset(val_dataset, val_transform)
    test_dataset_transformed = TransformedMaskSubset(test_dataset, val_transform)
    
    # Créer les data loaders
    train_loader = DataLoader(
        train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    test_loader = DataLoader(
        test_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    
    return train_loader, val_loader, test_loader, n_classes 