import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm
from torchvision import transforms

# Importez vos modules personnalisés
from model.unet import UNet
from dataloader.load_data import get_categorized_mask_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Visualisation des prédictions de segmentation')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--thumbnail_dir', type=str, default='categorized_thumbnail_windows_224_224', 
                        help='Dossier contenant les images')
    parser.add_argument('--mask_dir', type=str, default='categorized_mask', 
                        help='Dossier contenant les masques')
    parser.add_argument('--output_dir', type=str, default='prediction_samples', 
                        help='Dossier pour les visualisations')
    parser.add_argument('--n_classes', type=int, default=16, 
                        help='Nombre de classes (inclut le fond)')
    parser.add_argument('--n_samples', type=int, default=9, 
                        help='Nombre d\'échantillons à visualiser')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Taille des batches')
    parser.add_argument('--class_mapping_file', type=str, default='categorized_mask/class_mapping.txt',
                        help='Fichier de mapping des classes')
    return parser.parse_args()

def get_class_mapping(mapping_file):
    class_names = {}
    try:
        with open(mapping_file, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if '->' in line and not line.startswith('Valeur'):
                    parts = line.split('->')
                    if len(parts) == 2:
                        class_val = int(parts[0].strip())
                        class_name = parts[1].strip()
                        class_names[class_val] = class_name
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier de mapping: {e}")
        # Fallback basique
        for i in range(16):
            class_names[i] = f"Classe {i}"
    return class_names

def main():
    # Récupérer les arguments
    args = parse_args()
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Détecter le GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger le mapping des classes
    class_mapping = get_class_mapping(args.class_mapping_file)
    
    # Charger les données
    _, _, test_loader, n_classes = get_categorized_mask_loaders(
        img_root_dir=args.thumbnail_dir,
        mask_root_dir=args.mask_dir,
        batch_size=args.batch_size,
        workers=4,
        seed=42
    )
    
    if args.n_classes is not None:
        n_classes = args.n_classes
    print(f"Nombre de classes: {n_classes}")
    
    # Charger le modèle
    model = UNet(n_classes=n_classes)
    checkpoint = torch.load(args.model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Créer une liste pour stocker les échantillons
    samples = []
    
    # Collecter des échantillons aléatoires du loader de test
    print("Collecte des échantillons et prédictions...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.cpu().numpy()
            
            # Faire des prédictions
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confidences = torch.max(probs, dim=1)[0].cpu().numpy()
            
            # Pour chaque image dans ce batch
            for i in range(images.size(0)):
                # Ajouter l'échantillon à notre liste
                samples.append({
                    'image': images[i].cpu().numpy(),
                    'mask': masks[i],
                    'pred': preds[i],
                    'conf': confidences[i],
                })
                
            # Si nous avons assez d'échantillons, arrêter
            if len(samples) >= args.n_samples * 3:  # Collecter plus pour la sélection aléatoire
                break
    
    # Sélectionner aléatoirement n_samples
    if len(samples) > args.n_samples:
        samples = random.sample(samples, args.n_samples)
    
    # Calculer le nombre de lignes et de colonnes pour l'affichage
    n_rows = int(np.ceil(args.n_samples / 3))
    n_cols = min(3, args.n_samples)
    
    # Créer la figure
    plt.figure(figsize=(n_cols * 5, n_rows * 6))
    
    # Afficher chaque échantillon
    for i, sample in enumerate(samples):
        # Récupérer les données
        image = sample['image'].transpose(1, 2, 0)
        mask = sample['mask']
        pred = sample['pred']
        conf = sample['conf']
        
        # Dénormaliser l'image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Trouver les classes dominantes dans le masque et la prédiction
        mask_class = np.bincount(mask.flatten()).argmax()
        pred_class = np.bincount(pred.flatten()).argmax()
        
        # Calcul de la confiance moyenne pour la classe prédominante
        conf_mask = (pred == pred_class)
        mean_conf = np.mean(conf[conf_mask]) * 100 if np.any(conf_mask) else 0
        
        # Afficher l'image originale
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        
        # Ajouter le titre avec les classes et la confiance
        mask_name = class_mapping.get(mask_class, f"Classe {mask_class}")
        pred_name = class_mapping.get(pred_class, f"Classe {pred_class}")
        plt.title(f"Vérité: {mask_name}\nPréd: {pred_name}\nConf: {mean_conf:.1f}%", 
                  color='green' if mask_class == pred_class else 'red',
                  fontsize=10)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sample_predictions.png'), dpi=150)
    
    # Version avancée: afficher une grille avec image, masque réel et prédiction
    plt.figure(figsize=(n_cols * 6, n_rows * 6))
    
    for i, sample in enumerate(samples):
        # Récupérer les données
        image = sample['image'].transpose(1, 2, 0)
        mask = sample['mask']
        pred = sample['pred']
        conf = sample['conf']
        
        # Dénormaliser l'image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        
        # Trouver les classes dominantes
        mask_class = np.bincount(mask.flatten()).argmax()
        pred_class = np.bincount(pred.flatten()).argmax()
        
        # Calcul de la confiance moyenne
        mean_conf = np.mean(conf) * 100
        
        # Position dans la grille (3 images par échantillon)
        row = i // n_cols
        col = i % n_cols
        
        # Image originale
        plt.subplot(n_rows * 3, n_cols, row * 3 * n_cols + col + 1)
        plt.imshow(image)
        plt.title("Image originale", fontsize=10)
        plt.axis('off')
        
        # Masque réel
        plt.subplot(n_rows * 3, n_cols, row * 3 * n_cols + col + 1 + n_cols)
        plt.imshow(mask, cmap='viridis', vmin=0, vmax=n_classes-1)
        mask_name = class_mapping.get(mask_class, f"Classe {mask_class}")
        plt.title(f"Vérité: {mask_name}", fontsize=10)
        plt.axis('off')
        
        # Prédiction
        plt.subplot(n_rows * 3, n_cols, row * 3 * n_cols + col + 1 + 2*n_cols)
        plt.imshow(pred, cmap='viridis', vmin=0, vmax=n_classes-1)
        pred_name = class_mapping.get(pred_class, f"Classe {pred_class}")
        plt.title(f"Préd: {pred_name}\nConf: {mean_conf:.1f}%", 
                  color='green' if mask_class == pred_class else 'red',
                  fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'detailed_predictions.png'), dpi=150)
    print(f"Visualisations sauvegardées dans: {args.output_dir}")

if __name__ == "__main__":
    main()