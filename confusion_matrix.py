import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Importez vos modules personnalisés
from model.unet import UNet
from dataloader.load_data import get_categorized_mask_loaders

def parse_args():
    parser = argparse.ArgumentParser(description='Génération de matrice de confusion pour segmentation')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--thumbnail_dir', type=str, default='categorized_thumbnail_windows_224_224', 
                        help='Dossier contenant les images')
    parser.add_argument('--mask_dir', type=str, default='categorized_mask', 
                        help='Dossier contenant les masques')
    parser.add_argument('--output_dir', type=str, default='confusion_matrix', 
                        help='Dossier pour sauvegarder la matrice de confusion')
    parser.add_argument('--n_classes', type=int, default=None, 
                        help='Nombre de classes (déterminé automatiquement si non spécifié)')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille des batches')
    parser.add_argument('--class_mapping_file', type=str, default='categorized_mask/class_mapping.txt',
                        help='Fichier de mapping des classes')
    return parser.parse_args()

def get_class_mapping(mapping_file):
    class_names = {}
    try:
        # Essayer d'abord avec utf-8
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '->' in line and not line.startswith('Valeur'):
                    parts = line.split('->')
                    if len(parts) == 2:
                        class_val = int(parts[0].strip())
                        class_name = parts[1].strip()
                        class_names[class_val] = class_name
    except UnicodeDecodeError:
        # Si ça échoue, essayer avec latin-1 (ou cp1252 pour Windows)
        with open(mapping_file, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if '->' in line and not line.startswith('Valeur'):
                    parts = line.split('->')
                    if len(parts) == 2:
                        class_val = int(parts[0].strip())
                        class_name = parts[1].strip()
                        class_names[class_val] = class_name
    return class_names

def main():
    # Récupérer les arguments
    args = parse_args()
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Détecter le GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger les données
    _, val_loader, test_loader, n_classes = get_categorized_mask_loaders(
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
    
    # Obtenir les prédictions et les vérités terrain
    all_preds = []
    all_targets = []
    
    print("Évaluation sur l'ensemble de test...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Aplatir pour la matrice de confusion
            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_targets.extend(masks)
    
    # Calculer la matrice de confusion
    print("Génération de la matrice de confusion...")
    cm = confusion_matrix(all_targets, all_preds, labels=range(n_classes))
    
    # Normaliser par ligne (pourcentage d'échantillons correctement prédits pour chaque classe réelle)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_norm = np.round(cm_norm * 100, 2)  # Pourcentage avec 2 décimales
    
    # Charger le mapping des classes
    class_mapping = get_class_mapping(args.class_mapping_file)
    class_names = [class_mapping.get(i, f"Classe {i}") for i in range(n_classes)]
    
    # Créer une figure plus grande pour la lisibilité
    plt.figure(figsize=(14, 12))
    
    # Afficher la matrice de confusion
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.title('Matrice de confusion normalisée (%)')
    
    # Rotation des étiquettes pour la lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder la matrice de confusion
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Sauvegarder également la matrice brute pour référence
    np.save(os.path.join(args.output_dir, 'confusion_matrix_raw.npy'), cm)
    
    # Calculer et afficher les métriques globales
    accuracy = np.trace(cm) / np.sum(cm)
    class_accuracies = np.diag(cm_norm)
    
    print(f"Précision globale: {accuracy:.4f}")
    print("\nPrécision par classe:")
    for i, acc in enumerate(class_accuracies):
        print(f"{class_names[i]}: {acc:.2f}%")
    
    # Sauvegarder les métriques dans un fichier texte
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Précision globale: {accuracy:.4f}\n\n")
        f.write("Précision par classe:\n")
        for i, acc in enumerate(class_accuracies):
            f.write(f"{class_names[i]}: {acc:.2f}%\n")
    
    print(f"Résultats sauvegardés dans: {args.output_dir}")

if __name__ == "__main__":
    main()