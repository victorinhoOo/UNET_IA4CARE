import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import datetime

# Import des modules personnalisés
from model.unet import UNet
from utils.tiles import TileMerger

class ImageDataset(Dataset):
    """Dataset pour les images à segmenter"""
    
    def __init__(self, img_dir, transform=None):
        """
        Initialisation du dataset
        
        Args:
            img_dir (str): Chemin vers le dossier des images
            transform: Transformations à appliquer aux images
        """
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Trouver tous les fichiers d'images
        self.img_files = []
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(valid_extensions):
                self.img_files.append(os.path.join(img_dir, img_file))
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Charger l'image
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Stocker les dimensions originales
            original_width, original_height = image.size
            
            # Appliquer les transformations
            if self.transform:
                image = self.transform(image)
            
            return image, img_path, (original_width, original_height)
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            # En cas d'erreur, retourner une image noire
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, img_path, (224, 224)

def parse_args():
    parser = argparse.ArgumentParser(description='Prédiction avec le modèle UNet++ pour la segmentation')
    
    # Chemins
    parser.add_argument('--img_dir', type=str, required=True, 
                        help='Dossier contenant les images à analyser')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Chemin vers le modèle UNet++ entraîné')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Dossier où sauvegarder les résultats')
    
    # Paramètres du modèle
    parser.add_argument('--n_classes', type=int, default=6, 
                        help='Nombre de classes pour la segmentation')
    parser.add_argument('--encoder', type=str, default='resnet34', 
                        help='Encodeur utilisé dans le modèle')
    
    # Paramètres pour le tiling (découpage en tuiles)
    parser.add_argument('--use_tiling', action='store_true', 
                        help='Utiliser le tiling pour les images larges')
    parser.add_argument('--tile_size', type=int, default=224, 
                        help='Taille des tuiles')
    parser.add_argument('--tile_overlap', type=int, default=32, 
                        help='Chevauchement des tuiles en pixels')
    
    # Paramètres pour la visualisation
    parser.add_argument('--no_visualization', action='store_true', 
                        help='Ne pas générer de visualisations')
    parser.add_argument('--show_confidence', action='store_true', 
                        help='Afficher les cartes de confiance')
    
    # Autres paramètres
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Taille des batches')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Nombre de workers pour le chargement des données')
    
    return parser.parse_args()

def preprocess_large_image(image_path, tile_size=224, overlap=32):
    """
    Prétraite une image large en la découpant en tuiles
    
    Args:
        image_path (str): Chemin vers l'image
        tile_size (int): Taille des tuiles
        overlap (int): Chevauchement des tuiles
        
    Returns:
        tuple: (tiles, coordinates, original_size)
            - tiles: Liste des tuiles (torch.Tensor)
            - coordinates: Liste des coordonnées (x, y) des tuiles
            - original_size: Dimensions originales de l'image (width, height)
    """
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # Calculer le pas
    step_size = tile_size - overlap
    
    # Initialiser les listes
    tiles = []
    coordinates = []
    
    # Transformer pour normaliser
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Découper l'image en tuiles
    for y in range(0, image.height, step_size):
        for x in range(0, image.width, step_size):
            # Extraire la tuile
            box = (x, y, min(x + tile_size, image.width), min(y + tile_size, image.height))
            tile = image.crop(box)
            
            # Redimensionner si nécessaire
            if tile.width < tile_size or tile.height < tile_size:
                new_tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                new_tile.paste(tile, (0, 0))
                tile = new_tile
            
            # Transformer la tuile
            tile_tensor = transform(tile)
            tiles.append(tile_tensor)
            coordinates.append((x, y))
    
    return tiles, coordinates, original_size

def predict_large_image(model, image_path, n_classes, device, tile_size=224, overlap=32):
    """
    Prédit la segmentation pour une image large
    
    Args:
        model: Modèle UNet++
        image_path (str): Chemin vers l'image
        n_classes (int): Nombre de classes
        device: Device (CPU/GPU)
        tile_size (int): Taille des tuiles
        overlap (int): Chevauchement des tuiles
        
    Returns:
        tuple: (merged_prediction, confidence_map)
    """
    # Prétraiter l'image
    tiles, coordinates, (width, height) = preprocess_large_image(image_path, tile_size, overlap)
    
    # Calculer le pas
    step_size = tile_size - overlap
    
    # Initialiser le fusionneur de tuiles
    merger = TileMerger(height, width, n_classes, step_size, tile_size)
    
    # Prédire pour chaque tuile
    model.eval()
    with torch.no_grad():
        for tile, (x, y) in zip(tiles, coordinates):
            tile = tile.unsqueeze(0).to(device)  # Ajouter dimension de batch
            pred = model(tile)
            merger.add_tile(pred, x, y)
    
    # Fusionner les prédictions
    merged_prediction = merger.merge()
    
    # Calculer la carte de confiance (max de softmax)
    probs = np.max(merged_prediction, axis=0)
    
    return merged_prediction, probs

def visualize_prediction(image, prediction, confidence=None, output_path=None, class_colors=None, alpha=0.5):
    """
    Visualise la prédiction de segmentation
    
    Args:
        image: Image originale (PIL ou numpy)
        prediction: Prédiction (numpy array [C, H, W] ou [H, W])
        confidence: Carte de confiance (optional)
        output_path: Chemin où sauvegarder la visualisation
        class_colors: Couleurs pour chaque classe
        alpha: Transparence de la segmentation
    """
    # Assurer que l'image est en numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Créer une figure
    fig, axes = plt.subplots(1, 3 if confidence is not None else 2, figsize=(15, 5))
    
    # Afficher l'image originale
    axes[0].imshow(image)
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Préparer la carte de couleurs pour la segmentation
    if class_colors is None:
        # Couleurs par défaut pour 6 classes
        class_colors = [
            [0, 0, 0],        # Fond (noir)
            [255, 0, 0],      # Classe 1 (rouge)
            [0, 255, 0],      # Classe 2 (vert)
            [0, 0, 255],      # Classe 3 (bleu)
            [255, 255, 0],    # Classe 4 (jaune)
            [255, 0, 255]     # Classe 5 (magenta)
        ]
    
    # Convertir en matrice [H, W] si la prédiction est [C, H, W]
    if len(prediction.shape) == 3:
        pred_mask = np.argmax(prediction, axis=0)
    else:
        pred_mask = prediction
    
    # Créer un masque coloré
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(class_colors)):
        colored_mask[pred_mask == class_idx] = class_colors[class_idx]
    
    # Afficher le masque de segmentation
    axes[1].imshow(colored_mask)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Afficher la confiance si disponible
    if confidence is not None:
        axes[2].imshow(confidence, cmap='inferno')
        axes[2].set_title('Confiance')
        axes[2].axis('off')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_report(prediction, image_path, n_classes, output_path=None):
    """
    Génère un rapport d'analyse basé sur la segmentation
    
    Args:
        prediction: Prédiction (numpy array [C, H, W])
        image_path: Chemin de l'image analysée
        n_classes: Nombre de classes
        output_path: Chemin où sauvegarder le rapport
        
    Returns:
        dict: Rapport généré
    """
    # Calculer les statistiques de la segmentation
    if len(prediction.shape) == 3:
        pred_mask = np.argmax(prediction, axis=0)
    else:
        pred_mask = prediction
    
    # Calculer la proportion de chaque classe
    class_proportions = {}
    total_pixels = pred_mask.size
    for class_idx in range(n_classes):
        count = np.sum(pred_mask == class_idx)
        proportion = count / total_pixels
        class_proportions[f"class_{class_idx}"] = float(proportion)
    
    # Noms des classes (à adapter selon votre application)
    class_names = {
        0: "Fond",
        1: "Cellules inflammatoires",
        2: "Cellules vésiculaires",
        3: "Cellules épithéliales",
        4: "Colloïde",
        5: "Artefacts"
    }
    
    # Déterminer la catégorie Bethesda basée sur les proportions
    bethesda_category = determine_bethesda_category(class_proportions)
    
    # Créer le rapport
    report = {
        "image_path": image_path,
        "date_analyse": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "composition_cellulaire": {
            class_names.get(i, f"Classe {i}"): round(prop * 100, 2)
            for i, prop in enumerate(class_proportions.values())
        },
        "classification_bethesda": {
            "categorie": bethesda_category,
            "confiance": calculate_confidence_score(class_proportions, bethesda_category)
        },
        "recommandations": get_recommendations(bethesda_category)
    }
    
    # Sauvegarder le rapport si un chemin est spécifié
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
    
    return report

def determine_bethesda_category(class_proportions):
    """
    Détermine la catégorie Bethesda basée sur les proportions de classes
    
    Args:
        class_proportions: Dictionnaire des proportions de classes
        
    Returns:
        str: Catégorie Bethesda (I-VI)
    """
    # Extraire les proportions
    background = class_proportions.get("class_0", 0)
    inflammatory = class_proportions.get("class_1", 0)
    vesicular = class_proportions.get("class_2", 0)
    epithelial = class_proportions.get("class_3", 0)
    colloid = class_proportions.get("class_4", 0)
    artifacts = class_proportions.get("class_5", 0)
    
    # Logique de classification (à adapter selon les critères médicaux)
    total_cells = inflammatory + vesicular + epithelial
    
    # Bethesda I: Non diagnostique (pas assez de cellules ou trop d'artefacts)
    if total_cells < 0.05 or artifacts > 0.3:
        return "I - Non diagnostique/Non satisfaisant"
    
    # Bethesda II: Bénin (prédominance de cellules inflammatoires et vésiculaires avec colloïde)
    if (inflammatory + vesicular > 0.6 * total_cells) and colloid > 0.1:
        return "II - Bénin"
    
    # Bethesda III: Atypie de signification indéterminée
    if (vesicular > inflammatory) and (epithelial > 0.05) and (colloid < 0.1):
        return "III - Atypie de signification indéterminée"
    
    # Bethesda IV: Néoplasie folliculaire (forte proportion de cellules vésiculaires, peu de colloïde)
    if vesicular > 0.6 * total_cells and colloid < 0.05:
        return "IV - Néoplasie folliculaire"
    
    # Bethesda V: Suspect de malignité (forte proportion de cellules épithéliales)
    if epithelial > 0.2 * total_cells:
        return "V - Suspect de malignité"
    
    # Bethesda VI: Malin (très forte proportion de cellules épithéliales)
    if epithelial > 0.4 * total_cells:
        return "VI - Malin"
    
    # Par défaut
    return "II - Bénin"

def calculate_confidence_score(class_proportions, bethesda_category):
    """
    Calcule un score de confiance pour la classification Bethesda
    
    Args:
        class_proportions: Dictionnaire des proportions de classes
        bethesda_category: Catégorie Bethesda attribuée
        
    Returns:
        float: Score de confiance (0-1)
    """
    # Logique simplifiée pour le score de confiance (à adapter)
    # Plus les proportions sont nettement marquées, plus la confiance est élevée
    if bethesda_category == "I - Non diagnostique/Non satisfaisant":
        return 0.9  # Généralement sûr de cette classification
    
    # Calculer la "netteté" des proportions (écart-type)
    values = list(class_proportions.values())
    std_dev = np.std(values)
    
    # Plus l'écart est grand, plus les classes sont distinctes
    confidence = min(0.5 + std_dev * 5, 0.95)
    
    return float(confidence)

def get_recommendations(bethesda_category):
    """
    Fournit des recommandations basées sur la catégorie Bethesda
    
    Args:
        bethesda_category: Catégorie Bethesda
        
    Returns:
        str: Recommandations
    """
    recommendations = {
        "I - Non diagnostique/Non satisfaisant": 
            "Répétition de la cytoponction recommandée.",
        
        "II - Bénin": 
            "Suivi clinique et échographique régulier.",
        
        "III - Atypie de signification indéterminée": 
            "Répétition de la cytoponction dans 3-6 mois ou corrélation avec les données cliniques.",
        
        "IV - Néoplasie folliculaire": 
            "Corrélation avec les données cliniques et échographiques. Une lobectomie peut être envisagée.",
        
        "V - Suspect de malignité": 
            "Corrélation avec les données cliniques et échographiques. Une thyroïdectomie est généralement recommandée.",
        
        "VI - Malin": 
            "Thyroïdectomie totale recommandée avec analyse histologique complète."
    }
    
    return recommendations.get(bethesda_category, "Consultation de spécialiste recommandée.")

def predict_image(model, image_path, device, n_classes, use_tiling=False, tile_size=224, overlap=32):
    """
    Prédit la segmentation pour une image
    
    Args:
        model: Modèle UNet++
        image_path: Chemin vers l'image
        device: Device (CPU/GPU)
        n_classes: Nombre de classes
        use_tiling: Utiliser le tiling pour les images larges
        tile_size: Taille des tuiles
        overlap: Chevauchement des tuiles
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    
    # Si l'image est large et tiling activé, utiliser la fonction de tiling
    if use_tiling and (image.width > tile_size or image.height > tile_size):
        return predict_large_image(model, image_path, n_classes, device, tile_size, overlap)
    
    # Sinon, traiter l'image directement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prédire
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.softmax(pred, dim=1)
        
    # Extraire la prédiction et la confiance
    pred_np = pred.squeeze().cpu().numpy()
    confidence = np.max(pred_np, axis=0)
    
    return pred_np, confidence

def main():
    # Récupérer les arguments
    args = parse_args()
    
    # Détecter le GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Créer les dossiers de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'segmentations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)
    if not args.no_visualization:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Charger le modèle
    model = UNet(n_classes=args.n_classes, encoder=args.encoder)
    
    # Charger les poids du modèle
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print(f"Modèle chargé depuis {args.model_path}")
    
    # Créer un dataset pour les images
    if not args.use_tiling:
        dataset = ImageDataset(args.img_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        
        # Prédire pour chaque image
        model.eval()
        for images, paths, sizes in tqdm(dataloader, desc="Analyse des images"):
            images = images.to(device)
            
            # Prédire
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                confs = torch.max(probs, dim=1)[0]
            
            # Traiter chaque image du batch
            for i in range(images.shape[0]):
                # Obtenir le chemin et la taille originale
                img_path = paths[i]
                original_size = sizes[i]
                
                # Extraire le nom du fichier
                filename = os.path.basename(img_path)
                base_filename, _ = os.path.splitext(filename)
                
                # Redimensionner aux dimensions originales
                pred_np = preds[i].cpu().numpy()
                conf_np = confs[i].cpu().numpy()
                
                if pred_np.shape != original_size[::-1]:  # Inverser car size est (width, height)
                    pred_np = cv2.resize(
                        pred_np, original_size, interpolation=cv2.INTER_NEAREST
                    )
                    conf_np = cv2.resize(
                        conf_np, original_size, interpolation=cv2.INTER_LINEAR
                    )
                
                # Sauvegarder la segmentation
                np.save(
                    os.path.join(args.output_dir, 'segmentations', f"{base_filename}_seg.npy"), 
                    pred_np
                )
                
                # Générer et sauvegarder le rapport
                report = generate_report(
                    pred_np, img_path, args.n_classes,
                    os.path.join(args.output_dir, 'reports', f"{base_filename}_report.json")
                )
                
                # Visualiser si demandé
                if not args.no_visualization:
                    img = Image.open(img_path).convert('RGB')
                    visualize_prediction(
                        img, pred_np, 
                        confidence=conf_np if args.show_confidence else None,
                        output_path=os.path.join(args.output_dir, 'visualizations', f"{base_filename}_viz.png")
                    )
    else:
        # Traiter les images en utilisant le tiling
        image_files = [f for f in os.listdir(args.img_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in tqdm(image_files, desc="Analyse des images"):
            img_path = os.path.join(args.img_dir, filename)
            base_filename, _ = os.path.splitext(filename)
            
            # Prédire avec tiling
            pred_np, conf_np = predict_image(
                model, img_path, device, args.n_classes,
                use_tiling=True, tile_size=args.tile_size, overlap=args.tile_overlap
            )
            
            # Sauvegarder la segmentation
            np.save(
                os.path.join(args.output_dir, 'segmentations', f"{base_filename}_seg.npy"),
                pred_np if len(pred_np.shape) == 2 else np.argmax(pred_np, axis=0)
            )
            
            # Générer et sauvegarder le rapport
            report = generate_report(
                pred_np, img_path, args.n_classes,
                os.path.join(args.output_dir, 'reports', f"{base_filename}_report.json")
            )
            
            # Visualiser si demandé
            if not args.no_visualization:
                img = Image.open(img_path).convert('RGB')
                visualize_prediction(
                    img, np.argmax(pred_np, axis=0) if len(pred_np.shape) == 3 else pred_np,
                    confidence=conf_np if args.show_confidence else None,
                    output_path=os.path.join(args.output_dir, 'visualizations', f"{base_filename}_viz.png")
                )
    
    print(f"Analyse terminée. Résultats sauvegardés dans {args.output_dir}")

if __name__ == "__main__":
    main() 