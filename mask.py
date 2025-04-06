import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# === CHEMINS ===
input_dir = Path('categorized_thumbnail_windows_224_224')
output_dir = Path('categorized_mask_v2')
output_dir.mkdir(exist_ok=True)

# === CHARGEMENT DES CATÉGORIES ===
categories = [d for d in os.listdir(input_dir) 
              if os.path.isdir(input_dir / d) and not d.startswith('.')]
category_to_class = {cat: i+1 for i, cat in enumerate(sorted(categories))}
print(f"Classes détectées: {len(categories)}")
print(f"Mapping des classes: {category_to_class}")

# Vérifier si le dossier de debug existe, sinon le créer
debug_dir = output_dir / 'debug'
debug_dir.mkdir(exist_ok=True)

# Fonction pour détecter et segmenter les cellules dans une image
def segment_cells(image_path, debug=False):
    # Charger l'image avec OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Appliquer un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Appliquer une binarisation adaptative pour identifier les cellules
    # Les paramètres peuvent nécessiter des ajustements selon vos images
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Opérations morphologiques pour nettoyer le masque
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours des cellules
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours trop petits (bruit)
    min_contour_area = 50  # Ajuster selon la taille de vos cellules
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Si aucun contour valide n'est trouvé, utiliser une approche différente
    if not valid_contours:
        # Essayer une autre méthode de segmentation (seuillage d'Otsu)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # Trouver les contours avec cette nouvelle méthode
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    # Si toujours aucun contour valide, considérer l'image entière comme une cellule
    if not valid_contours:
        height, width = gray.shape
        # Créer un contour rectangulaire pour toute l'image
        rect_contour = np.array([[[10, 10]], [[width-10, 10]], [[width-10, height-10]], [[10, height-10]]])
        valid_contours = [rect_contour]
    
    # Créer le masque de segmentation
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    # Optionnel : ajouter des bordures aux cellules pour mieux les visualiser
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, valid_contours, -1, 255, 2)
    
    if debug:
        # Retourner les masques et contours pour le débogage
        return mask, contour_mask, valid_contours
    else:
        return mask, valid_contours

# === GÉNÉRATION DES MASQUES PAR CATÉGORIE ===
for category in categories:
    category_dir = input_dir / category
    mask_category_dir = output_dir / category
    mask_category_dir.mkdir(exist_ok=True)
    
    # Liste tous les fichiers d'image dans la catégorie
    image_files = [f for f in os.listdir(category_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    
    print(f"Traitement de {len(image_files)} images pour la catégorie '{category}'")
    class_value = category_to_class[category]
    print(f"  Valeur de classe: {class_value}")
    
    for img_file in image_files:
        try:
            # Chemin de l'image source
            img_path = category_dir / img_file
            
            # Nom du fichier de masque (même nom que l'image)
            mask_file = img_file
            mask_path = mask_category_dir / mask_file
            
            # Segmenter les cellules dans l'image
            if img_file == image_files[0]:  # Pour le premier fichier, générer des infos de debug
                segmentation_mask, contour_mask, contours = segment_cells(img_path, debug=True)
                
                # Sauvegarder le masque de contours pour le débogage
                cv2.imwrite(str(debug_dir / f"contours_{category}_{img_file}"), contour_mask)
                
                # Sauvegarder une visualisation de la segmentation
                original_img = cv2.imread(str(img_path))
                overlay = original_img.copy()
                # Dessiner les contours en rouge sur l'image originale
                cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
                # Mélanger avec l'image originale
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, original_img, 1-alpha, 0, original_img)
                cv2.imwrite(str(debug_dir / f"segmentation_{category}_{img_file}"), original_img)
            else:
                segmentation_mask, _ = segment_cells(img_path)
            
            # Créer le masque final avec la valeur de classe pour les cellules segmentées
            # 0 = fond, class_value = cellules
            final_mask = np.zeros_like(segmentation_mask)
            final_mask[segmentation_mask > 0] = class_value
            
            # Convertir en image PIL et sauvegarder
            mask_img = Image.fromarray(final_mask.astype(np.uint8))
            mask_img.save(mask_path)
            
            # Pour le premier fichier de chaque catégorie, créer des versions de debug
            if img_file == image_files[0]:
                # 1. Version colorée pour la vérification des classes
                colors = [
                    (255, 0, 0, 128),     # Rouge
                    (0, 255, 0, 128),     # Vert
                    (0, 0, 255, 128),     # Bleu
                    (255, 255, 0, 128),   # Jaune
                    (255, 0, 255, 128),   # Magenta
                    (0, 255, 255, 128),   # Cyan
                    (128, 0, 0, 128),     # Rouge foncé
                    (0, 128, 0, 128),     # Vert foncé
                    (0, 0, 128, 128),     # Bleu foncé
                    (128, 128, 0, 128),   # Olive
                    (128, 0, 128, 128),   # Pourpre
                    (0, 128, 128, 128),   # Teal
                    (192, 192, 192, 128), # Argent
                    (128, 128, 128, 128), # Gris
                    (255, 128, 0, 128),   # Orange
                ]
                
                # Créer une version plus visible du masque pour debugging
                visible_mask = np.zeros_like(segmentation_mask)
                visible_mask[segmentation_mask > 0] = class_value * 50  # Amplifier pour la visibilité
                visible_mask_img = Image.fromarray(visible_mask.astype(np.uint8))
                visible_mask_img.save(debug_dir / f"visible_{category}_{img_file}")
                
                # Créer une version colorée pour la visualisation des classes
                color_idx = (class_value - 1) % len(colors)
                
                # Créer un masque RGBA avec le canal alpha pour la transparence
                color_mask = Image.new("RGBA", mask_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(color_mask)
                
                # Obtenir le masque comme array numpy pour le traitement
                mask_array = np.array(mask_img)
                
                # Convertir l'image originale en RGBA
                original_img = Image.open(img_path).convert("RGBA")
                
                # Créer un overlay coloré pour les régions de cellules
                overlay = Image.new("RGBA", original_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Pour chaque point du masque qui a la valeur de classe
                for y in range(mask_array.shape[0]):
                    for x in range(mask_array.shape[1]):
                        if mask_array[y, x] == class_value:
                            draw.point((x, y), colors[color_idx])
                
                # Superposer sur l'image originale
                composite = Image.alpha_composite(original_img, overlay)
                composite.save(debug_dir / f"overlay_{category}_{img_file}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {e}")

print("\nGénération des masques terminée")
print(f"Images source: {input_dir}")
print(f"Masques de classe: {output_dir}")
print(f"Images de debug: {debug_dir}")

# Créer un fichier de correspondance des classes
with open(output_dir / 'class_mapping.txt', 'w') as f:
    f.write("Valeur de classe -> Nom de catégorie\n")
    f.write("0 -> Fond (arrière-plan)\n")
    for cat, val in sorted(category_to_class.items(), key=lambda x: x[1]):
        f.write(f"{val} -> {cat}\n")

print(f"Fichier de mapping des classes créé: {output_dir / 'class_mapping.txt'}")

# === VISUALISATION (OPTIONNEL) ===
try:
    import matplotlib.pyplot as plt
    import random
    
    # Sélectionner quelques catégories aléatoires pour la visualisation
    sample_categories = random.sample(categories, min(3, len(categories)))
    
    for cat in sample_categories:
        # Obtenir un fichier image aléatoire
        cat_dir = input_dir / cat
        image_files = [f for f in os.listdir(cat_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        
        if image_files:
            sample_file = random.choice(image_files)
            
            # Charger l'image et le masque de classe
            img_path = input_dir / cat / sample_file
            mask_path = output_dir / cat / sample_file
            
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            # Créer une visualisation
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f"Image: {cat}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='viridis')
            plt.title(f"Masque classe: {category_to_class[cat]}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(debug_dir / f"compare_{cat}_{sample_file}")
            plt.close()
    
    # Créer une visualisation du mapping des classes
    plt.figure(figsize=(12, 8))
    
    for cat, val in sorted(category_to_class.items(), key=lambda x: x[1]):
        # Créer un carré de couleur pour chaque classe
        color_idx = (val - 1) % len(colors)
        color = [c/255 for c in colors[color_idx][:3]]  # Convertir en format matplotlib
        
        plt.bar(cat, 1, color=color, alpha=0.7)
        plt.text(cat, 0.5, f"Classe {val}", ha='center', va='center', rotation=90)
    
    plt.title("Mapping des classes")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_dir / "class_mapping_visualization.png")
    plt.close()
    
    print(f"Visualisations créées dans le dossier {debug_dir}")
    
except Exception as e:
    print(f"Impossible de générer la visualisation: {e}")

print("\nIMPORTANT:")
print(f"1. Les masques dans le dossier '{output_dir}' contiennent maintenant des segmentations de cellules.")
print(f"   - Valeur 0: Fond (arrière-plan)")
print(f"   - Valeur 1-{len(categories)}: Cellules selon leur classe")
print(f"2. Vérifiez les images de debug dans le dossier {debug_dir} pour confirmer que la segmentation est correcte.")
print(f"3. Les masques peuvent nécessiter des ajustements des paramètres de segmentation selon vos types d'images.")