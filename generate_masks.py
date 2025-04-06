import os
from pathlib import Path
import json
from PIL import Image, ImageDraw
import shutil
import numpy as np

# === CHEMINS ===
input_dir = Path('categorized_thumbnail_windows_224_224')
output_dir = Path('generated_data')
output_images_dir = output_dir / 'images'
output_masks_dir = output_dir / 'masks'

# Créer les dossiers de sortie
output_dir.mkdir(exist_ok=True)
output_images_dir.mkdir(exist_ok=True)
output_masks_dir.mkdir(exist_ok=True)

# Dictionnaire pour mapper les catégories à des valeurs de classe
categories = [d for d in os.listdir(input_dir) 
              if os.path.isdir(input_dir / d) and not d.startswith('.')]
category_to_class = {cat: i+1 for i, cat in enumerate(sorted(categories))}
print(f"Classes détectées: {category_to_class}")

# Liste pour sauvegarder les informations sur les images
image_info = []

# Compteur d'images
image_counter = 0

# === GÉNÉRATION DES MASQUES ===
print("Génération des masques...")
for category in categories:
    category_dir = input_dir / category
    
    # Liste tous les fichiers d'image dans la catégorie
    image_files = [f for f in os.listdir(category_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    
    for img_file in image_files:
        try:
            # Chemin de l'image source
            img_path = category_dir / img_file
            
            # Ouvrir l'image
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            
            # Générer un nom unique pour l'image et le masque
            base_name = f"{category}_{img_file.split('.')[0]}"
            output_img_name = f"{base_name}.png"
            output_mask_name = f"{base_name}_mask.png"
            
            # Copier l'image vers le dossier de sortie
            image.save(output_images_dir / output_img_name)
            
            # Créer le masque
            # Pour la segmentation, nous considérons que l'ensemble de l'image appartient à la classe de la catégorie
            mask = Image.new("L", (width, height), 0)  # Fond = 0
            draw = ImageDraw.Draw(mask)
            
            # On dessine un rectangle couvrant toute l'image avec la valeur de classe
            class_value = category_to_class[category]
            draw.rectangle([(0, 0), (width, height)], fill=class_value)
            
            # Sauvegarde du masque
            mask.save(output_masks_dir / output_mask_name)
            
            # Enregistrer les informations sur l'image
            image_info.append({
                "id": image_counter,
                "file_name": output_img_name,
                "mask_name": output_mask_name,
                "category": category,
                "class_value": class_value,
                "width": width,
                "height": height
            })
            
            image_counter += 1
            
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {e}")
    
    print(f"Traité {len(image_files)} images pour la catégorie '{category}'")

# Sauvegarde des informations sur les images dans un fichier JSON
with open(output_dir / 'image_info.json', 'w') as f:
    json.dump({"images": image_info, "categories": category_to_class}, f, indent=2)

# Créer un fichier de correspondance des classes
with open(output_dir / 'class_mapping.txt', 'w') as f:
    f.write("Valeur de classe -> Nom de catégorie\n")
    f.write("0 -> Fond\n")
    for cat, val in sorted(category_to_class.items(), key=lambda x: x[1]):
        f.write(f"{val} -> {cat}\n")

print(f"Génération terminée. {image_counter} images et masques ont été générés.")
print(f"Images: {output_images_dir}")
print(f"Masques: {output_masks_dir}")
print(f"Correspondance des classes: {output_dir / 'class_mapping.txt'}")
print(f"Informations des images: {output_dir / 'image_info.json'}")

# Optional: Generate a few example visualizations
try:
    import matplotlib.pyplot as plt
    import random
    
    # Select 3 random images to visualize
    sample_indices = random.sample(range(len(image_info)), min(3, len(image_info)))
    
    for idx in sample_indices:
        info = image_info[idx]
        
        # Open the image and its mask
        img = Image.open(output_images_dir / info["file_name"])
        mask = Image.open(output_masks_dir / info["mask_name"])
        
        # Create a visualization
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Image: {info['category']}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='viridis')
        plt.title(f"Masque (Classe: {info['class_value']})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"example_{idx}.png")
        plt.close()
    
    print(f"Des exemples de visualisation ont été générés dans le dossier {output_dir}")
    
except Exception as e:
    print(f"Impossible de générer les visualisations: {e}") 