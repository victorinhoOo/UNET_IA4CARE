import os
from pathlib import Path
from PIL import Image, ImageDraw

# === CHEMINS ===
input_dir = Path('categorized_thumbnail_windows_224_224')
output_dir = Path('categorized_mask')
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
            
            # Ouvrir l'image pour obtenir ses dimensions
            image = Image.open(img_path)
            width, height = image.size
            
            # Nom du fichier de masque (même nom que l'image)
            mask_file = img_file
            mask_path = mask_category_dir / mask_file
            
            # Créer le masque avec valeur de classe (pour l'entraînement)
            # La valeur de pixel correspond à la classe de la catégorie
            mask = Image.new("L", (width, height), class_value)
            mask.save(mask_path)
            
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
                visible_mask = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(visible_mask)
                draw.rectangle([(0, 0), (width, height)], fill=class_value*50)  # Valeur amplifiée pour la visibilité
                visible_mask.save(debug_dir / f"visible_{category}_{img_file}")
                
                # Créer une version colorée pour la visualisation des classes
                color_idx = (class_value - 1) % len(colors)
                color_mask = Image.new("RGBA", (width, height), colors[color_idx])
                
                # Superposer sur l'image originale pour vérification
                image_rgba = image.convert("RGBA")
                composite = Image.alpha_composite(image_rgba, color_mask)
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
print(f"1. Les masques dans le dossier '{output_dir}' contiennent les valeurs de classe (1-{len(categories)}).")
print("   Ils sont destinés à l'entraînement du modèle de segmentation multiclasse.")
print(f"2. Les masques peuvent apparaître très sombres lors de la visualisation car les valeurs")
print("   de classe (1-15) sont faibles en tant que valeurs de pixels en niveaux de gris.")
print(f"3. Vérifiez les images de debug dans le dossier {debug_dir} pour confirmer que tout est correct.")