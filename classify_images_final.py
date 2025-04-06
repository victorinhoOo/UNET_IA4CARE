#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de classification d'images en fonction des annotations COCO

Ce script traite les images d'un dossier source et les classe dans des dossiers 
organisés par catégorie selon les annotations définies dans un fichier JSON au format COCO.
"""

import json
import os
import shutil
import re
import argparse
from datetime import datetime

def clean_folder_name(name):
    """
    Nettoie et normalise un nom de dossier pour qu'il soit compatible avec le système de fichiers.
    
    Args:
        name (str): Nom à nettoyer
    
    Returns:
        str: Nom nettoyé
    """
    # Remplacer les caractères accentués et spéciaux
    name = name.replace('Ã¯', 'i').replace('Ã©', 'e').replace('Ã§', 'c')
    name = name.replace('Ã´', 'o').replace('Ã', 'e')
    # Supprime tout caractère non alphanumérique sauf espaces et tirets
    name = re.sub(r'[^\w\s-]', '_', name)
    return name

def main():
    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Classer les images selon les annotations COCO')
    parser.add_argument('--source', default='masks_224x224', 
                        help='Dossier source contenant les images à classer')
    parser.add_argument('--output', default='categorized_masks_224_224', 
                        help='Dossier de sortie où les sous-dossiers par catégorie seront créés')
    parser.add_argument('--annotations', default='coco_export/annotations_coco.json', 
                        help='Fichier JSON contenant les annotations au format COCO')
    args = parser.parse_args()
    
    # Début du timer pour mesurer le temps d'exécution
    start_time = datetime.now()
    
    # Chemins des dossiers
    source_folder = args.source
    base_output_folder = args.output
    annotations_file = args.annotations
    
    print(f"Classification des images de {source_folder}")
    print(f"Dossier de sortie: {base_output_folder}")
    print(f"Fichier d'annotations: {annotations_file}")
    
    # Charger le fichier JSON
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier JSON: {e}")
        return
    
    # Vérifier si les données nécessaires existent
    if 'annotations' not in data or 'categories' not in data:
        print("Le fichier JSON ne contient pas les données nécessaires (annotations ou catégories)")
        return
    
    # Créer un dictionnaire des catégories (id -> nom)
    categories = {}
    category_counts = {}  # Pour compter le nombre d'images par catégorie
    for category in data['categories']:
        category_id = category['id']
        category_name = category['name']
        clean_name = clean_folder_name(category_name)
        categories[category_id] = clean_name
        category_counts[category_id] = 0
        
        # Créer le dossier de catégorie s'il n'existe pas déjà
        category_folder = os.path.join(base_output_folder, clean_name)
        os.makedirs(category_folder, exist_ok=True)
    
    print(f"Nombre de catégories: {len(categories)}")
    
    # Créer un dictionnaire pour associer les IDs d'annotation aux IDs de catégorie
    annotation_to_category = {}
    for annotation in data['annotations']:
        annotation_id = annotation['id']
        category_id = annotation['category_id']
        annotation_to_category[annotation_id] = category_id
    
    print(f"Nombre d'annotations: {len(annotation_to_category)}")
    
    # Parcourir les fichiers dans le dossier source
    if not os.path.exists(source_folder):
        print(f"Le dossier source {source_folder} n'existe pas")
        return
    
    total_files = 0
    processed_files = 0
    skipped_files = 0
    invalid_id_files = 0
    
    # Liste tous les fichiers du dossier source
    try:
        files = os.listdir(source_folder)
        total_files = len([f for f in files if f.endswith('.png')])
    except Exception as e:
        print(f"Erreur lors de la lecture du dossier source: {e}")
        return
    
    print(f"\nTotal de fichiers PNG trouvés: {total_files}")
    print("Traitement en cours...")
    
    for filename in files:
        # Ne traiter que les fichiers PNG
        if not filename.endswith('.png'):
            continue
        
        # Essayer de traiter les fichiers avec et sans préfixe "._"
        original_filename = filename
        if filename.startswith('._'):
            # Enlever le préfixe "._" pour l'extraction de l'ID
            filename_for_id = filename[2:]
        else:
            filename_for_id = filename
        
        # Extraire l'ID de l'annotation depuis le nom du fichier
        match = re.search(r'_ann_(\d+)\.png$', filename_for_id)
        if not match:
            skipped_files += 1
            continue
        
        annotation_id = int(match.group(1))
        
        # Vérifier si l'ID d'annotation existe dans notre dictionnaire
        if annotation_id in annotation_to_category:
            category_id = annotation_to_category[annotation_id]
            category_name = categories.get(category_id)
            
            if category_name:
                # Créer le chemin du fichier source et destination
                source_file = os.path.join(source_folder, original_filename)
                destination_folder = os.path.join(base_output_folder, category_name)
                destination_file = os.path.join(destination_folder, original_filename)
                
                # Copier le fichier dans le dossier de catégorie correspondant
                try:
                    shutil.copy2(source_file, destination_file)
                    processed_files += 1
                    category_counts[category_id] += 1
                    # Afficher un '.' pour chaque 10 fichiers traités pour montrer la progression
                    if processed_files % 10 == 0:
                        print('.', end='', flush=True)
                    if processed_files % 500 == 0:
                        print(f" {processed_files}", end='', flush=True)
                except Exception as e:
                    print(f"\nErreur lors de la copie de {original_filename}: {e}")
        else:
            invalid_id_files += 1
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    print("\n\nRapport de traitement:")
    print(f"Durée du traitement: {processing_time:.2f} secondes")
    print(f"Total de fichiers PNG: {total_files}")
    print(f"Fichiers traités avec succès: {processed_files}")
    print(f"Fichiers ignorés (format non reconnu): {skipped_files}")
    print(f"Fichiers avec ID non trouvé dans les annotations: {invalid_id_files}")
    
    print("\nDistribution par catégorie:")
    for category_id, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            category_name = categories.get(category_id, "Catégorie inconnue")
            print(f"  {category_name}: {count} images")
    
    print(f"\nLes images ont été classées dans le dossier: {base_output_folder}")

if __name__ == "__main__":
    main() 