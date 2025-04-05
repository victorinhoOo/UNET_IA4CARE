# Analyse de Cytoponction Thyroïdienne avec U-Net++

Ce projet implémente un système d'analyse automatique de cytoponctions thyroïdiennes utilisant la segmentation par U-Net++ et la classification par deep learning. Le système permet l'identification et la quantification des différents types cellulaires, puis propose une classification selon les catégories Bethesda.

## Caractéristiques principales

- **Architecture U-Net++** avec encodeur préentraîné pour une segmentation précise
- **Segmentation multiclasses** des différents types cellulaires
- **Classification Bethesda** automatique basée sur l'analyse quantitative
- **Génération de rapports standardisés** avec visualisations
- **Interface en ligne de commande** simple et flexible

## Structure du Projet

```
.
├── dataloader/            # Chargeurs de données
│   └── load_data.py       # Dataloaders et fonctions d'analyse
├── model/                 # Définition du modèle
│   └── unet.py            # Implémentation du modèle U-Net++
├── utils/                 # Utilitaires
│   └── tiles.py           # Gestion des tuiles pour grandes images
├── saved_models/          # Dossier pour stocker les modèles entraînés
├── results/               # Résultats de l'analyse
│   ├── segmentations/     # Images segmentées
│   ├── reports/           # Rapports JSON
│   └── visualizations/    # Visualisations
├── train.py               # Script d'entraînement
├── predict.py             # Script d'analyse et génération de rapports
└── requirements.txt       # Dépendances du projet
```

## Installation

```bash
pip install -r requirements.txt
```

## Fonctionnement
Le projet utilise la bibliothèque `segmentation-models-pytorch` qui fournit une implémentation optimisée de U-Net++, avec des encodeurs préentraînés comme ResNet34, EfficientNet, etc.

### Préparation des Données

Organisez vos images cellulaires et masques de segmentation dans des dossiers séparés. Pour la classification Bethesda, vous pouvez fournir un fichier CSV avec les étiquettes.

Format des données:
- Images: format 224x224 pixels en RGB
- Masques: même dimensions que les images, avec valeurs d'intensité correspondant aux classes
- CSV (optionnel): colonne `filename` et `bethesda_category` pour l'entraînement supervisé

### Classes de Segmentation

Par défaut, le système reconnaît ces types cellulaires:
1. Fond (background)
2. Cellules inflammatoires
3. Cellules vésiculaires 
4. Cellules épithéliales
5. Colloïde
6. Artefacts

## Entraînement

Pour entraîner le modèle U-Net++ sur vos données de cytoponction:

```bash
python train.py --img_dir ./data/images --mask_dir ./data/masks --encoder resnet34 --loss_type combined
```

Arguments principaux:
- `--img_dir` : Dossier contenant les images cellulaires
- `--mask_dir` : Dossier contenant les masques de segmentation
- `--n_classes` : Nombre de classes pour la segmentation (défaut: 6)
- `--encoder` : Encodeur à utiliser (défaut: resnet34)
- `--pretrained` : Utiliser un encodeur préentraîné (activé par défaut)
- `--loss_type` : Type de perte (ce, dice, combined) (défaut: combined)
- `--optimizer` : Optimiseur à utiliser (adam, sgd, adamw) (défaut: adamw)
- `--scheduler` : Scheduler pour le taux d'apprentissage (plateau, cosine, step) (défaut: plateau)

## Analyse d'Images

Pour analyser des images de cytoponction thyroïdienne et générer des rapports:

```bash
python predict.py --img_dir ./data/test_images --model_path ./saved_models/unetpp_resnet34_6classes_best.pt
```

Arguments principaux:
- `--img_dir` : Dossier contenant les images à analyser
- `--model_path` : Chemin vers le modèle UNet++ entraîné
- `--n_classes` : Nombre de classes pour la segmentation (défaut: 6)
- `--encoder` : Encodeur utilisé dans le modèle (défaut: resnet34)
- `--output_dir` : Dossier où sauvegarder les résultats (défaut: results)
- `--no_visualization` : Ne pas générer de visualisations
- `--use_tiling` : Utiliser le tiling pour les images larges
- `--tile_size` : Taille des tuiles (défaut: 224)
- `--tile_overlap` : Chevauchement des tuiles en pixels (défaut: 32)

## Résultats et Visualisations

Le système génère trois types de sorties:

1. **Segmentations**: Masques colorés identifiant les différents composants cellulaires
2. **Rapports JSON**: Analyses quantitatives et classification Bethesda
3. **Visualisations**: Représentations graphiques combinant l'image, la segmentation et les résultats

### Format du Rapport

Le rapport JSON contient:
- Informations du patient
- Date d'analyse
- Composition cellulaire détaillée (types et pourcentages)
- Classification Bethesda (catégorie I-VI)
- Score de confiance
- Recommandations cliniques

## Classification Bethesda

Le système associe les caractéristiques suivantes aux catégories Bethesda:

- **Bethesda I**: Non diagnostique/Non satisfaisant (< 5% de cellules ou >30% d'artefacts)
- **Bethesda II**: Bénin (prédominance de cellules inflammatoires ou vésiculaires avec colloïde)
- **Bethesda III**: Atypie de signification indéterminée
- **Bethesda IV**: Néoplasie folliculaire (forte proportion de cellules vésiculaires, peu de colloïde)
- **Bethesda V**: Suspect de malignité
- **Bethesda VI**: Malin

## Évaluation du Modèle

Pour évaluer les performances d'un modèle entraîné:

```bash
python train.py --img_dir ./data/images --mask_dir ./data/masks --eval_only --model_path ./saved_models/unetpp_resnet34_6classes_best.pt
```

Les métriques calculées incluent:
- Dice score par classe
- Précision par pixel
- Matrices de confusion

## Encodeurs disponibles

U-Net++ supporte plusieurs encodeurs préentraînés, y compris:
- `resnet18`, `resnet34`, `resnet50`, etc.
- `efficientnet-b0` à `efficientnet-b7`
- `densenet121`, `densenet169`, etc.
- `vgg16`, `vgg19`
- `mobilenet_v2`

Pour changer d'encodeur, utilisez le paramètre `--encoder`. 