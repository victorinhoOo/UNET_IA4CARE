import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import torch.nn as nn

# Import des modules personnalisés
from model.unet import UNet, CombinedLoss
from dataloader.load_data import get_data_loaders, get_categorized_data_loaders, get_categorized_mask_loaders

# Fonction pour analyser les arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Entraînement de modèle pour cytoponctions thyroïdiennes')
    
    # Chemins des données
    parser.add_argument('--thumbnail_dir', type=str, default='categorized_thumbnail_windows_224_224', 
                        help='Dossier contenant les images catégorisées')
    parser.add_argument('--mask_dir', type=str, default='categorized_mask', 
                        help='Dossier contenant les masques (categorized_mask pour masques de classe ou binary_masks pour silhouettes blanches)')
    parser.add_argument('--output_dir', type=str, default='saved_models', 
                        help='Dossier pour sauvegarder les modèles')
    
    # Mode de fonctionnement
    parser.add_argument('--mode', type=str, default='segmentation', choices=['segmentation', 'classification', 'binary_segmentation'],
                        help='Mode d\'entraînement: segmentation (multiclasse), binary_segmentation (fond/silhouette) ou classification')
    
    # Paramètres du modèle
    parser.add_argument('--n_classes', type=int, default=None, 
                        help='Nombre de classes (déterminé automatiquement si non spécifié)')
    parser.add_argument('--encoder', type=str, default='resnet34', 
                        help='Encodeur à utiliser (resnet34, efficientnet-b0, etc.)')
    parser.add_argument('--pretrained', action='store_true', default=True, 
                        help='Utiliser un encodeur préentraîné')
    
    # Paramètres d'entraînement
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Taille des batches')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Taux d\'apprentissage initial')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Coefficient de régularisation L2')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience pour l\'early stopping')
    
    # Choix de l'optimiseur et du scheduler
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], 
                        help='Optimiseur à utiliser')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine', 'step'], 
                        help='Scheduler pour le taux d\'apprentissage')
    
    # Choix de la fonction de perte
    parser.add_argument('--loss_type', type=str, default='combined', choices=['ce', 'dice', 'combined', 'bce'], 
                        help='Type de fonction de perte à utiliser (bce pour segmentation binaire)')
    parser.add_argument('--weight_ce', type=float, default=0.5, 
                        help='Poids pour la loss CrossEntropy dans la perte combinée')
    parser.add_argument('--weight_dice', type=float, default=0.5, 
                        help='Poids pour la Dice Loss dans la perte combinée')
    
    # Autres paramètres
    parser.add_argument('--workers', type=int, default=4, 
                        help='Nombre de workers pour le chargement des données')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Seed pour la reproductibilité')
    parser.add_argument('--val_interval', type=int, default=1, 
                        help='Fréquence de validation (en époques)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Chemin vers un modèle à reprendre')
    parser.add_argument('--eval_only', action='store_true', 
                        help='Évaluer uniquement le modèle sans entraînement')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Chemin vers le modèle à évaluer (si eval_only=True)')
    parser.add_argument('--generate_masks', action='store_true', 
                        help='Générer les masques avant l\'entraînement')
    
    return parser.parse_args()

# Fonction pour sélectionner la fonction de perte
def get_loss_function(loss_type, n_classes, weight_ce=0.5, weight_dice=0.5):
    if loss_type == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        from model.unet import DiceLoss
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss(weight_ce=weight_ce, weight_dice=weight_dice)
    elif loss_type == 'bce':
        # Pour la segmentation binaire (silhouettes)
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Type de perte non reconnu: {loss_type}")

# Fonction pour sélectionner l'optimiseur
def get_optimizer(optimizer_name, model_parameters, lr, weight_decay):
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimiseur non reconnu: {optimizer_name}")

# Fonction pour sélectionner le scheduler
def get_scheduler(scheduler_name, optimizer, patience=10, T_max=10, step_size=30):
    if scheduler_name == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        raise ValueError(f"Scheduler non reconnu: {scheduler_name}")

# Fonction pour calculer le Dice Score (pour la segmentation)
def dice_score(pred, target, smooth=1e-6, n_classes=6):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    dice_scores = []
    for c in range(n_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)

# Fonction pour calculer la précision (pour la classification)
def accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    return (preds == targets).float().mean().item()

# Fonction pour visualiser les résultats de segmentation
def visualize_segmentation(image, mask, pred, epoch, output_dir='results/segmentation'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir en numpy
    image = image.cpu().numpy().transpose(1, 2, 0)
    # Dénormaliser l'image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    mask = mask.cpu().numpy()
    pred = torch.argmax(torch.softmax(pred, dim=0), dim=0).cpu().numpy()
    
    # Créer la figure
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Masque réel')
    plt.imshow(mask, cmap='viridis')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Prédiction')
    plt.imshow(pred, cmap='viridis')
    plt.axis('off')
    
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()

# Fonction pour visualiser les résultats de classification
def visualize_classification(images, targets, preds, class_names, epoch, output_dir='results/classification'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Prendre un maximum de 8 images pour la visualisation
    num_images = min(8, images.size(0))
    images = images[:num_images]
    targets = targets[:num_images]
    preds = preds[:num_images]
    
    # Créer une grille d'images
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, (img, target, pred) in enumerate(zip(images, targets, preds)):
        if i >= len(axes):
            break
            
        # Convertir et dénormaliser l'image
        img = img.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Afficher l'image et les étiquettes
        axes[i].imshow(img)
        title = f"Vraie: {class_names[target]}\nPrédite: {class_names[pred]}"
        axes[i].set_title(title, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()

# Fonction d'entraînement pour la segmentation
def train_segmentation_epoch(model, dataloader, criterion, optimizer, device, n_classes):
    model.train()
    running_loss = 0.0
    dice_scores = []
    
    pbar = tqdm(dataloader, desc='Entraînement')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        batch_dice = dice_score(outputs, masks, n_classes=n_classes)
        dice_scores.append(batch_dice)
        
        pbar.set_postfix({'loss': loss.item(), 'dice': batch_dice})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = np.mean(dice_scores)
    
    return epoch_loss, epoch_dice

# Fonction d'entraînement pour la classification
def train_classification_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Entraînement')
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Fonction d'évaluation pour la segmentation
def evaluate_segmentation(model, dataloader, criterion, device, n_classes, visualize=False, epoch=0):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(dataloader, desc='Évaluation')):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Statistiques
            running_loss += loss.item()
            batch_dice = dice_score(outputs, masks, n_classes=n_classes)
            dice_scores.append(batch_dice)
            
            # Visualiser le premier batch
            if visualize and i == 0:
                visualize_segmentation(images[0], masks[0], outputs[0], epoch)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = np.mean(dice_scores)
    
    return epoch_loss, epoch_dice

# Fonction d'évaluation pour la classification
def evaluate_classification(model, dataloader, criterion, device, class_names, visualize=False, epoch=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(dataloader, desc='Évaluation')):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Stocker les prédictions pour visualisation
            if visualize and i == 0:
                all_targets = targets.cpu()
                all_preds = predicted.cpu()
                visualize_images = images
        
        # Visualiser les résultats
        if visualize:
            visualize_classification(visualize_images, all_targets, all_preds, 
                                    class_names, epoch)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Fonction principale
def main():
    # Récupérer les arguments
    args = parse_args()
    
    # Définir le seed pour la reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Générer les masques si besoin
    if args.generate_masks:
        print("Génération des masques...")
        import subprocess
        subprocess.run(["python", "mask.py"])
        print("Masques générés avec succès.")
    
    # Détecter le GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger les données selon le mode
    if args.mode in ['segmentation', 'binary_segmentation']:
        print(f"Chargement des données de segmentation depuis {args.thumbnail_dir} et {args.mask_dir}...")
        train_loader, val_loader, test_loader, n_classes = get_categorized_mask_loaders(
            img_root_dir=args.thumbnail_dir,
            mask_root_dir=args.mask_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed
        )
        
        if args.mode == 'binary_segmentation':
            # Pour la segmentation binaire (silhouettes/fond), nous avons 2 classes
            n_classes = 2
            print("Mode segmentation binaire sélectionné. Utilisation de 2 classes (fond/silhouette).")
        elif args.n_classes is not None:
            n_classes = args.n_classes
            
        print(f"Nombre de classes pour la segmentation: {n_classes}")
    else:  # classification
        print(f"Chargement des données de classification depuis {args.thumbnail_dir}...")
        train_loader, val_loader, test_loader, categories = get_categorized_data_loaders(
            root_dir=args.thumbnail_dir,
            batch_size=args.batch_size,
            workers=args.workers,
            seed=args.seed
        )
        n_classes = len(categories)
        print(f"Nombre de classes pour la classification: {n_classes}")
        print(f"Catégories: {categories}")
    
    # Initialiser le modèle selon le mode
    if args.mode in ['segmentation', 'binary_segmentation']:
        print(f"Initialisation du modèle de segmentation UNet++ pour {n_classes} classes...")
        model = UNet(
            n_classes=n_classes,
            encoder=args.encoder,
            pretrained=args.pretrained
        )
    else:  # classification
        print("Initialisation du modèle de classification...")
        # Pour la classification, on utilise uniquement l'encodeur avec une tête de classification
        import torchvision.models as models
        
        if args.encoder == 'resnet34':
            model = models.resnet34(pretrained=args.pretrained)
            # Remplacer la dernière couche FC par une nouvelle adaptée au nombre de classes
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif args.encoder == 'efficientnet_b0':
            try:
                model = models.efficientnet_b0(pretrained=args.pretrained)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
            except:
                print("EfficientNet n'est pas disponible, utilisation de ResNet34 à la place.")
                model = models.resnet34(pretrained=args.pretrained)
                model.fc = nn.Linear(model.fc.in_features, n_classes)
        else:
            print(f"Encodeur {args.encoder} non reconnu pour la classification, utilisation de ResNet34.")
            model = models.resnet34(pretrained=args.pretrained)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
    
    print("Modèle initialisé.")
    model = model.to(device)
    print("Modèle déplacé vers", device)
    
    # Définir la fonction de perte selon le mode
    if args.mode == 'segmentation':
        criterion = get_loss_function(
            args.loss_type, 
            n_classes, 
            weight_ce=args.weight_ce, 
            weight_dice=args.weight_dice
        )
    elif args.mode == 'binary_segmentation':
        # Pour la segmentation binaire, utiliser BCEWithLogitsLoss
        criterion = get_loss_function('bce', n_classes)
    else:  # classification
        criterion = nn.CrossEntropyLoss()
    
    # Si on évalue seulement
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("Pour l'évaluation, spécifiez un chemin de modèle avec --model_path")
        
        # Charger le modèle
        model.load_state_dict(torch.load(args.model_path))
        
        # Évaluer sur l'ensemble de test
        if args.mode == 'segmentation':
            test_loss, test_metric = evaluate_segmentation(model, test_loader, criterion, device, n_classes)
            print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_metric:.4f}")
        else:  # classification
            test_loss, test_metric = evaluate_classification(model, test_loader, criterion, device, categories)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metric:.2f}%")
        
        return
    
    # Initialiser l'optimiseur
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    
    # Initialiser le scheduler
    scheduler = get_scheduler(
        args.scheduler, 
        optimizer, 
        patience=args.patience, 
        T_max=args.epochs // 10, 
        step_size=args.epochs // 3
    )
    
    # Reprendre l'entraînement si spécifié
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if args.mode == 'segmentation':
            best_metric = checkpoint['best_dice']
        else:  # classification
            best_metric = checkpoint['best_acc']
        print(f"Reprise de l'entraînement à l'époque {start_epoch}")
    
    # Boucle d'entraînement
    for epoch in range(start_epoch, args.epochs):
        print(f"\nÉpoque {epoch+1}/{args.epochs}")
        
        # Entraînement selon le mode
        if args.mode == 'segmentation':
            train_loss, train_metric = train_segmentation_epoch(model, train_loader, criterion, optimizer, device, n_classes)
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metric:.4f}")
        else:  # classification
            train_loss, train_metric = train_classification_epoch(model, train_loader, criterion, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metric:.2f}%")
        
        # Validation périodique
        if (epoch + 1) % args.val_interval == 0:
            if args.mode == 'segmentation':
                val_loss, val_metric = evaluate_segmentation(model, val_loader, criterion, device, n_classes, 
                                              visualize=True, epoch=epoch+1)
                print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metric:.4f}")
            else:  # classification
                val_loss, val_metric = evaluate_classification(model, val_loader, criterion, device, categories, 
                                              visualize=True, epoch=epoch+1)
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_metric:.2f}%")
            
            # Mettre à jour le scheduler
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Sauvegarder le meilleur modèle
            if val_metric > best_metric:
                best_metric = val_metric
                model_name = f"model_{args.mode}_{args.encoder}_{n_classes}classes_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'best_metric': best_metric
                }, os.path.join(args.output_dir, model_name))
                print(f"Meilleur modèle sauvegardé avec {'Dice' if args.mode == 'segmentation' else 'Accuracy'}: {best_metric:.4f}")
        
        # Sauvegarder le dernier modèle
        if (epoch + 1) % 10 == 0:
            model_name = f"model_{args.mode}_{args.encoder}_{n_classes}classes_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_metric': train_metric,
                'best_metric': best_metric
            }, os.path.join(args.output_dir, model_name))
    
    # Évaluation finale sur l'ensemble de test
    model_name = f"model_{args.mode}_{args.encoder}_{n_classes}classes_best.pt"
    model.load_state_dict(torch.load(os.path.join(args.output_dir, model_name), weights_only=False)['model_state_dict'])
    
    if args.mode == 'segmentation':
        test_loss, test_metric = evaluate_segmentation(model, test_loader, criterion, device, n_classes)
        print(f"\nTest Loss final: {test_loss:.4f}, Test Dice final: {test_metric:.4f}")
    else:  # classification
        test_loss, test_metric = evaluate_classification(model, test_loader, criterion, device, categories)
        print(f"\nTest Loss final: {test_loss:.4f}, Test Accuracy finale: {test_metric:.2f}%")

if __name__ == "__main__":
    main() 