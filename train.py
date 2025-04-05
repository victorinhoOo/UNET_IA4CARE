import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

# Import des modules personnalisés
from model.unet import UNet, CombinedLoss
from dataloader.load_data import get_data_loaders

# Fonction pour analyser les arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Entraînement du modèle UNet++ pour la segmentation de cytoponctions thyroïdiennes')
    
    # Chemins des données
    parser.add_argument('--img_dir', type=str, default='coco_export/thumbnail_224x224', 
                        help='Dossier contenant les images')
    parser.add_argument('--mask_dir', type=str, default='coco_export/masks_224x224', 
                        help='Dossier contenant les masques')
    parser.add_argument('--output_dir', type=str, default='saved_models', 
                        help='Dossier pour sauvegarder les modèles')
    
    # Paramètres du modèle
    parser.add_argument('--n_classes', type=int, default=6, 
                        help='Nombre de classes pour la segmentation')
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
    parser.add_argument('--loss_type', type=str, default='combined', choices=['ce', 'dice', 'combined'], 
                        help='Type de fonction de perte à utiliser')
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

# Fonction pour calculer le Dice Score
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

# Fonction pour visualiser les résultats
def visualize_results(image, mask, pred, epoch, output_dir='results/visualizations'):
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

# Fonction d'entraînement
def train_epoch(model, dataloader, criterion, optimizer, device, n_classes):
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

# Fonction d'évaluation
def evaluate(model, dataloader, criterion, device, n_classes, visualize=False, epoch=0):
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
                visualize_results(images[0], masks[0], outputs[0], epoch)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = np.mean(dice_scores)
    
    return epoch_loss, epoch_dice

# Fonction principale
def main():
    # Récupérer les arguments
    args = parse_args()
    
    # Définir le seed pour la reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Créer le dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Détecter le GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de: {device}")
    
    # Charger les données
    train_loader, val_loader, test_loader = get_data_loaders(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        n_classes=args.n_classes
    )
    
    # Initialiser le modèle
    model = UNet(
        n_classes=args.n_classes,
        encoder=args.encoder,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Définir la fonction de perte
    criterion = get_loss_function(
        args.loss_type, 
        args.n_classes, 
        weight_ce=args.weight_ce, 
        weight_dice=args.weight_dice
    )
    
    # Si on évalue seulement
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("Pour l'évaluation, spécifiez un chemin de modèle avec --model_path")
        
        # Charger le modèle
        model.load_state_dict(torch.load(args.model_path))
        
        # Évaluer sur l'ensemble de test
        test_loss, test_dice = evaluate(model, test_loader, criterion, device, args.n_classes)
        print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
        
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
    best_dice = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print(f"Reprise de l'entraînement à l'époque {start_epoch}")
    
    # Boucle d'entraînement
    for epoch in range(start_epoch, args.epochs):
        print(f"\nÉpoque {epoch+1}/{args.epochs}")
        
        # Entraînement
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device, args.n_classes)
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validation périodique
        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_dice = evaluate(model, val_loader, criterion, device, args.n_classes, 
                                          visualize=True, epoch=epoch+1)
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            # Mettre à jour le scheduler
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Sauvegarder le meilleur modèle
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'best_dice': best_dice
                }, os.path.join(args.output_dir, f"unetpp_{args.encoder}_{args.n_classes}classes_best.pt"))
                print(f"Meilleur modèle sauvegardé avec Dice: {best_dice:.4f}")
        
        # Sauvegarder le dernier modèle
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_dice': train_dice,
                'best_dice': best_dice
            }, os.path.join(args.output_dir, f"unetpp_{args.encoder}_{args.n_classes}classes_epoch{epoch+1}.pt"))
    
    # Évaluation finale sur l'ensemble de test
    model.load_state_dict(torch.load(os.path.join(args.output_dir, f"unetpp_{args.encoder}_{args.n_classes}classes_best.pt"))['model_state_dict'])
    test_loss, test_dice = evaluate(model, test_loader, criterion, device, args.n_classes)
    print(f"\nTest Loss final: {test_loss:.4f}, Test Dice final: {test_dice:.4f}")

if __name__ == "__main__":
    main() 