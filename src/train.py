"""
Training Loop for GoogLeNet Blood Cell Classifier
===================================================
Handles main + auxiliary loss, cosine annealing, early stopping.
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device,
                    aux_loss_weight=0.3):
    """
    Train for one epoch with auxiliary loss.

    GoogLeNet returns (main_output, aux1_output, aux2_output) in training mode.
    Total loss = main_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False, ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward (returns main + 2 auxiliary outputs during training)
        main_out, aux1_out, aux2_out = model(images)

        # Compute losses
        main_loss = criterion(main_out, labels)
        aux1_loss = criterion(aux1_out, labels)
        aux2_loss = criterion(aux2_out, labels)

        # Total weighted loss
        total_loss = main_loss + aux_loss_weight * aux1_loss + aux_loss_weight * aux2_loss

        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += total_loss.item() * images.size(0)
        _, predicted = main_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model (no auxiliary outputs in eval mode)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Validating", leave=False, ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def run_training(model, train_loader, val_loader, config, device,
                 checkpoint_dir, log_dir):
    """
    Full training loop with cosine annealing, early stopping, and checkpointing.
    """
    import os
    import csv
    from src.utils import save_checkpoint

    epochs = config['training']['epochs']
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    aux_weight = config['training']['aux_loss_weight']
    patience = config['training']['early_stopping_patience']

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    print("=" * 60)
    print("  TRAINING GOOGLENET - Blood Cell Classifier")
    print("=" * 60)
    print(f"  Epochs:     {epochs}")
    print(f"  LR:         {lr}")
    print(f"  Optimizer:  Adam (wd={wd})")
    print(f"  Scheduler:  CosineAnnealing (T_max={epochs})")
    print(f"  Aux weight: {aux_weight}")
    print(f"  Patience:   {patience}")
    print("=" * 60)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    patience_counter = 0

    # CSV log
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                         'val_loss', 'val_acc', 'lr'])

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, aux_weight
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # CSV log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.2f}',
                             f'{val_loss:.4f}', f'{val_acc:.2f}',
                             f'{current_lr:.6f}'])

        # Print epoch summary
        print(f"\n  Epoch [{epoch:3d}/{epochs}]  "
              f"Train: {train_acc:.2f}% (loss={train_loss:.4f})  |  "
              f"Val: {val_acc:.2f}% (loss={val_loss:.4f})  |  "
              f"LR: {current_lr:.6f}  |  Time: {elapsed:.1f}s", end="")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_acc': val_acc, 'val_loss': val_loss},
                os.path.join(checkpoint_dir, 'best_model.pth')
            )
            print(" * Best!", end="")
        else:
            patience_counter += 1

        # Save last model
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_acc': val_acc, 'val_loss': val_loss},
            os.path.join(checkpoint_dir, 'last_model.pth')
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\n\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    print(f"\n\n  Training complete! Best Val Accuracy: {best_val_acc:.2f}%")
    return history
