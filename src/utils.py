"""
Utilities -- Logging, Plotting, Grad-CAM, Checkpoints
=====================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import random


# ---- Reproducibility ----

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to {seed}")


# ---- Checkpoint helpers ----

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
    }, path)


def load_checkpoint(model, path, device='cpu'):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('metrics', {}), checkpoint.get('epoch', 0)


# ---- Plotting ----

def plot_training_curves(history, save_dir):
    """Plot training & validation loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, class_names, save_dir):
    """Plot confusion matrix as a heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={"size": 13})
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(fpr_dict, tpr_dict, auc_dict, class_names, save_dir):
    """Plot per-class and macro ROC curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, cls in enumerate(class_names):
        ax.plot(fpr_dict[i], tpr_dict[i],
                color=colors[i % len(colors)], linewidth=2,
                label=f'{cls} (AUC = {auc_dict[i]:.3f})')

    if 'macro' in auc_dict:
        ax.plot(fpr_dict['macro'], tpr_dict['macro'],
                color='navy', linewidth=2.5, linestyle='--',
                label=f'Macro (AUC = {auc_dict["macro"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves - One-vs-Rest', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_f1_scores(f1_per_class, class_names, save_dir):
    """Plot per-class F1 scores as a bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    bars = ax.bar(class_names, f1_per_class, color=colors[:len(class_names)],
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, f1_per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_xlabel('Cell Type', fontsize=13)
    ax.set_ylabel('F1 Score', fontsize=13)
    ax.set_title('Per-Class F1 Scores', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, 'f1_scores.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ---- Grad-CAM ----

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Targets the last Inception module (inception_5b) by default.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Default target: last inception module
        if target_layer is None:
            target_layer = model.inception_5b
        self.target_layer = target_layer

        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for input image."""
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)

        # Normalize
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), target_class


def generate_gradcam_grid(model, dataset, class_names, device, save_dir,
                          samples_per_class=4):
    """Generate a grid of Grad-CAM visualizations."""
    os.makedirs(save_dir, exist_ok=True)

    gradcam = GradCAM(model)

    # ImageNet denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Collect samples per class
    class_samples = {i: [] for i in range(len(class_names))}
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if len(class_samples[label]) < samples_per_class:
            class_samples[label].append(img)
        if all(len(v) >= samples_per_class for v in class_samples.values()):
            break

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, samples_per_class * 2,
                              figsize=(samples_per_class * 5, num_classes * 2.8))

    for cls_idx in range(num_classes):
        for sample_idx in range(samples_per_class):
            img = class_samples[cls_idx][sample_idx]
            input_tensor = img.unsqueeze(0).to(device)

            cam, pred_class = gradcam.generate(input_tensor)

            # Denormalize image for display
            img_display = img.cpu() * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)

            # Original image
            ax_orig = axes[cls_idx, sample_idx * 2]
            ax_orig.imshow(img_display)
            ax_orig.set_title(f'{class_names[cls_idx]}', fontsize=9)
            ax_orig.axis('off')

            # Grad-CAM overlay
            ax_cam = axes[cls_idx, sample_idx * 2 + 1]
            ax_cam.imshow(img_display)
            ax_cam.imshow(cam, cmap='jet', alpha=0.5)
            pred_label = class_names[pred_class]
            ax_cam.set_title(f'Pred: {pred_label}', fontsize=9)
            ax_cam.axis('off')

    plt.suptitle('Grad-CAM Visualizations (GoogLeNet)', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'gradcam_grid.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
