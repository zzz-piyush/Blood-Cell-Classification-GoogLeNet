"""
Evaluation Module -- Metrics & Visualization
=============================================
Accuracy, F1, Precision, Recall, AUC-ROC, Confusion Matrix, Grad-CAM
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from src.utils import (
    load_checkpoint, plot_confusion_matrix, plot_roc_curves,
    plot_f1_scores, generate_gradcam_grid
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Run inference and collect all predictions, labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob, class_names):
    """Compute all evaluation metrics."""
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted'))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
    metrics['weighted_precision'] = float(precision_score(y_true, y_pred, average='weighted'))
    metrics['weighted_recall'] = float(recall_score(y_true, y_pred, average='weighted'))

    f1_per_class = f1_score(y_true, y_pred, average=None)
    metrics['f1_per_class'] = {class_names[i]: float(f1_per_class[i]) for i in range(len(class_names))}

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    num_classes = len(class_names)
    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    for i in range(num_classes):
        y_true_bin = (y_true == i).astype(int)
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin, y_prob[:, i])
        auc_dict[i] = float(auc(fpr_dict[i], tpr_dict[i]))

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_classes
    fpr_dict['macro'] = all_fpr
    tpr_dict['macro'] = mean_tpr
    auc_dict['macro'] = float(auc(all_fpr, mean_tpr))

    metrics['auc_per_class'] = {class_names[i]: auc_dict[i] for i in range(num_classes)}
    metrics['macro_auc'] = auc_dict['macro']

    return metrics, cm, fpr_dict, tpr_dict, auc_dict, f1_per_class


def run_evaluation(model, test_loader, class_names, device, config):
    """Full evaluation pipeline."""
    plot_dir = config['output']['plot_dir']
    metrics_dir = config['output']['metrics_dir']
    gradcam_dir = config['output']['gradcam_dir']

    print("\n" + "=" * 60)
    print("  EVALUATION - Test Set")
    print("=" * 60)

    y_pred, y_true, y_prob = get_predictions(model, test_loader, device)

    metrics, cm, fpr_dict, tpr_dict, auc_dict, f1_per_class = \
        compute_metrics(y_true, y_pred, y_prob, class_names)

    print(f"\n  Test Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
    print(f"  Macro AUC-ROC:       {metrics['macro_auc']:.4f}")
    print(f"  Weighted Precision:  {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall:     {metrics['weighted_recall']:.4f}")
    print("\n  Per-Class F1:")
    for cls, f1 in metrics['f1_per_class'].items():
        print(f"    {cls:15s}: {f1:.4f}")
    print("\n  Per-Class AUC:")
    for cls, auc_val in metrics['auc_per_class'].items():
        print(f"    {cls:15s}: {auc_val:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(f"\n  Classification Report:\n{report}")

    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    print("\n  Generating plots...")
    plot_confusion_matrix(cm, class_names, plot_dir)
    plot_roc_curves(fpr_dict, tpr_dict, auc_dict, class_names, plot_dir)
    plot_f1_scores(f1_per_class, class_names, plot_dir)

    print("\n  Generating Grad-CAM visualizations...")
    try:
        test_dataset = test_loader.dataset
        generate_gradcam_grid(model, test_dataset, class_names, device, gradcam_dir,
                              samples_per_class=config['evaluation']['gradcam_samples'])
    except Exception as e:
        print(f"  Warning: Grad-CAM failed: {e}")

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    return metrics
